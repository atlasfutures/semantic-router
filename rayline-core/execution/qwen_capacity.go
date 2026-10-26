package execution

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/tokencount"
)

// The default Rayline Local deployment serves Qwen3.8-27B. Its prepared Chat
// request is rendered with the model's chat template (Hugging Face snapshot
// 1d4bf0f2, chat_template.jinja) and counted with the embedded tokenizer, so
// its capacity fact is a count, not an estimate. This assumes the runtime
// really serves Qwen3.8-27B with that template; the core does not verify the
// served model's identity.
const qwen38RegistryModel = "qwen/qwen3.8-27b"

// The rendering takes the template's most expensive branch wherever the
// request does not pin one (xhigh reasoning instructions, a think block on
// every assistant turn) and adds a fixed margin per message and per tool for
// formatting the core does not reproduce byte for byte (Python's tojson of a
// float, an argument value's JSON spelling).
const (
	qwenMarginTokens        = 16
	qwenMessageMarginTokens = 2
	qwenToolMarginTokens    = 4

	qwenReasoningXHigh = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
	qwenToolsHeader    = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
	qwenToolsFooter    = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)

func isQwen38Deployment(deployment Deployment) bool {
	return deployment.Kind == "local" && strings.EqualFold(deployment.RegistryModel, qwen38RegistryModel) && deployment.Protocol == llmprotocol.OpenAIChatV1
}

type qwenMessage struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	ReasoningContent *string         `json:"reasoning_content"`
	Reasoning        *string         `json:"reasoning"`
	ToolCalls        []struct {
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

// qwenPromptTokens counts the prepared Chat body rendered with the Qwen3.8
// template, plus the fixed margins. Images are returned separately: the
// runtime expands each placeholder to a resolution-dependent token count.
func qwenPromptTokens(body []byte) (int64, int64, error) {
	rendered, messages, tools, images, err := qwenRender(body)
	if err != nil {
		return 0, 0, err
	}
	counter, err := tokencount.Qwen38()
	if err != nil {
		return 0, 0, err
	}
	tokens := int64(counter.Count(rendered)) + qwenMarginTokens + qwenMessageMarginTokens*int64(messages) + qwenToolMarginTokens*int64(tools)
	return tokens, images, nil
}

// qwenRender renders the prepared Chat body as the template would, with a
// generation prompt.
func qwenRender(body []byte) (string, int, int, int64, error) {
	var request struct {
		Messages []qwenMessage     `json:"messages"`
		Tools    []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		return "", 0, 0, 0, err
	}
	if len(request.Messages) == 0 {
		return "", 0, 0, 0, fmt.Errorf("no messages")
	}
	if request.Messages[0].Role == "developer" {
		request.Messages[0].Role = "system"
	}
	contents := make([]string, len(request.Messages))
	var images int64
	for index, message := range request.Messages {
		text, count, err := qwenContent(message.Content)
		if err != nil {
			return "", 0, 0, 0, err
		}
		contents[index], images = strings.TrimSpace(text), images+count
	}
	var out strings.Builder
	if len(request.Tools) > 0 {
		out.WriteString("<|im_start|>system\n" + qwenReasoningXHigh + "\n\n" + qwenToolsHeader)
		for _, tool := range request.Tools {
			rendered, err := pythonJSON(tool)
			if err != nil {
				return "", 0, 0, 0, err
			}
			out.WriteString("\n" + rendered)
		}
		out.WriteString(qwenToolsFooter)
		if request.Messages[0].Role == "system" && contents[0] != "" {
			out.WriteString("\n\n" + contents[0])
		}
		out.WriteString("<|im_end|>\n")
	} else if request.Messages[0].Role == "system" && contents[0] != "" {
		out.WriteString("<|im_start|>system\n" + qwenReasoningXHigh + "\n\n" + contents[0] + "<|im_end|>\n")
	} else {
		out.WriteString("<|im_start|>system\n" + qwenReasoningXHigh + "<|im_end|>\n")
	}
	for index, message := range request.Messages {
		content := contents[index]
		switch message.Role {
		case "system", "developer":
			if index > 0 || message.Role == "developer" {
				// The template refuses a late system message; count it as a
				// user turn rather than drop it.
				out.WriteString("<|im_start|>user\n" + content + "<|im_end|>\n")
			}
		case "assistant":
			reasoning := ""
			if message.ReasoningContent != nil {
				reasoning = *message.ReasoningContent
			} else if message.Reasoning != nil {
				reasoning = *message.Reasoning
			}
			out.WriteString("<|im_start|>assistant\n<think>\n" + strings.TrimSpace(reasoning) + "\n</think>\n\n" + content)
			for call, toolCall := range message.ToolCalls {
				switch {
				case call > 0:
					out.WriteString("\n<tool_call>\n<function=" + toolCall.Function.Name + ">\n")
				case strings.TrimSpace(content) != "":
					out.WriteString("\n\n<tool_call>\n<function=" + toolCall.Function.Name + ">\n")
				default:
					out.WriteString("<tool_call>\n<function=" + toolCall.Function.Name + ">\n")
				}
				if err := writeQwenArguments(&out, toolCall.Function.Arguments); err != nil {
					return "", 0, 0, 0, err
				}
				out.WriteString("</function>\n</tool_call>")
			}
			out.WriteString("<|im_end|>\n")
		case "tool":
			if index == 0 || request.Messages[index-1].Role != "tool" {
				out.WriteString("<|im_start|>user")
			}
			out.WriteString("\n<tool_response>\n" + content + "\n</tool_response>")
			if index == len(request.Messages)-1 || request.Messages[index+1].Role != "tool" {
				out.WriteString("<|im_end|>\n")
			}
		default:
			out.WriteString("<|im_start|>" + message.Role + "\n" + content + "<|im_end|>\n")
		}
	}
	out.WriteString("<|im_start|>assistant\n<think>\n")
	return out.String(), len(request.Messages), len(request.Tools), images, nil
}

// qwenContent is the template's render_content: a string, or text parts and
// vision placeholders.
func qwenContent(raw json.RawMessage) (string, int64, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", 0, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, 0, nil
	}
	var parts []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", 0, fmt.Errorf("unsupported content shape")
	}
	var out strings.Builder
	var images int64
	for _, part := range parts {
		var kind string
		_ = json.Unmarshal(part["type"], &kind)
		_, image := part["image_url"]
		_, video := part["video"]
		switch {
		case image || kind == "image" || kind == "image_url":
			out.WriteString("<|vision_start|><|image_pad|><|vision_end|>")
			images++
		case video || kind == "video":
			out.WriteString("<|vision_start|><|video_pad|><|vision_end|>")
			images++
		case part["text"] != nil:
			var value string
			if err := json.Unmarshal(part["text"], &value); err != nil {
				return "", 0, err
			}
			out.WriteString(value)
		default:
			return "", 0, fmt.Errorf("unsupported content part %q", kind)
		}
	}
	return out.String(), images, nil
}

// writeQwenArguments renders a call's arguments object as the template's
// <parameter=...> blocks: a string value verbatim, any other as JSON.
func writeQwenArguments(out *strings.Builder, arguments string) error {
	if strings.TrimSpace(arguments) == "" {
		return nil
	}
	decoder := json.NewDecoder(strings.NewReader(arguments))
	decoder.UseNumber()
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		// Not an object: the runtime cannot render it either; count it whole.
		out.WriteString("<parameter=arguments>\n" + arguments + "\n</parameter>\n")
		return nil
	}
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return err
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return err
		}
		rendered := ""
		var text string
		if json.Unmarshal(value, &text) == nil {
			rendered = text
		} else if rendered, err = pythonJSON(value); err != nil {
			return err
		}
		out.WriteString("<parameter=" + fmt.Sprint(key) + ">\n" + rendered + "\n</parameter>\n")
	}
	return nil
}

// pythonJSON spells a JSON value as Python's json.dumps(ensure_ascii=False)
// does (", " and ": " separators, key order kept), which the template's
// tojson filter uses.
func pythonJSON(raw json.RawMessage) (string, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var out strings.Builder
	var write func() error
	write = func() error {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		switch value := token.(type) {
		case json.Delim:
			closing, separator := "}", ", "
			if value == '[' {
				closing = "]"
			}
			out.WriteString(string(value))
			for first := true; decoder.More(); first = false {
				if !first {
					out.WriteString(separator)
				}
				if value == '{' {
					key, err := decoder.Token()
					if err != nil {
						return err
					}
					out.WriteString(pythonString(key.(string)) + ": ")
				}
				if err := write(); err != nil {
					return err
				}
			}
			if _, err := decoder.Token(); err != nil {
				return err
			}
			out.WriteString(closing)
		case string:
			out.WriteString(pythonString(value))
		case json.Number:
			out.WriteString(value.String())
		case bool:
			out.WriteString(strconv.FormatBool(value))
		case nil:
			out.WriteString("null")
		}
		return nil
	}
	if err := write(); err != nil && err != io.EOF {
		return "", err
	}
	return out.String(), nil
}

func pythonString(value string) string {
	var out strings.Builder
	out.WriteByte('"')
	for _, char := range value {
		switch char {
		case '"':
			out.WriteString(`\"`)
		case '\\':
			out.WriteString(`\\`)
		case '\n':
			out.WriteString(`\n`)
		case '\r':
			out.WriteString(`\r`)
		case '\t':
			out.WriteString(`\t`)
		case '\b':
			out.WriteString(`\b`)
		case '\f':
			out.WriteString(`\f`)
		default:
			if char < 0x20 {
				fmt.Fprintf(&out, `\u%04x`, char)
			} else {
				out.WriteRune(char)
			}
		}
	}
	out.WriteByte('"')
	return out.String()
}
