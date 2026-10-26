package execution

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
	"github.com/atlasfutures/semantic-router/rayline-core/tokencount"
)

// The default Rayline Local deployment serves Qwen3.8-27B. Its prepared Chat
// request is rendered with the model's chat template (Hugging Face snapshot
// 1d4bf0f2, chat_template.jinja) and counted with the embedded tokenizer, so
// its capacity fact is a count, not an estimate. This assumes the runtime
// really serves Qwen3.8-27B with that template; the core does not verify the
// served model's identity.
const qwen38RegistryModel = "qwen/qwen3.8-27b"

// The rendering follows vLLM's Chat path: tool-call arguments and tools are
// reloaded and re-serialized by the template's tojson (Python spelling:
// ", " / ": " separators, floats by repr, so 1e15 is 1000000000000000.0), a
// tool is spelt as vLLM's model_dump() hands it over ("description": null,
// "parameters": null, declared fields first), and text parts are joined with
// "\n" in the rendered text (a plain concatenation is charged too where it
// costs more). It
// takes the template's most expensive branch wherever the request does not
// pin one (xhigh reasoning instructions, a think block on every assistant
// turn) and adds a fixed margin per message and per tool.
const (
	qwenMarginTokens        = 16
	qwenMessageMarginTokens = 2
	qwenToolMarginTokens    = 8

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

// qwenPrompt is a rendered prompt and what its count must add.
type qwenPrompt struct {
	text     string
	messages int
	tools    int
	images   int64
	// extra is tokens charged beside the text: for a message of several
	// text parts, what concatenating them (a Hugging Face template) costs
	// beyond vLLM's "\n" join, when it costs more.
	extra int64
	// multipart reports a message whose parts vLLM joined with "\n".
	multipart bool
}

// qwenPromptTokens counts the prepared Chat body rendered with the Qwen3.8
// template, plus the fixed margins. Images are returned separately: the
// runtime expands each placeholder to a resolution-dependent token count.
// exact is false when the counter had to bound a segment by its bytes.
func qwenPromptTokens(body []byte) (int64, int64, bool, error) {
	counter, err := tokencount.Qwen38()
	if err != nil {
		return 0, 0, false, err
	}
	prompt, err := qwenRender(body, counter)
	if err != nil {
		return 0, 0, false, err
	}
	count, exact := counter.CountExact(prompt.text)
	tokens := int64(count) + prompt.extra + qwenMarginTokens + qwenMessageMarginTokens*int64(prompt.messages) + qwenToolMarginTokens*int64(prompt.tools)
	return tokens, prompt.images, exact, nil
}

// qwenRender renders the prepared Chat body as the template would, with a
// generation prompt.
func qwenRender(body []byte, counter *tokencount.BPE) (qwenPrompt, error) {
	var request struct {
		Messages []qwenMessage     `json:"messages"`
		Tools    []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		return qwenPrompt{}, err
	}
	if len(request.Messages) == 0 {
		return qwenPrompt{}, fmt.Errorf("no messages")
	}
	if request.Messages[0].Role == "developer" {
		request.Messages[0].Role = "system"
	}
	prompt := qwenPrompt{messages: len(request.Messages), tools: len(request.Tools)}
	contents := make([]string, len(request.Messages))
	for index, message := range request.Messages {
		text, glued, images, err := qwenContent(message.Content)
		if err != nil {
			return qwenPrompt{}, err
		}
		contents[index] = thinkingcontrol.PythonStrip(text)
		prompt.images += images
		if glued != text {
			// Pieces never cross the template's special tokens around a
			// content, so the two joins can be compared alone.
			prompt.multipart = true
			prompt.extra += max(int64(counter.Count(thinkingcontrol.PythonStrip(glued))-counter.Count(contents[index])), 0)
		}
	}
	var out strings.Builder
	if len(request.Tools) > 0 {
		out.WriteString("<|im_start|>system\n" + qwenReasoningXHigh + "\n\n" + qwenToolsHeader)
		for _, tool := range request.Tools {
			rendered, err := qwenToolSpelling(tool)
			if err != nil {
				return qwenPrompt{}, err
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
			out.WriteString("<|im_start|>assistant\n<think>\n" + thinkingcontrol.PythonStrip(reasoning) + "\n</think>\n\n" + content)
			for call, toolCall := range message.ToolCalls {
				switch {
				case call > 0:
					out.WriteString("\n<tool_call>\n<function=" + toolCall.Function.Name + ">\n")
				case thinkingcontrol.PythonStrip(content) != "":
					out.WriteString("\n\n<tool_call>\n<function=" + toolCall.Function.Name + ">\n")
				default:
					out.WriteString("<tool_call>\n<function=" + toolCall.Function.Name + ">\n")
				}
				writeQwenArguments(&out, toolCall.Function.Arguments)
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
	prompt.text = out.String()
	return prompt, nil
}

// qwenToolSpelling is the template's tojson of a tool as vLLM hands it over:
// pydantic's model_dump() of the request's tool (declared fields first, an
// absent one null, type defaulting to "function").
func qwenToolSpelling(raw json.RawMessage) (string, error) {
	tool, err := thinkingcontrol.PythonMembers(raw)
	if err != nil {
		return thinkingcontrol.PythonDumps(raw, 0, false)
	}
	replaced := map[string]string{}
	for _, member := range tool {
		if member.Key == "function" {
			if function, err := thinkingcontrol.PythonMembers([]byte(member.JSON)); err == nil {
				replaced["function"] = modelDump(function, []string{"name", "description", "parameters"}, nil, nil)
			}
		}
	}
	return modelDump(tool, []string{"type", "function"}, map[string]string{"type": `"function"`}, replaced), nil
}

// modelDump spells an object as pydantic's model_dump() then tojson would:
// declared fields first (their default, else null, when absent), then the
// rest in order.
func modelDump(members []thinkingcontrol.PythonMember, declared []string, defaults, replaced map[string]string) string {
	spelling := func(member thinkingcontrol.PythonMember) string {
		if value, ok := replaced[member.Key]; ok {
			return value
		}
		return member.JSON
	}
	parts := make([]string, 0, len(members)+len(declared))
	seen := map[string]bool{}
	for _, key := range declared {
		value, ok := defaults[key]
		if !ok {
			value = "null"
		}
		for _, member := range members {
			if member.Key == key {
				value = spelling(member)
			}
		}
		parts = append(parts, thinkingcontrol.PythonString(key)+": "+value)
		seen[key] = true
	}
	for _, member := range members {
		if !seen[member.Key] {
			parts = append(parts, thinkingcontrol.PythonString(member.Key)+": "+spelling(member))
		}
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

// qwenContent is the template's render_content of a message's content as
// vLLM hands it over: a string, or text parts joined with "\n", and vision
// placeholders. glued is the same parts concatenated, as a Hugging Face
// template renders a list.
func qwenContent(raw json.RawMessage) (text, glued string, images int64, err error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", "", 0, nil
	}
	if json.Unmarshal(raw, &text) == nil {
		return text, text, 0, nil
	}
	var parts []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", "", 0, fmt.Errorf("unsupported content shape")
	}
	rendered := make([]string, 0, len(parts))
	for _, part := range parts {
		var kind string
		_ = json.Unmarshal(part["type"], &kind)
		_, image := part["image_url"]
		_, video := part["video"]
		switch {
		case image || kind == "image" || kind == "image_url":
			rendered = append(rendered, "<|vision_start|><|image_pad|><|vision_end|>")
			images++
		case video || kind == "video":
			rendered = append(rendered, "<|vision_start|><|video_pad|><|vision_end|>")
			images++
		case part["text"] != nil:
			var value string
			if err := json.Unmarshal(part["text"], &value); err != nil {
				return "", "", 0, err
			}
			rendered = append(rendered, value)
		default:
			return "", "", 0, fmt.Errorf("unsupported content part %q", kind)
		}
	}
	return strings.Join(rendered, "\n"), strings.Join(rendered, ""), images, nil
}

// writeQwenArguments renders a call's arguments as the template's
// <parameter=...> blocks after vLLM reloads them: a string value verbatim,
// any other as tojson spells it.
func writeQwenArguments(out *strings.Builder, arguments string) {
	if thinkingcontrol.PythonStrip(arguments) == "" {
		return
	}
	members, err := thinkingcontrol.PythonMembers([]byte(arguments))
	if err != nil {
		// Not an object: the runtime cannot render it either; count it whole.
		out.WriteString("<parameter=arguments>\n" + arguments + "\n</parameter>\n")
		return
	}
	for _, member := range members {
		value := member.JSON
		if member.IsString {
			value = member.Str
		}
		out.WriteString("<parameter=" + member.Key + ">\n" + value + "\n</parameter>\n")
	}
}
