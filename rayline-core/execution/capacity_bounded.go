package execution

import (
	"encoding/json"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// A byte bound is caller_bounded only over a body the core fully
// understands: every top-level member, message member, input item, content
// part and tool type below is one that adds no prompt tokens beyond its own
// bytes and the template margin. Anything else (media, content the body only
// references, a template or role prefixes of the request's own, a
// runtime-specific member) may make the runtime's prompt larger than the
// bound, so the fact stays caller_estimated. Unknown fails toward estimated.

type nameSet map[string]bool

func setOf(names ...string) nameSet {
	result := make(nameSet, len(names))
	for _, name := range names {
		result[name] = true
	}
	return result
}

var (
	boundedChatFields = setOf("model", "messages", "tools", "tool_choice", "parallel_tool_calls",
		"stream", "stream_options", "max_tokens", "max_completion_tokens", "temperature", "top_p",
		"top_k", "min_p", "stop", "seed", "presence_penalty", "frequency_penalty",
		"repetition_penalty", "logprobs", "top_logprobs", "user", "metadata", "response_format",
		"reasoning_effort", "reasoning", "chat_template_kwargs", "prompt_cache_key", "store", "n",
		"prompt_cache_retention")
	boundedChatMessageFields = setOf("role", "content", "name", "tool_calls", "tool_call_id", "id",
		"reasoning_content", "reasoning", "reasoning_details", "refusal")
	// Template switches only: a template that renders them is the served one.
	boundedChatTemplateKwargs = setOf("enable_thinking", "thinking")

	boundedMessagesFields = setOf("model", "messages", "system", "tools", "tool_choice",
		"max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "stream", "metadata",
		"thinking", "output_config", "cache_control", "context_management", "diagnostics")
	boundedMessagesDiagnostics = setOf("previous_message_id")
	boundedMessagesParts       = setOf("text", "tool_use", "tool_result", "thinking", "redacted_thinking")

	boundedResponsesFields = setOf("model", "input", "instructions", "tools", "tool_choice",
		"parallel_tool_calls", "max_output_tokens", "temperature", "top_p", "stream", "reasoning",
		"text", "store", "metadata", "user", "include", "prompt_cache_key", "client_metadata",
		"stream_options", "prompt_cache_retention")
	// The members each input item type may carry; a role or content on a
	// call item, for one, is something a runtime may still read.
	boundedResponsesItems = map[string]nameSet{
		"message":                 setOf("type", "role", "content", "id", "status"),
		"function_call":           setOf("type", "id", "call_id", "name", "arguments", "status"),
		"function_call_output":    setOf("type", "id", "call_id", "output", "status"),
		"custom_tool_call":        setOf("type", "id", "call_id", "name", "input", "status"),
		"custom_tool_call_output": setOf("type", "id", "call_id", "output", "status"),
		"reasoning":               setOf("type", "id", "summary", "content", "encrypted_content", "status"),
	}
	boundedResponsesParts = setOf("input_text", "output_text", "refusal")

	boundedTextParts = setOf("text")

	// The members each content part type may carry: vLLM, for one, reads an
	// image_url on a part whatever its type says.
	boundedPartFields = map[string]nameSet{
		"text":              setOf("type", "text", "cache_control", "citations"),
		"tool_use":          setOf("type", "id", "name", "input", "cache_control", "caller"),
		"tool_result":       setOf("type", "tool_use_id", "content", "is_error", "cache_control"),
		"thinking":          setOf("type", "thinking", "signature"),
		"redacted_thinking": setOf("type", "data"),
		"input_text":        setOf("type", "text"),
		"output_text":       setOf("type", "text", "annotations", "logprobs"),
		"refusal":           setOf("type", "refusal"),
		"summary_text":      setOf("type", "text"),
		"reasoning_text":    setOf("type", "text"),
	}
)

// boundedBody reports whether the byte bound over a prepared body for
// protocol is an upper bound on the runtime's count.
func boundedBody(protocol llmprotocol.WireFormat, body []byte) bool {
	var top map[string]any
	if json.Unmarshal(body, &top) != nil {
		return false
	}
	switch protocol {
	case llmprotocol.OpenAIChatV1:
		kwargs, _ := top["chat_template_kwargs"].(map[string]any)
		return membersIn(top, boundedChatFields) &&
			(top["chat_template_kwargs"] == nil || kwargs != nil && membersIn(kwargs, boundedChatTemplateKwargs)) &&
			toolsOf(top["tools"], "function") &&
			eachObject(top["messages"], func(message map[string]any) bool {
				return membersIn(message, boundedChatMessageFields) && partsIn(message["content"], boundedTextParts)
			})
	case llmprotocol.AnthropicMessagesV1:
		diagnostics, _ := top["diagnostics"].(map[string]any)
		return membersIn(top, boundedMessagesFields) &&
			(top["diagnostics"] == nil || diagnostics != nil && membersIn(diagnostics, boundedMessagesDiagnostics)) &&
			toolsOf(top["tools"], "", "custom") &&
			partsIn(top["system"], boundedTextParts) &&
			eachObject(top["messages"], func(message map[string]any) bool {
				return membersIn(message, setOf("role", "content")) && partsIn(message["content"], boundedMessagesParts) &&
					eachPart(message["content"], func(part map[string]any) bool {
						return part["type"] != "tool_result" || partsIn(part["content"], boundedTextParts)
					})
			})
	case llmprotocol.OpenAIResponsesV1:
		return membersIn(top, boundedResponsesFields) &&
			(top["instructions"] == nil || isString(top["instructions"])) &&
			toolsOf(top["tools"], "function", "custom") &&
			(isString(top["input"]) || eachObject(top["input"], boundedResponsesItem))
	}
	return false
}

func boundedResponsesItem(item map[string]any) bool {
	kind, _ := item["type"].(string)
	if kind == "" && item["role"] != nil {
		kind = "message"
	}
	fields, known := boundedResponsesItems[kind]
	if !known || !membersIn(item, fields) {
		return false
	}
	switch kind {
	case "message":
		return partsIn(item["content"], boundedResponsesParts)
	case "function_call_output", "custom_tool_call_output":
		return partsIn(item["output"], boundedResponsesParts)
	case "reasoning":
		return partsIn(item["summary"], setOf("summary_text")) && partsIn(item["content"], setOf("reasoning_text"))
	}
	return true
}

func membersIn(object map[string]any, allowed nameSet) bool {
	for key := range object {
		if !allowed[key] {
			return false
		}
	}
	return true
}

func isString(value any) bool {
	_, ok := value.(string)
	return ok
}

// eachObject reports whether value is a list of objects that all satisfy ok.
func eachObject(value any, ok func(map[string]any) bool) bool {
	list, isList := value.([]any)
	if !isList {
		return false
	}
	for _, entry := range list {
		object, isObject := entry.(map[string]any)
		if !isObject || !ok(object) {
			return false
		}
	}
	return true
}

// partsIn reports whether content is absent, a string, or a list of parts
// whose types are all allowed.
func partsIn(content any, allowed nameSet) bool {
	if content == nil || isString(content) {
		return true
	}
	return eachObject(content, func(part map[string]any) bool {
		kind, _ := part["type"].(string)
		return allowed[kind] && membersIn(part, boundedPartFields[kind])
	})
}

// eachPart applies ok to every part of a list content; a string is fine.
func eachPart(content any, ok func(map[string]any) bool) bool {
	if content == nil || isString(content) {
		return true
	}
	return eachObject(content, ok)
}

// toolsOf reports whether tools is absent or a list of tools of the given
// types ("" for a tool without one).
func toolsOf(tools any, kinds ...string) bool {
	if tools == nil {
		return true
	}
	allowed := setOf(kinds...)
	return eachObject(tools, func(tool map[string]any) bool {
		kind, present := tool["type"].(string)
		if !present && tool["type"] != nil {
			return false
		}
		return allowed[kind]
	})
}
