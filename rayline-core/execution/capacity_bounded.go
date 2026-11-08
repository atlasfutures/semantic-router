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
		"reasoning_effort", "reasoning", "chat_template_kwargs", "prompt_cache_key")
	boundedChatMessageFields = setOf("role", "content", "name", "tool_calls", "tool_call_id",
		"reasoning_content", "reasoning", "reasoning_details", "refusal")
	// Template switches only: a template that renders them is the served one.
	boundedChatTemplateKwargs = setOf("enable_thinking", "thinking")

	boundedMessagesFields = setOf("model", "messages", "system", "tools", "tool_choice",
		"max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "stream", "metadata",
		"thinking", "output_config")
	boundedMessagesParts = setOf("text", "tool_use", "tool_result", "thinking", "redacted_thinking")

	boundedResponsesFields = setOf("model", "input", "instructions", "tools", "tool_choice",
		"parallel_tool_calls", "max_output_tokens", "temperature", "top_p", "stream", "reasoning",
		"text", "store", "metadata", "user", "include", "prompt_cache_key")
	boundedResponsesItems = setOf("message", "function_call", "function_call_output", "reasoning",
		"custom_tool_call", "custom_tool_call_output")
	boundedResponsesMessageFields = setOf("type", "role", "content", "id", "status")
	boundedResponsesParts         = setOf("input_text", "output_text", "refusal")

	boundedTextParts = setOf("text")
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
		return membersIn(top, boundedMessagesFields) &&
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
	switch {
	case !boundedResponsesItems[kind]:
		return false
	case kind == "message":
		return membersIn(item, boundedResponsesMessageFields) && partsIn(item["content"], boundedResponsesParts)
	case kind == "function_call_output" || kind == "custom_tool_call_output":
		return partsIn(item["output"], boundedResponsesParts)
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
		return allowed[kind]
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
