package execution

import (
	"encoding/json"
	"regexp"
	"sort"
	"strconv"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// A byte bound is caller_bounded only over a body the core fully
// understands: every top-level member, message member, input item, content
// part and tool type below is one that adds no prompt tokens beyond its own
// bytes and the template margin. Anything else (media, content the body only
// references, a template or role prefixes of the request's own, a
// runtime-specific member) may make the runtime's prompt larger than the
// bound, so the fact stays caller_estimated, and unboundedMember names the
// first such member for the log. Unknown fails toward estimated.
// The allowlist does not guard against a template that renders an allowed
// member more than once; that assumption is stated in README.md.

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
	// Claude Code sends a mid-conversation system message with its own
	// output_config (effort), a setting like the top-level one. It is
	// bounded only in that shape: on a system message, holding effort only.
	boundedMessagesMessageFields = setOf("role", "content", "output_config")
	boundedMessageOutputConfig   = setOf("effort")
	boundedMessagesParts         = setOf("text", "tool_use", "tool_result", "thinking", "redacted_thinking")

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

// unboundedMember is the first part of a prepared body for protocol that
// keeps the byte bound from being an upper bound on the runtime's count, as a
// path into the body ("/tools/4/type=web_search"), or "" when there is none.
func unboundedMember(protocol llmprotocol.WireFormat, body []byte) string {
	var top map[string]any
	if json.Unmarshal(body, &top) != nil {
		return "body is not JSON"
	}
	switch protocol {
	case llmprotocol.OpenAIChatV1:
		return firstOf(
			func() string { return membersIn("", top, boundedChatFields) },
			func() string {
				kwargs, isObject := top["chat_template_kwargs"].(map[string]any)
				if top["chat_template_kwargs"] == nil {
					return ""
				}
				if !isObject {
					return "/chat_template_kwargs"
				}
				return membersIn("/chat_template_kwargs", kwargs, boundedChatTemplateKwargs)
			},
			func() string { return toolsOf(top["tools"], "function") },
			func() string {
				return eachObject("/messages", top["messages"], func(path string, message map[string]any) string {
					return firstOf(
						func() string { return membersIn(path, message, boundedChatMessageFields) },
						func() string { return partsIn(path+"/content", message["content"], boundedTextParts) })
				})
			})
	case llmprotocol.AnthropicMessagesV1:
		return firstOf(
			func() string { return membersIn("", top, boundedMessagesFields) },
			func() string {
				diagnostics, isObject := top["diagnostics"].(map[string]any)
				if top["diagnostics"] == nil {
					return ""
				}
				if !isObject {
					return "/diagnostics"
				}
				return membersIn("/diagnostics", diagnostics, boundedMessagesDiagnostics)
			},
			func() string { return toolsOf(top["tools"], "", "custom") },
			func() string { return partsIn("/system", top["system"], boundedTextParts) },
			func() string {
				return eachObject("/messages", top["messages"], func(path string, message map[string]any) string {
					return firstOf(
						func() string { return membersIn(path, message, boundedMessagesMessageFields) },
						func() string {
							if message["output_config"] == nil {
								return ""
							}
							config, isObject := message["output_config"].(map[string]any)
							if message["role"] != "system" || !isObject {
								return path + "/output_config"
							}
							return membersIn(path+"/output_config", config, boundedMessageOutputConfig)
						},
						func() string { return partsIn(path+"/content", message["content"], boundedMessagesParts) },
						func() string {
							return eachPart(path+"/content", message["content"], func(path string, part map[string]any) string {
								if part["type"] != "tool_result" {
									return ""
								}
								return partsIn(path+"/content", part["content"], boundedTextParts)
							})
						})
				})
			})
	case llmprotocol.OpenAIResponsesV1:
		return firstOf(
			func() string { return membersIn("", top, boundedResponsesFields) },
			func() string {
				if top["instructions"] == nil || isString(top["instructions"]) {
					return ""
				}
				return "/instructions"
			},
			func() string { return toolsOf(top["tools"], "function", "custom", "namespace") },
			func() string {
				if isString(top["input"]) {
					return ""
				}
				return eachObject("/input", top["input"], boundedResponsesItem)
			})
	}
	return "protocol " + logName(string(protocol))
}

// firstOf runs checks in order and returns the first failure.
func firstOf(checks ...func() string) string {
	for _, check := range checks {
		if failure := check(); failure != "" {
			return failure
		}
	}
	return ""
}

func boundedResponsesItem(path string, item map[string]any) string {
	kind, _ := item["type"].(string)
	if kind == "" && item["role"] != nil {
		kind = "message"
	}
	fields, known := boundedResponsesItems[kind]
	if !known {
		return path + "/type=" + logName(kind)
	}
	if failure := membersIn(path, item, fields); failure != "" {
		return failure
	}
	switch kind {
	case "message":
		return partsIn(path+"/content", item["content"], boundedResponsesParts)
	case "function_call_output", "custom_tool_call_output":
		return partsIn(path+"/output", item["output"], boundedResponsesParts)
	case "reasoning":
		return firstOf(
			func() string { return partsIn(path+"/summary", item["summary"], setOf("summary_text")) },
			func() string { return partsIn(path+"/content", item["content"], setOf("reasoning_text")) })
	}
	return ""
}

// membersIn is the path of a member of object at path that is not allowed.
// Members are checked in sorted order so the reported one is stable.
func membersIn(path string, object map[string]any, allowed nameSet) string {
	keys := make([]string, 0, len(object))
	for key := range object {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if !allowed[key] {
			return path + "/" + logName(key)
		}
	}
	return ""
}

// logName is a member name or type from the request as it may enter the
// log: an identifier of at most 64 characters, else a fixed placeholder, so
// a request cannot write arbitrary text into the log.
func logName(name string) string {
	if name == "" {
		return "<none>"
	}
	if logNamePattern.MatchString(name) {
		return name
	}
	return "<name>"
}

var logNamePattern = regexp.MustCompile(`^[A-Za-z0-9_.-]{0,64}$`)

func isString(value any) bool {
	_, ok := value.(string)
	return ok
}

// eachObject is the first failure of ok over a list of objects at path, or
// path itself when value is not such a list.
func eachObject(path string, value any, ok func(string, map[string]any) string) string {
	list, isList := value.([]any)
	if !isList {
		return path
	}
	for index, entry := range list {
		at := path + "/" + strconv.Itoa(index)
		object, isObject := entry.(map[string]any)
		if !isObject {
			return at
		}
		if failure := ok(at, object); failure != "" {
			return failure
		}
	}
	return ""
}

// partsIn checks that content is absent, a string, or a list of parts whose
// types and members are all allowed.
func partsIn(path string, content any, allowed nameSet) string {
	if content == nil || isString(content) {
		return ""
	}
	return eachObject(path, content, func(at string, part map[string]any) string {
		kind, _ := part["type"].(string)
		if !allowed[kind] {
			return at + "/type=" + logName(kind)
		}
		return membersIn(at, part, boundedPartFields[kind])
	})
}

// eachPart applies ok to every part of a list content; a string is fine.
func eachPart(path string, content any, ok func(string, map[string]any) string) string {
	if content == nil || isString(content) {
		return ""
	}
	return eachObject(path, content, ok)
}

// toolsOf checks that tools is absent or a list of tools of the given types
// ("" for a tool without one). A namespace may hold only function tools: a
// runtime flattens it into one function per member (namespaceFlatteningBytes
// charges what that adds). As for a top-level tool, the other members of a
// tool are not checked: they render as part of the tool's JSON, which the
// bound counts.
func toolsOf(tools any, kinds ...string) string {
	if tools == nil {
		return ""
	}
	allowed := setOf(kinds...)
	return eachObject("/tools", tools, func(path string, tool map[string]any) string {
		kind, present := tool["type"].(string)
		if !present && tool["type"] != nil || !allowed[kind] {
			return path + "/type=" + logName(kind)
		}
		if kind != "namespace" {
			return ""
		}
		return eachObject(path+"/tools", tool["tools"], func(path string, member map[string]any) string {
			if member["type"] != "function" {
				kind, _ := member["type"].(string)
				return path + "/type=" + logName(kind)
			}
			return ""
		})
	})
}
