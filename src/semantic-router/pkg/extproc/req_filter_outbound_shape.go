package extproc

import (
	"bytes"
	"encoding/json"
	"sort"
	"strconv"
)

// outboundShape is the generic half of the outbound record: one pass over the
// final provider body that names every parameter it carries, so a parameter
// the named fields do not model is still on the record. It is content-free by
// construction:
//
//   - message, input, instruction, system, prompt and tool bodies are recorded
//     only as counts or lengths, never read further, never digested;
//   - a string is recorded verbatim only when it is a protocol-defined enum
//     value, otherwise as its sha256 prefix and length;
//   - a key is recorded verbatim only when it is a known request parameter,
//     otherwise as a digest, so a client-chosen key cannot publish a value;
//   - arrays are a count, an element kind and their distinct element shapes
//     under these same rules;
//   - objects are recursed to outboundShapeDepth, then only their key count;
//   - every object and array value list is bounded, and the whole pass stops
//     recording values after outboundShapeBudget of them.
//
// Every key is recorded, so an unmodeled parameter appears (as a digest key if
// it is unknown) rather than being silently absent.
func outboundShape(body []byte) map[string]interface{} {
	var top map[string]json.RawMessage
	if json.Unmarshal(body, &top) != nil || top == nil {
		return nil
	}
	shaper := &outboundShaper{remaining: outboundShapeBudget}
	shape := map[string]interface{}{}
	for _, key := range shapeKeyOrder(top) {
		if len(shape) >= outboundShapeKeyLimit {
			shape["+more"] = len(top) - outboundShapeKeyLimit
			break
		}
		value := top[key]
		if outboundContentKeys[key] {
			shape[key] = contentShape(value)
			continue
		}
		shape[shapeKey(key)] = shaper.describe(value, 1)
	}
	return shape
}

// outboundShaper carries the pass's remaining value budget.
type outboundShaper struct {
	remaining int
}

const (
	outboundShapeDepth    = 5
	outboundShapeKeyLimit = 48
	outboundShapeBudget   = 256
)

// outboundContentKeys hold content: counted, never read.
var outboundContentKeys = enumSet("messages", "input", "instructions", "system", "prompt", "contents", "tools",
	"functions", "documents", "suffix")

// shapeKeyOrder puts known keys first, then the rest, each sorted, so unknown
// keys cannot crowd a known parameter out of the bounded record.
func shapeKeyOrder(object map[string]json.RawMessage) []string {
	keys := make([]string, 0, len(object))
	for key := range object {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if outboundKnownKeys[keys[i]] != outboundKnownKeys[keys[j]] {
			return outboundKnownKeys[keys[i]]
		}
		return keys[i] < keys[j]
	})
	return keys
}

func shapeKey(key string) string {
	if outboundKnownKeys[key] {
		return key
	}
	return outboundDigest(key)
}

func contentShape(raw json.RawMessage) interface{} {
	var list []json.RawMessage
	if json.Unmarshal(raw, &list) == nil && list != nil {
		return map[string]interface{}{"count": len(list)}
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return map[string]interface{}{"length": len(text)}
	}
	return map[string]interface{}{"type": jsonKind(raw)}
}

func (shaper *outboundShaper) describe(raw json.RawMessage, depth int) interface{} {
	if shaper.remaining <= 0 {
		return "over_budget"
	}
	shaper.remaining--
	switch jsonKind(raw) {
	case "null":
		return nil
	case "bool":
		return string(bytes.TrimSpace(raw)) == "true"
	case "number":
		number, err := strconv.ParseFloat(string(bytes.TrimSpace(raw)), 64)
		if err != nil {
			return "number"
		}
		return number
	case "string":
		var text string
		_ = json.Unmarshal(raw, &text)
		return shapeString(text)
	case "array":
		return shaper.array(raw, depth)
	case "object":
		var object map[string]json.RawMessage
		if json.Unmarshal(raw, &object) != nil {
			return "object"
		}
		if depth >= outboundShapeDepth {
			return map[string]interface{}{"keys": len(object)}
		}
		shape := map[string]interface{}{}
		for _, key := range shapeKeyOrder(object) {
			if len(shape) >= outboundListLimit {
				shape["+more"] = len(object) - outboundListLimit
				break
			}
			if outboundContentKeys[key] {
				shape[key] = contentShape(object[key])
				continue
			}
			shape[shapeKey(key)] = shaper.describe(object[key], depth+1)
		}
		return shape
	}
	return "unknown"
}

// shapeString is a known enum value verbatim, anything else as a digest and
// its length.
func shapeString(text string) interface{} {
	if outboundKnownValues[text] {
		return text
	}
	return map[string]interface{}{"sha256": outboundDigest(text), "length": len(text)}
}

// array is the count, the element kind and the distinct element shapes
// (bounded): strings under the string rule, objects described like any
// object one level deeper, so an array of parameter objects (context edits,
// hosted tools' options) keeps its members.
func (shaper *outboundShaper) array(raw json.RawMessage, depth int) interface{} {
	var list []json.RawMessage
	if json.Unmarshal(raw, &list) != nil {
		return "array"
	}
	shape := map[string]interface{}{"count": len(list)}
	if len(list) == 0 {
		return shape
	}
	kind := jsonKind(list[0])
	seen := map[string]bool{}
	values := []interface{}{}
	for _, element := range list {
		if jsonKind(element) != kind {
			kind = "mixed"
		}
		if len(values) == outboundListLimit || shaper.remaining <= 0 {
			values = append(values, "+more")
			break
		}
		value := shaper.describe(element, depth+1)
		encoded, _ := json.Marshal(value)
		if seen[string(encoded)] {
			continue
		}
		seen[string(encoded)] = true
		values = append(values, value)
	}
	shape["element"] = kind
	shape["values"] = values
	return shape
}

func jsonKind(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return "null"
	}
	switch trimmed[0] {
	case 'n':
		return "null"
	case 't', 'f':
		return "bool"
	case '"':
		return "string"
	case '[':
		return "array"
	case '{':
		return "object"
	}
	return "number"
}

// outboundKnownKeys are request parameter names the Chat, Responses and
// Messages APIs, OpenRouter and the catalog's reasoning families define. They
// decide only whether a key is spelled out; an unknown key is still recorded.
var outboundKnownKeys = enumSet(
	// shared and Chat
	"model", "max_tokens", "max_completion_tokens", "max_output_tokens", "temperature", "top_p", "top_k", "min_p",
	"seed", "n", "stop", "stop_sequences", "stream", "stream_options", "include_usage", "include_obfuscation",
	"frequency_penalty", "presence_penalty", "repetition_penalty", "logit_bias", "logprobs", "top_logprobs",
	"tool_choice", "parallel_tool_calls", "disable_parallel_tool_use", "response_format", "json_schema", "schema",
	"strict", "name", "type", "function", "user", "metadata", "store", "auto_store", "service_tier", "verbosity",
	"modalities", "audio", "prediction", "web_search_options", "reasoning_effort", "reasoning_budget_tokens",
	"prompt_cache_key", "prompt_cache_retention", "safety_identifier", "transforms", "models", "route", "usage",
	// Responses
	"reasoning", "effort", "summary", "generate_summary", "text", "format", "truncation", "include",
	"previous_response_id", "conversation", "id", "background", "max_tool_calls", "prompt_template",
	// Messages
	"thinking", "budget_tokens", "display", "output_config", "task_budget", "cache_control", "ttl", "scope",
	"evict_on_complete", "context_management", "edits", "trigger", "keep", "clear_at_least", "exclude_tools",
	"clear_tool_inputs", "container", "mcp_servers", "betas",
	// OpenRouter provider routing
	"provider", "order", "only", "ignore", "allow_fallbacks", "require_parameters", "data_collection", "max_price",
	"completion", "request", "image", "quantizations", "sort", "zdr", "enforce_distillable_text",
	// vLLM chat template and catalog reasoning switches
	"chat_template_kwargs", "enable_thinking", "thinking_mode", "reasoning_strength", "low_effort",
	"medium_effort", "high_effort", "max_tokens_per_turn",
)

// outboundKnownValues are enum values those parameters define. A string
// outside them is digested.
var outboundKnownValues = enumSet(
	"none", "minimal", "low", "medium", "high", "xhigh", "max",
	"enabled", "disabled", "adaptive", "auto", "summarized", "omitted", "concise", "detailed",
	"text", "json_object", "json_schema", "required", "any", "tool", "function", "allowed_tools", "custom",
	"ephemeral", "5m", "1h", "in_memory", "24h", "allow", "deny", "default", "flex", "priority", "scale",
	"price", "throughput", "latency", "audio", "image", "input_tokens", "tool_uses",
	"reasoning.encrypted_content", "file_search_call.results", "web_search_call.results",
	"web_search_call.action.sources", "message.input_image.image_url", "computer_call_output.output.image_url",
	"code_interpreter_call.outputs", "message.output_text.logprobs",
	"clear_tool_uses_20250919", "clear_thinking_20251015", "all", "thinking_turns",
	"web_search", "file_search", "code_interpreter", "computer_use_preview", "image_generation", "mcp",
	"local_shell", "apply_patch",
)
