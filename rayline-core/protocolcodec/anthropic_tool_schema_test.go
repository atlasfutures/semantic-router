package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// cuaReplJSSchema is the runtime parameter schema of Codex Computer Use's
// cua_repl js tool (unified-computer-use 26.1103.41825), trimmed to its
// shape: a top-level oneOf of a code call and a completion_summary call, with
// every property also declared at the top level (semantic-router#445).
const cuaReplJSSchema = `{
	"type": "object",
	"additionalProperties": false,
	"properties": {
		"code": {"type": "string"},
		"timeout_ms": {"type": "integer", "minimum": 1},
		"title": {"type": "string", "maxLength": 80},
		"completion_summary": {"type": "string", "maxLength": 80}
	},
	"oneOf": [
		{"type": "object", "additionalProperties": false, "required": ["code"],
		 "properties": {"code": {"type": "string"}, "timeout_ms": {"type": "integer"}, "title": {"type": "string"}}},
		{"type": "object", "additionalProperties": false, "required": ["completion_summary"],
		 "properties": {"completion_summary": {"type": "string"}}}
	]
}`

func TestAnthropicTargetFoldsTopLevelSchemaCombinators(t *testing.T) {
	tests := []struct {
		name   string
		schema string
		want   string
	}{
		{
			name:   "Codex cua_repl js: oneOf over properties the object already declares",
			schema: cuaReplJSSchema,
			want: `{"type": "object", "additionalProperties": false, "properties": {
				"code": {"type": "string"},
				"timeout_ms": {"type": "integer", "minimum": 1},
				"title": {"type": "string", "maxLength": 80},
				"completion_summary": {"type": "string", "maxLength": 80}}}`,
		},
		{
			name: "anyOf branches alone declare the properties, and none is required",
			schema: `{"required": ["id"], "properties": {"id": {"type": "string"}}, "anyOf": [
				{"required": ["path"], "properties": {"path": {"type": "string"}}},
				{"required": ["url"], "properties": {"url": {"type": "string"}, "id": {"type": "integer"}}}]}`,
			want: `{"type": "object", "required": ["id"], "properties": {
				"id": {"type": "string"}, "path": {"type": "string"}, "url": {"type": "string"}}}`,
		},
		{
			name: "allOf keeps every branch's required names",
			schema: `{"type": "object", "allOf": [
				{"required": ["a"], "properties": {"a": {"type": "string"}}},
				{"required": ["b", "a"], "properties": {"b": {"type": "number"}}}]}`,
			want: `{"type": "object", "required": ["a", "b"], "properties": {
				"a": {"type": "string"}, "b": {"type": "number"}}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, source := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1} {
				result := translateToolTurn(t, source, llmprotocol.AnthropicMessagesV1, test.schema)
				if got := anthropicToolSchema(t, result.Body); !jsonEqual(t, got, json.RawMessage(test.want)) {
					t.Fatalf("%s -> Messages input_schema = %s, want %s", source, got, test.want)
				}
				if !hasDiagnostic(result.Diagnostics, "tools.input_schema.combinator", llmprotocol.DiagnosticApproximated) {
					t.Fatalf("%s -> Messages folded the schema without counting it: %+v", source, result.Diagnostics)
				}
			}
		})
	}
}

func TestAnthropicTargetLeavesSchemasWithoutTopLevelCombinators(t *testing.T) {
	for _, schema := range []string{
		`{"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}`,
		`{"type": "object", "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}}`,
	} {
		result := translateToolTurn(t, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1, schema)
		if got := anthropicToolSchema(t, result.Body); !jsonEqual(t, got, json.RawMessage(schema)) {
			t.Fatalf("input_schema = %s, want it unchanged: %s", got, schema)
		}
		if hasDiagnostic(result.Diagnostics, "tools.input_schema.combinator", llmprotocol.DiagnosticApproximated) {
			t.Fatalf("an unchanged schema was counted as folded: %+v", result.Diagnostics)
		}
	}
}

// Chat and Responses accept a top-level combinator, so only a Messages target
// folds it.
func TestNonAnthropicTargetsKeepTopLevelSchemaCombinators(t *testing.T) {
	result := translateToolTurn(t, llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, cuaReplJSSchema)
	var body struct {
		Tools []struct {
			Function struct {
				Parameters json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(result.Body, &body); err != nil || len(body.Tools) != 1 {
		t.Fatalf("Chat body = %s (%v)", result.Body, err)
	}
	if !jsonEqual(t, body.Tools[0].Function.Parameters, json.RawMessage(cuaReplJSSchema)) {
		t.Fatalf("Chat parameters = %s, want the client's schema", body.Tools[0].Function.Parameters)
	}
}

func translateToolTurn(t *testing.T, source, target llmprotocol.WireFormat, schema string) RequestResult {
	t.Helper()
	var body string
	switch source {
	case llmprotocol.OpenAIResponsesV1:
		body = `{"model": "m", "input": "hi", "tools": [{"type": "function", "name": "js", "parameters": ` + schema + `}]}`
	case llmprotocol.OpenAIChatV1:
		body = `{"model": "m", "messages": [{"role": "user", "content": "hi"}],
			"tools": [{"type": "function", "function": {"name": "js", "parameters": ` + schema + `}}]}`
	default:
		t.Fatalf("no tool turn for %s", source)
	}
	result, err := NewBuiltinEngine().TranslateRequest(source, target, []byte(body), nil)
	if err != nil {
		t.Fatalf("%s -> %s refused the turn: %v", source, target, err)
	}
	return result
}

func anthropicToolSchema(t *testing.T, body []byte) json.RawMessage {
	t.Helper()
	var wire struct {
		Tools []struct {
			InputSchema json.RawMessage `json:"input_schema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &wire); err != nil || len(wire.Tools) != 1 {
		t.Fatalf("Messages body = %s (%v)", body, err)
	}
	return wire.Tools[0].InputSchema
}
