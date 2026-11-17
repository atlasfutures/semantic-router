package protocolcodec

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// computerUseJSSchema is the shape of the parameter schema of Codex Computer
// Use's js tool, descriptions trimmed: a top-level oneOf of a code call and a
// completion_summary call, with every property also declared at the top
// level. The full schema is in testdata/codex/computer-use-js-tool-request.json.
const computerUseJSSchema = `{
	"type": "object",
	"additionalProperties": false,
	"properties": {
		"code": {"type": "string"},
		"completion_summary": {"type": "string"},
		"timeout_ms": {"type": "integer"},
		"title": {"type": "string"}
	},
	"oneOf": [
		{"type": "object", "additionalProperties": false, "required": ["code"],
		 "properties": {"code": {"type": "string"}, "timeout_ms": {"type": "integer"}, "title": {"type": "string"}}},
		{"type": "object", "additionalProperties": false, "required": ["completion_summary"],
		 "properties": {"completion_summary": {"type": "string"}}}
	]
}`

const schemaCombinatorField = "tools.input_schema.combinator"

func TestAnthropicInputSchemaFoldsTopLevelCombinators(t *testing.T) {
	tests := []struct {
		name   string
		schema string
		want   string
	}{
		{
			name:   "oneOf over properties the object already declares",
			schema: computerUseJSSchema,
			want: `{"type": "object", "additionalProperties": false, "properties": {
				"code": {"type": "string"}, "completion_summary": {"type": "string"},
				"timeout_ms": {"type": "integer"}, "title": {"type": "string"}}}`,
		},
		{
			name: "a branch that leaves a property out and allows any other allows it with any value",
			schema: `{"required": ["id"], "properties": {"id": {"type": "string"}}, "anyOf": [
				{"required": ["path"], "properties": {"path": {"type": "string"}}},
				{"required": ["url"], "properties": {"url": {"type": "string"}}}]}`,
			want: `{"type": "object", "required": ["id"], "properties": {
				"id": {"type": "string"},
				"path": {"anyOf": [{"type": "string"}, {}]},
				"url": {"anyOf": [{}, {"type": "string"}]}}}`,
		},
		{
			name: "oneOf branches that type one property differently keep every alternative",
			schema: `{"type": "object", "required": ["action"], "oneOf": [
				{"additionalProperties": false, "properties": {"action": {"enum": ["run"]}, "code": {"type": "string"}}},
				{"additionalProperties": false, "properties": {"action": {"enum": ["finish"]}, "code": {"type": "string"}}}]}`,
			want: `{"additionalProperties": false, "type": "object", "required": ["action"], "properties": {
				"action": {"anyOf": [{"enum": ["run"]}, {"enum": ["finish"]}]}, "code": {"type": "string"}}}`,
		},
		{
			name: "branches that name definitions fold those definitions, URI and JSON Pointer escapes decoded",
			schema: `{"$defs": {
					"run/code": {"type": "object", "additionalProperties": false, "required": ["code"], "properties": {"code": {"type": "string"}}},
					"done~": {"type": "object", "additionalProperties": false, "properties": {"completion_summary": {"type": "string"}}}},
				"anyOf": [{"$ref": "#/$defs/run~1code"}, {"$ref": "#/$defs/done~0"}, {"$ref": "#/$defs/done%7E0"}]}`,
			want: `{"additionalProperties": false, "type": "object", "$defs": {
					"run/code": {"type": "object", "additionalProperties": false, "required": ["code"], "properties": {"code": {"type": "string"}}},
					"done~": {"type": "object", "additionalProperties": false, "properties": {"completion_summary": {"type": "string"}}}},
				"properties": {"code": {"type": "string"}, "completion_summary": {"type": "string"}}}`,
		},
		{
			name: "references to the root and its definitions stay valid and fold",
			schema: `{"$defs": {"Id": {"type": "string"}}, "properties": {"id": {"$ref": "#/$defs/Id"}},
				"oneOf": [{"additionalProperties": false, "properties": {"id": {"$ref": "#/$defs/Id"}, "parent": {"$ref": "#"}}}]}`,
			want: `{"additionalProperties": false, "type": "object", "$defs": {"Id": {"type": "string"}}, "properties": {
				"id": {"$ref": "#/$defs/Id"}, "parent": {"$ref": "#"}}}`,
		},
		{
			name:   "the client's strings keep their bytes",
			schema: `{"properties": {"q": {"description": "a <b> & c"}}, "oneOf": [{"properties": {"r": {}}}]}`,
			want:   `{"type": "object", "properties": {"q": {"description": "a <b> & c"}, "r": {}}}`,
		},
		{
			name:   "a singleton object type array becomes the object type",
			schema: `{"type": ["object"], "oneOf": [{"properties": {"a": {}}}]}`,
			want:   `{"type": "object", "properties": {"a": {}}}`,
		},
		{
			name: "a name every oneOf branch requires stays required",
			schema: `{"oneOf": [
				{"additionalProperties": false, "required": ["action", "code"], "properties": {"action": {"const": "run"}, "code": {"type": "string"}}},
				{"additionalProperties": false, "required": ["action"], "properties": {"action": {"const": "finish"}}}]}`,
			want: `{"additionalProperties": false, "type": "object", "required": ["action"], "properties": {
				"action": {"anyOf": [{"const": "run"}, {"const": "finish"}]}, "code": {"type": "string"}}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, folded := anthropicInputSchema(json.RawMessage(test.schema))
			if !folded || !jsonEqual(t, got, json.RawMessage(test.want)) {
				t.Fatalf("anthropicInputSchema = %s (folded %v), want %s", got, folded, test.want)
			}
			again, refolded := anthropicInputSchema(got)
			if refolded || string(again) != string(got) {
				t.Fatalf("folding a folded schema changed it: %s -> %s", got, again)
			}
		})
	}
}

// The fold re-encodes the schema without escaping <, > and &, so the
// client's strings reach Messages as the client wrote them.
func TestAnthropicInputSchemaKeepsTheClientsStringBytes(t *testing.T) {
	schema := json.RawMessage(`{"properties": {"q": {"description": "a <b> & c"}}, "oneOf": [{"properties": {"r": {}}}]}`)
	got, folded := anthropicInputSchema(schema)
	if !folded || !strings.Contains(string(got), `"a <b> & c"`) {
		t.Fatalf("anthropicInputSchema = %s (folded %v), want the description's bytes kept", got, folded)
	}
}

// A fold costs about what the schema is: repeated branches are weighed once,
// and a schema with very many branch-property pairs is sent unchanged.
func TestAnthropicInputSchemaFoldCostStaysNearTheSchemaSize(t *testing.T) {
	wide := func(count int) string {
		properties := map[string]json.RawMessage{}
		for i := range count {
			properties[fmt.Sprintf("p%04d", i)] = json.RawMessage(`{"type": "string"}`)
		}
		encoded, _ := json.Marshal(map[string]any{"properties": properties})
		return string(encoded)
	}
	repeated := `{"$defs": {"Wide": ` + wide(500) + `}, "anyOf": [` +
		strings.TrimSuffix(strings.Repeat(`{"$ref": "#/$defs/Wide"},`, 1000), ",") + `]}`
	if got, folded := anthropicInputSchema(json.RawMessage(repeated)); !folded || len(got) > len(repeated) {
		t.Fatalf("one definition named by many branches: folded %v into %d bytes from %d", folded, len(got), len(repeated))
	}
	// The same definition named by references spelled differently is still
	// decoded once.
	const name = "WideDefinition"
	var spellings []string
	for i := range 512 {
		var spelled strings.Builder
		for bit, letter := range name {
			if i&(1<<bit) != 0 {
				fmt.Fprintf(&spelled, "%%%02X", letter)
			} else {
				spelled.WriteRune(letter)
			}
		}
		spellings = append(spellings, `{"$ref": "#/$defs/`+spelled.String()+`"}`)
	}
	respelled := json.RawMessage(`{"$defs": {"` + name + `": ` + wide(2000) + `}, "anyOf": [` + strings.Join(spellings, ", ") + `]}`)
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	if _, folded := anthropicInputSchema(respelled); !folded {
		t.Fatalf("one definition named by differently spelled references was not folded")
	}
	runtime.ReadMemStats(&after)
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 32<<20 {
		t.Fatalf("folding a %d-byte schema allocated %d bytes", len(respelled), allocated)
	}
	var branches []string
	for i := range 300 {
		branches = append(branches, fmt.Sprintf(`{"properties": {"p%03d": {}}}`, i))
	}
	many := `{"anyOf": [` + strings.Join(branches, ", ") + `]}`
	if got, folded := anthropicInputSchema(json.RawMessage(many)); folded || string(got) != many {
		t.Fatalf("300 branches by 300 properties were folded")
	}
}

// A schema with no top-level combinator, or one that cannot be folded without
// losing a branch, is sent as it came.
func TestAnthropicInputSchemaLeavesSchemasItCannotFold(t *testing.T) {
	for name, schema := range map[string]string{
		"no top-level combinator":       `{"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}`,
		"a combinator below the top":    `{"type": "object", "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}}`,
		"an external $ref":              `{"oneOf": [{"$ref": "https://example.com/run.json"}, {"properties": {"y": {"type": "string"}}}]}`,
		"an undefined $ref":             `{"$defs": {}, "oneOf": [{"$ref": "#/$defs/Missing"}]}`,
		"a $ref more than one deep":     `{"$defs": {"a": {"b": {"properties": {}}}}, "oneOf": [{"$ref": "#/$defs/a/b"}]}`,
		"a $ref to a $ref":              `{"$defs": {"a": {"$ref": "#/$defs/b"}, "b": {}}, "oneOf": [{"$ref": "#/$defs/a"}]}`,
		"a $ref with siblings":          `{"$defs": {"Base": {"properties": {"id": {"type": "string"}}}}, "oneOf": [{"$ref": "#/$defs/Base", "properties": {"code": {"type": "string"}}}]}`,
		"a branch that is a combinator": `{"oneOf": [{"anyOf": [{"properties": {"code": {"type": "string"}}}, {"properties": {"done": {"type": "string"}}}]}]}`,
		"a branch with patternProperties": `{"oneOf": [{"properties": {"a": {"type": "string"}}},
			{"patternProperties": {"^x-": {"type": "string"}}}]}`,
		"a branch that is not an object":    `{"anyOf": [{"properties": {"a": {"type": "string"}}}, true]}`,
		"a null branch":                     `{"anyOf": [null]}`,
		"a combinator that is not an array": `{"oneOf": {"properties": {"a": {"type": "string"}}}}`,
		"null properties in a branch":       `{"oneOf": [{"properties": null}, {"properties": {"x": {}}}]}`,
		"null required at the top":          `{"required": null, "oneOf": [{"properties": {"x": {}}}]}`,
		"a null definitions table":          `{"$defs": null, "oneOf": [{"properties": {"x": {}}}]}`,
		"a top-level additionalProperties schema over a promoted property": `{"type": "object", "additionalProperties": {"type": "string"},
			"anyOf": [{"properties": {"x": {}}}]}`,
		"a root type other than object": `{"type": ["object", "null"], "oneOf": [{"properties": {"a": {}}}]}`,
		"a closed root over a property only a branch declares": `{"additionalProperties": false, "properties": {"x": {}},
			"anyOf": [{"properties": {"y": {}}}]}`,
		"a top-level allOf": `{"allOf": [{"properties": {"a": {"type": "string"}}}]}`,
		"root patternProperties when every branch is closed": `{"patternProperties": {"^x-": {}},
			"oneOf": [{"additionalProperties": false, "properties": {"id": {}}}]}`,
		"a $ref with an invalid pointer escape":          `{"$defs": {"run~2code": {"properties": {"a": {}}}}, "oneOf": [{"$ref": "#/$defs/run~2code"}]}`,
		"a root property every closed branch leaves out": `{"properties": {"x": {}}, "oneOf": [{"additionalProperties": false, "properties": {"y": {}}}]}`,
		"a closed branch requiring a name it does not declare": `{"oneOf": [{"additionalProperties": false, "required": ["x"], "properties": {}},
			{"additionalProperties": false, "required": ["y"], "properties": {"y": {}}}]}`,
		"a reference to a root property": `{"oneOf": [{"properties": {"x": {"type": "string"}, "y": {"$ref": "#/properties/x"}}}]}`,
		"a closed branch that leaves out a name the root requires": `{"required": ["x"], "oneOf": [{"additionalProperties": false},
			{"additionalProperties": false, "required": ["y"], "properties": {"x": {}, "y": {}}}]}`,
		"a repeated oneOf branch": `{"oneOf": [{"properties": {"a": {}}}, {"properties": {"a": {}}}]}`,
		"a branch constraining a root property differently": `{"required": ["x"], "properties": {"x": {"type": "string"}},
			"anyOf": [{"properties": {"x": {"maxLength": 1}}}]}`,
		"a branch requiring a property declared false": `{"anyOf": [{"additionalProperties": false, "required": ["x"], "properties": {"x": false}},
			{"additionalProperties": false, "required": ["y"], "properties": {"y": {}}}]}`,
		"a required name listed twice": `{"oneOf": [{"required": ["x", "x"], "properties": {"x": {}}}]}`,
		"two definitions with the same content in a oneOf": `{"$defs": {"A": {"properties": {"x": {}}}, "B": {"properties": {"x": {}}}},
			"oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]}`,
		"a branch declaring false a name the root requires": `{"required": ["x"], "anyOf": [{"properties": {"x": false}},
			{"required": ["y"], "properties": {"y": {}}}]}`,
		"a branch with a title that is not a string": `{"oneOf": [{"title": 42, "properties": {"x": {}}}]}`,
		"an empty combinator":                        `{"oneOf": []}`,
		"a branch with an empty type":                `{"oneOf": [{"type": "", "properties": {"a": {}}}]}`,
		"a null combinator":                          `{"oneOf": null}`,
		"properties that are not an object":          `{"properties": [], "oneOf": [{"properties": {"a": {}}}]}`,
		"a branch's required that is not names":      `{"oneOf": [{"required": [1], "properties": {"a": {}}}]}`,
		"definitions that are not an object":         `{"$defs": [], "oneOf": [{"properties": {"a": {}}}]}`,
		"a malformed additionalProperties": `{"oneOf": [{"properties": {"a": {"type": "string"}}},
			{"additionalProperties": "no", "properties": {"b": {}}}]}`,
		"a $ref into a combinator branch": `{"oneOf": [{"$defs": {"Value": {"type": "string"}},
			"properties": {"value": {"$ref": "#/oneOf/0/$defs/Value"}}}]}`,
		"a $ref to an anchor a branch declares": `{"type": "object", "properties": {"v": {"$ref": "#Value"}},
			"oneOf": [{"$defs": {"V": {"$anchor": "Value", "type": "string"}}}]}`,
		"a $dynamicRef": `{"anyOf": [{"properties": {"items": {"$dynamicRef": "#node"}}}]}`,
		"a branch with its own $id and $defs": `{"oneOf": [{"$id": "https://example.com/run", "$defs": {"Value": {"type": "string"}},
			"properties": {"value": {"$ref": "#/$defs/Value"}}}]}`,
		"a branch with a keyword the fold would drop": `{"anyOf": [{"properties": {"a": {}}, "minProperties": 1}]}`,
		"a branch of another type":                    `{"oneOf": [{"type": "string"}, {"properties": {"a": {}}}]}`,
		"a top-level unevaluatedProperties":           `{"type": "object", "unevaluatedProperties": false, "oneOf": [{"properties": {"a": {}}}]}`,
		"two top-level combinators": `{"anyOf": [{"properties": {"x": {"type": "string"}}}],
			"allOf": [{"properties": {"x": {"maxLength": 8}}}]}`,
		"a branch that allows other properties through a schema": `{"oneOf": [{"properties": {"a": {"type": "string"}}},
			{"additionalProperties": {"type": "number"}, "properties": {"b": {}}}]}`,
		"not an object": `[{"oneOf": []}]`,
	} {
		t.Run(name, func(t *testing.T) {
			got, folded := anthropicInputSchema(json.RawMessage(schema))
			if folded || string(got) != schema {
				t.Fatalf("anthropicInputSchema = %s (folded %v), want it unchanged: %s", got, folded, schema)
			}
		})
	}
}

// Every source format reaches a Claude arm through the Messages encoder, so
// each one is folded there, and the fold is counted as an approximation.
func TestAnthropicTargetFoldsToolSchemasFromEverySource(t *testing.T) {
	want := `{"type": "object", "additionalProperties": false, "properties": {
		"code": {"type": "string"}, "completion_summary": {"type": "string"},
		"timeout_ms": {"type": "integer"}, "title": {"type": "string"}}}`
	for _, source := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		result := translateToolTurn(t, source, llmprotocol.AnthropicMessagesV1, computerUseJSSchema, routeToArm)
		if got := anthropicToolSchemas(t, result.Body)["js"]; !jsonEqual(t, got, json.RawMessage(want)) {
			t.Fatalf("%s -> Messages input_schema = %s, want %s", source, got, want)
		}
		if !hasDiagnostic(result.Diagnostics, schemaCombinatorField, llmprotocol.DiagnosticApproximated) {
			t.Fatalf("%s -> Messages folded the schema without counting it: %+v", source, result.Diagnostics)
		}
	}
}

// A Messages turn sent on as the client's own bytes is not re-encoded, so its
// schema is the client's: Anthropic answers that client as it would directly.
func TestAnthropicReplayKeepsTheClientsToolSchema(t *testing.T) {
	result := translateToolTurn(t, llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, computerUseJSSchema, nil)
	if got := anthropicToolSchemas(t, result.Body)["js"]; !jsonEqual(t, got, json.RawMessage(computerUseJSSchema)) {
		t.Fatalf("replayed input_schema = %s, want the client's", got)
	}
}

// Chat and Responses accept a top-level combinator, so only a Messages target
// folds it.
func TestNonAnthropicTargetsKeepTopLevelSchemaCombinators(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		result := translateToolTurn(t, llmprotocol.OpenAIResponsesV1, target, computerUseJSSchema, routeToArm)
		var body struct {
			Tools []struct {
				Parameters json.RawMessage `json:"parameters"`
				Function   struct {
					Parameters json.RawMessage `json:"parameters"`
				} `json:"function"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(result.Body, &body); err != nil || len(body.Tools) != 1 {
			t.Fatalf("%s body = %s (%v)", target, result.Body, err)
		}
		got := body.Tools[0].Parameters
		if target == llmprotocol.OpenAIChatV1 {
			got = body.Tools[0].Function.Parameters
		}
		if !jsonEqual(t, got, json.RawMessage(computerUseJSSchema)) {
			t.Fatalf("%s parameters = %s, want the client's schema", target, got)
		}
		if hasDiagnostic(result.Diagnostics, schemaCombinatorField, llmprotocol.DiagnosticApproximated) {
			t.Fatalf("%s counted a fold it did not make: %+v", target, result.Diagnostics)
		}
	}
}

// The Computer Use tools of a Responses request Codex sends, as captured with
// their long descriptions shortened: the js tool sits in the mcp__cua_repl
// namespace. Sent to a Claude arm, no tool
// keeps a top-level combinator and js keeps all four of its properties, each
// with the schema the client gave it.
func TestCodexComputerUseRequestEncodesForAnthropic(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("testdata", "codex", "computer-use-js-tool-request.json"))
	if err != nil {
		t.Fatal(err)
	}
	var captured struct {
		Tools []struct {
			Tools []struct {
				Name       string          `json:"name"`
				Parameters json.RawMessage `json:"parameters"`
			} `json:"tools"`
		} `json:"tools"`
	}
	if err = json.Unmarshal(body, &captured); err != nil || len(captured.Tools) != 1 ||
		len(captured.Tools[0].Tools) == 0 || captured.Tools[0].Tools[0].Name != "js" {
		t.Fatalf("fixture = %s (%v)", body, err)
	}
	var clientJS struct {
		Properties map[string]json.RawMessage `json:"properties"`
		OneOf      json.RawMessage            `json:"oneOf"`
	}
	if err = json.Unmarshal(captured.Tools[0].Tools[0].Parameters, &clientJS); err != nil || clientJS.OneOf == nil {
		t.Fatalf("the fixture's js schema has no top-level oneOf: %v", err)
	}

	result, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1, body, routeToArm)
	if err != nil {
		t.Fatalf("Responses -> Messages refused the turn: %v", err)
	}
	schemas := anthropicToolSchemas(t, result.Body)
	if len(schemas) != 2 {
		t.Fatalf("Messages tools = %s, want js and js_reset", result.Body)
	}
	for name, schema := range schemas {
		var members map[string]json.RawMessage
		if err := json.Unmarshal(schema, &members); err != nil {
			t.Fatalf("%s input_schema = %s (%v)", name, schema, err)
		}
		for _, combinator := range anthropicSchemaCombinators {
			if _, present := members[combinator]; present {
				t.Fatalf("%s input_schema keeps a top-level %s: %s", name, combinator, schema)
			}
		}
	}
	var js struct {
		Type                 string                     `json:"type"`
		AdditionalProperties *bool                      `json:"additionalProperties"`
		Properties           map[string]json.RawMessage `json:"properties"`
		Required             []string                   `json:"required"`
	}
	if err := json.Unmarshal(schemas["mcp__cua_repl__js"], &js); err != nil {
		t.Fatalf("js input_schema = %s (%v)", schemas["mcp__cua_repl__js"], err)
	}
	if js.Type != "object" || js.AdditionalProperties == nil || *js.AdditionalProperties || len(js.Required) != 0 {
		t.Fatalf("js input_schema = %s, want a closed object with nothing required", schemas["mcp__cua_repl__js"])
	}
	if len(js.Properties) != 4 {
		t.Fatalf("js properties = %v, want code, completion_summary, timeout_ms and title", js.Properties)
	}
	for name, property := range clientJS.Properties {
		if !jsonEqual(t, js.Properties[name], property) {
			t.Fatalf("js property %s = %s, want the client's %s", name, js.Properties[name], property)
		}
	}
}

// routeToArm stands for the routing step that picks an arm: the request is
// rewritten, so a same-format turn is encoded rather than replayed.
func routeToArm(request *llmprotocol.Request) error {
	request.Model = "arm"
	return nil
}

func translateToolTurn(t *testing.T, source, target llmprotocol.WireFormat, schema string, mutate RequestMutation) RequestResult {
	t.Helper()
	var body string
	switch source {
	case llmprotocol.OpenAIResponsesV1:
		body = `{"model": "m", "input": "hi", "tools": [{"type": "function", "name": "js", "parameters": ` + schema + `}]}`
	case llmprotocol.OpenAIChatV1:
		body = `{"model": "m", "messages": [{"role": "user", "content": "hi"}],
			"tools": [{"type": "function", "function": {"name": "js", "parameters": ` + schema + `}}]}`
	case llmprotocol.AnthropicMessagesV1:
		body = `{"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
			"tools": [{"name": "js", "input_schema": ` + schema + `}]}`
	default:
		t.Fatalf("no tool turn for %s", source)
	}
	result, err := NewBuiltinEngine().TranslateRequest(source, target, []byte(body), mutate)
	if err != nil {
		t.Fatalf("%s -> %s refused the turn: %v", source, target, err)
	}
	return result
}

// anthropicToolSchemas is each Messages tool's input_schema by name.
func anthropicToolSchemas(t *testing.T, body []byte) map[string]json.RawMessage {
	t.Helper()
	var wire struct {
		Tools []struct {
			Name        string          `json:"name"`
			InputSchema json.RawMessage `json:"input_schema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &wire); err != nil || len(wire.Tools) == 0 {
		t.Fatalf("Messages body = %s (%v)", body, err)
	}
	schemas := map[string]json.RawMessage{}
	for _, tool := range wire.Tools {
		if strings.TrimSpace(string(tool.InputSchema)) == "" {
			t.Fatalf("Messages tool %s has no input_schema: %s", tool.Name, body)
		}
		schemas[tool.Name] = tool.InputSchema
	}
	return schemas
}
