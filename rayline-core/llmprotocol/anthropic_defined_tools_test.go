package llmprotocol

import (
	"encoding/json"
	"errors"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// The table is the whole of what a foreign target knows about these tools, so
// every entry has to be a declaration the contract itself would accept.
func TestAnthropicDefinedToolsAreCallableDeclarations(t *testing.T) {
	limits := DefaultPolicy().Limits
	listed := AnthropicDefinedToolTypes()
	if !sort.StringsAreSorted(listed) {
		t.Fatalf("AnthropicDefinedToolTypes() is not sorted: %v", listed)
	}
	tabled := make([]string, 0, len(anthropicDefinedTools))
	for toolType := range anthropicDefinedTools {
		tabled = append(tabled, toolType)
	}
	sort.Strings(tabled)
	if !reflect.DeepEqual(listed, tabled) {
		t.Fatalf("AnthropicDefinedToolTypes() = %v, table holds %v", listed, tabled)
	}
	for _, toolType := range listed {
		t.Run(toolType, func(t *testing.T) {
			materialized := Tool{Type: toolType}.Materialized()
			if materialized.Type != "" {
				t.Fatalf("materialized tool still carries type %q", materialized.Type)
			}
			if materialized.ServerTool() {
				t.Fatal("a materialized tool reads as a server tool")
			}
			if err := validateRequestTool(materialized, limits, "tool 0"); err != nil {
				t.Fatalf("the documented declaration is not one the contract accepts: %v", err)
			}
			var schema struct {
				Type       string                     `json:"type"`
				Properties map[string]json.RawMessage `json:"properties"`
			}
			if err := json.Unmarshal(materialized.InputSchema, &schema); err != nil {
				t.Fatalf("documented schema is not JSON: %v", err)
			}
			if schema.Type != "object" || len(schema.Properties) == 0 {
				t.Fatalf("documented schema describes no arguments: %s", materialized.InputSchema)
			}
		})
	}
}

// The caller's declaration wins. Workshop names its editor
// str_replace_based_edit_tool and marks it as a cache breakpoint; the arm has
// to see that name, or the tool_use it emits will not match the client's
// handler, and the breakpoint has to survive for the same reason it survives
// on any custom tool.
func TestMaterializedToolKeepsWhatTheCallerStated(t *testing.T) {
	ttl := "5m"
	declared := Tool{
		Type: "text_editor_20250728", Name: "str_replace_based_edit_tool",
		Cache: &CacheDirective{TTL: ttl},
	}
	materialized := declared.Materialized()
	if materialized.Name != "str_replace_based_edit_tool" {
		t.Fatalf("name = %q, want the caller's", materialized.Name)
	}
	if materialized.Cache == nil || materialized.Cache.TTL != ttl {
		t.Fatalf("cache directive was lost: %+v", materialized.Cache)
	}
	if materialized.Description == "" || len(materialized.InputSchema) == 0 {
		t.Fatalf("the documented description and schema were not filled in: %+v", materialized)
	}
	// The declaration itself is untouched: a Messages target re-emits it.
	if declared.Type != "text_editor_20250728" || len(declared.InputSchema) != 0 {
		t.Fatalf("Materialized() mutated its receiver: %+v", declared)
	}

	own := json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}}}`)
	stated := Tool{Type: "bash_20250124", Description: "the caller's words", InputSchema: own}.Materialized()
	if stated.Description != "the caller's words" || string(stated.InputSchema) != string(own) {
		t.Fatalf("the caller's description or schema was overwritten: %+v", stated)
	}
}

func TestMaterializedLeavesOtherToolsAlone(t *testing.T) {
	for _, tool := range []Tool{
		{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)},
		{Name: "lookup", Type: "custom", InputSchema: json.RawMessage(`{"type":"object"}`)},
		{Type: "web_search_20250305"},
		{Type: "computer_toolset_20260801"},
	} {
		if got := tool.Materialized(); !reflect.DeepEqual(got, tool) {
			t.Fatalf("Materialized(%+v) = %+v", tool, got)
		}
		if _, defined := tool.AnthropicDefined(); defined {
			t.Fatalf("%+v reads as Anthropic-defined", tool)
		}
	}
}

// The shape Workshop sends: a type and a name, no schema. Validation must
// accept it the way it accepts a server tool, since the type is the schema.
func TestAnthropicDefinedToolsNeedNoSchema(t *testing.T) {
	limits := DefaultPolicy().Limits
	tools := []Tool{
		{Type: "text_editor_20250728", Name: "str_replace_based_edit_tool"},
		{Type: "bash_20250124"},
	}
	if _, _, err := validateRequestTools(tools, limits); err != nil {
		t.Fatalf("an Anthropic-defined tool declaration was refused: %v", err)
	}
}

// The distinction the routing gate reads. A server tool needs an arm that
// holds it; an Anthropic-defined tool needs nothing, since the caller runs
// it and the codec supplies the schema.
func TestAnthropicDefinedToolsAreNotServerTools(t *testing.T) {
	for _, toolType := range AnthropicDefinedToolTypes() {
		if (Tool{Type: toolType}).ServerTool() {
			t.Fatalf("%q reads as a server tool", toolType)
		}
	}
	for _, toolType := range []string{"web_search_20250305", "advisor_20260301", "computer_toolset_20260801"} {
		if !(Tool{Type: toolType}).ServerTool() {
			t.Fatalf("%q does not read as a server tool", toolType)
		}
	}
	if (Tool{Name: "lookup"}).ServerTool() || (Tool{Name: "lookup", Type: "custom"}).ServerTool() {
		t.Fatal("a custom tool reads as a server tool")
	}
}

// A dated revision the table has not seen resolves to its family's newest
// tabled schema, so the next text_editor revision Anthropic ships does not
// re-open the no_capable_arm refusal. Anything that is not a bare date stamp
// after the prefix is a new kind and stays a server tool.
func TestAnUnseenRevisionFallsBackToItsFamily(t *testing.T) {
	unseen := Tool{Type: "text_editor_20260401", Name: "str_replace_based_edit_tool"}
	definition, defined := unseen.AnthropicDefined()
	if !defined || unseen.ServerTool() {
		t.Fatalf("text_editor_20260401 is not read as Anthropic-defined")
	}
	newest, _ := Tool{Type: "text_editor_20250728"}.AnthropicDefined()
	if !reflect.DeepEqual(definition, newest) {
		t.Fatalf("text_editor_20260401 resolved to %q, want the newest text editor", definition.Name)
	}
	for _, toolType := range []string{"text_editor_toolset_20270101", "text_editor_2026", "text_editor_2026040x", "bash_toolset_20260801"} {
		if _, defined := (Tool{Type: toolType}).AnthropicDefined(); defined {
			t.Fatalf("%q is read as Anthropic-defined", toolType)
		}
	}
	for _, family := range anthropicDefinedToolFamilies {
		if _, tabled := anthropicDefinedTools[family.newest]; !tabled {
			t.Fatalf("family %q names %q as newest, which the table lacks", family.prefix, family.newest)
		}
		for toolType := range anthropicDefinedTools {
			if strings.HasPrefix(toolType, family.prefix) && toolType > family.newest {
				t.Fatalf("family %q names %q as newest, but the table holds %q", family.prefix, family.newest, toolType)
			}
		}
	}
}

// The documented schema is not one strict mode accepts, so a strict flag
// travels only beside the caller's own schema.
func TestMaterializedDropsStrictFromTheDocumentedSchema(t *testing.T) {
	strict := true
	documented := Tool{Type: "text_editor_20250728", Strict: &strict}.Materialized()
	if documented.Strict != nil {
		t.Fatal("strict survived onto the documented schema")
	}
	own := Tool{Type: "text_editor_20250728", Strict: &strict, InputSchema: json.RawMessage(`{"type":"object","additionalProperties":false}`)}.Materialized()
	if own.Strict == nil || !*own.Strict {
		t.Fatal("strict was dropped from the caller's own schema")
	}
}

// Validation reads the tool the arm will see. Two declarations that collide
// only once materialized are one name at the provider; a choice naming the
// documented name has to resolve; and the documented bytes count.
func TestValidationReadsTheMaterializedTool(t *testing.T) {
	limits := DefaultPolicy().Limits
	for name, tools := range map[string][]Tool{
		"two nameless revisions": {{Type: "text_editor_20250429"}, {Type: "text_editor_20250728"}},
		"custom tool under the documented name": {
			{Name: "str_replace_based_edit_tool", InputSchema: json.RawMessage(`{"type":"object"}`)},
			{Type: "text_editor_20250728"},
		},
	} {
		_, _, err := validateRequestTools(tools, limits)
		var protocolError *ProtocolError
		if !errors.As(err, &protocolError) || protocolError.Code != "duplicate_tool" {
			t.Fatalf("%s: validateRequestTools() error = %v, want duplicate_tool", name, err)
		}
	}
	namedTools, schemaBytes, err := validateRequestTools([]Tool{{Type: "text_editor_20250728"}}, limits)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateNamedToolChoice("str_replace_based_edit_tool", namedTools); err != nil {
		t.Fatalf("a choice naming the documented name was refused: %v", err)
	}
	if schemaBytes == 0 {
		t.Fatal("the documented schema was not counted")
	}
	small := limits
	small.SchemaBytes = 64
	_, _, err = validateRequestTools([]Tool{{Type: "text_editor_20250728"}}, small)
	var protocolError *ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "schema_limit" {
		t.Fatalf("a documented schema over the budget was accepted: %v", err)
	}
}
