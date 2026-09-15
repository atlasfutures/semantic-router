package llmprotocol

import (
	"encoding/json"
	"reflect"
	"sort"
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
