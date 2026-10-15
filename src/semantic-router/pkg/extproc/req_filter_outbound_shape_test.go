package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// A parameter no named field models still reaches the record: a known key
// spelled out with its enum value, an unknown key and its string as digests.
func TestOutboundShapeRecordsAnUnmodeledParameter(t *testing.T) {
	body := `{"model":"m","service_tier":"flex","future_knob":"turbo","context_management":{"edits":[{"type":"clear_tool_uses_20250919"},{"type":"clear_thinking_20251015","keep":"all"}]},` +
		`"thinking":{"type":"adaptive","budget_tokens":2048}}`
	shape := readDispatchedOutboundParams([]byte(body), nil).Shape
	if shape["service_tier"] != "flex" {
		t.Errorf("service_tier = %v", shape["service_tier"])
	}
	future, ok := shape[outboundDigest("future_knob")].(map[string]interface{})
	if !ok || future["sha256"] != outboundDigest("turbo") || future["length"] != 5 {
		t.Errorf("an unknown parameter is not recorded as a digest: %v", shape)
	}
	edits := shape["context_management"].(map[string]interface{})["edits"].(map[string]interface{})
	if edits["count"] != 2 || fmt.Sprint(edits["values"]) != "[map[type:clear_tool_uses_20250919] map[keep:all type:clear_thinking_20251015]]" {
		t.Errorf("context_management.edits = %v", edits)
	}
	thinking := shape["thinking"].(map[string]interface{})
	if thinking["type"] != "adaptive" || thinking["budget_tokens"] != 2048.0 {
		t.Errorf("thinking = %v", thinking)
	}
}

// No content reaches the record, whichever member carries it: content members
// are counted, every other string and every unknown key is digested.
func TestOutboundShapeCarriesNoContent(t *testing.T) {
	s := outboundSecret
	body := `{"model":"` + s + `","messages":[{"role":"user","content":"` + s + `"}],"system":"` + s + `",` +
		`"input":[{"type":"message","content":[{"type":"input_text","text":"` + s + `"}]}],"instructions":"` + s + `",` +
		`"tools":[{"name":"` + s + `","description":"` + s + `","input_schema":{"title":"` + s + `"}}],` +
		`"tool_choice":{"type":"tool","name":"` + s + `"},"metadata":{"` + s + `":"` + s + `"},"user":"` + s + `",` +
		`"stop":["` + s + `"],"provider":{"order":["` + s + `"],"max_price":{"` + s + `":1}},` +
		`"output_config":{"format":{"type":"json_schema","schema":{"description":"` + s + `","properties":{"` + s + `":{"enum":["` + s + `"]}}}}},` +
		`"chat_template_kwargs":{"` + s + `":"` + s + `"},"previous_response_id":"` + s + `","` + s + `":{"` + s + `":["` + s + `"]}}`
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(body), nil))
	encoded, err := json.Marshal(record["outbound_params"])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), s) {
		t.Fatalf("content reached outbound_params: %s", encoded)
	}
	// Content members are counts or lengths only, never digests.
	if strings.Contains(string(encoded), `"messages":{"count":1}`) == false || !strings.Contains(string(encoded), `"system":{"length":18}`) {
		t.Fatalf("content members are not counted: %s", encoded)
	}
}

// An enum word is verbatim only under its own parameter, free-form maps are
// opaque, and repeated array elements do not spend the pass's budget.
func TestOutboundShapeScopesEnumsToTheirParameter(t *testing.T) {
	body := `{"model":"m","metadata":{"endpoint":"mcp"},"prediction":{"type":"content","content":"high"},` +
		`"user":"high","include":[` + strings.Repeat(`"reasoning.encrypted_content",`, 400) + `"reasoning.encrypted_content"],` +
		`"temperature":0.2,"reasoning":{"effort":"high","summary":"high"}}`
	shape := outboundShape([]byte(body))
	encoded, _ := json.Marshal(shape)
	if strings.Contains(string(encoded), `"mcp"`) || strings.Count(string(encoded), `"high"`) != 1 {
		t.Fatalf("an enum word was published outside its parameter: %s", encoded)
	}
	if shape["temperature"] != 0.2 {
		t.Fatalf("a repeated include spent the budget: %s", encoded)
	}
	if fmt.Sprint(shape["metadata"]) != "map[keys:1]" {
		t.Fatalf("metadata = %v", shape["metadata"])
	}
}

// A client-sized body yields a bounded shape.
func TestOutboundShapeIsBounded(t *testing.T) {
	members := make([]string, 0, 500)
	for i := 0; i < 500; i++ {
		members = append(members, fmt.Sprintf(`"k%d":{"a":[%s]}`, i, strings.Repeat(`"x",`, 50)+`"y"`))
	}
	encoded, _ := json.Marshal(outboundShape([]byte(`{` + strings.Join(members, ",") + `}`)))
	if len(encoded) > 64*1024 {
		t.Fatalf("shape is %d bytes", len(encoded))
	}
}
