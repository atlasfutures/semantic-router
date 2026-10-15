package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const outboundSecret = "SECRET-PROMPT-TEXT"

// A Messages body's outbound parameters reach the record as ids and numbers,
// and no content does.
func TestOutboundParamsReadAMessagesBody(t *testing.T) {
	body := `{"model":"anthropic/claude-sonnet-5.5","max_tokens":32000,"temperature":0.5,"stream":true,` +
		`"cache_control":{"type":"ephemeral","ttl":"1h"},` +
		`"system":[{"type":"text","text":"` + outboundSecret + `","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"` + outboundSecret + `"},` +
		`{"role":"user","content":[{"type":"text","text":"` + outboundSecret + `","cache_control":{"type":"ephemeral"}},` +
		`{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"` + outboundSecret + `","cache_control":{"type":"ephemeral"}}]}]}],` +
		// A schema property named cache_control is data, not a directive.
		`"tools":[{"name":"Bash","description":"` + outboundSecret + `","input_schema":{"properties":{"cache_control":{"type":"string"}}}},` +
		`{"name":"Edit","input_schema":{},"cache_control":{"type":"ephemeral"}}],` +
		`"top_p":0.9,"top_k":40,"seed":7,"frequency_penalty":0.1,"presence_penalty":0.2,"n":1,"tool_choice":{"type":"tool","name":"Bash","disable_parallel_tool_use":true},"stop_sequences":["` + outboundSecret + `","x"],` +
		`"thinking":{"type":"enabled","budget_tokens":4096,"display":"summarized"},"output_config":{"effort":"high","format":{"type":"json_schema","schema":{"title":"` + outboundSecret + `"}}},` +
		`"provider":{"only":["anthropic"],"order":["anthropic"],"allow_fallbacks":false,"api_key":"` + outboundSecret + `"}}`
	dispatch := &providerDispatch{
		targetFormat: llmprotocol.AnthropicMessagesV1, backendName: "openrouter",
		profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"},
	}
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(body), dispatch))
	want := map[string]interface{}{
		"outbound_model":                  "anthropic/claude-sonnet-5.5",
		"outbound_protocol":               "anthropic.messages.v1",
		"outbound_provider":               "openrouter",
		"outbound_backend":                "openrouter",
		"outbound_output_config_effort":   "high",
		"outbound_thinking_type":          "enabled",
		"outbound_thinking_display":       "summarized",
		"outbound_thinking_budget_tokens": int64(4096),
		"outbound_max_tokens":             int64(32000),
		"outbound_temperature":            0.5,
		"outbound_stream":                 true,
		"outbound_tool_count":             2,
		"outbound_cache_control_markers":  5,
		"outbound_top_p":                  0.9,
		"outbound_top_k":                  int64(40),
		"outbound_seed":                   int64(7),
		"outbound_frequency_penalty":      0.1,
		"outbound_presence_penalty":       0.2,
		"outbound_n":                      int64(1),
		"outbound_parallel_tool_calls":    false,
		"outbound_tool_choice":            "tool",
		"outbound_tool_choice_name":       outboundDigest("Bash"),
		"outbound_stop_sequences":         int64(2),
		"outbound_provider_preferences":   `{"order":["anthropic"],"allow_fallbacks":false,"only":["anthropic"]}`,
		"outbound_output_format":          "json_schema",
		"outbound_prompt_cache_key":       false,
	}
	for key, value := range want {
		if fmt.Sprint(record[key]) != fmt.Sprint(value) {
			t.Errorf("%s = %v, want %v", key, record[key], value)
		}
	}
	if policies, _ := record["outbound_cache_control_policies"].([]string); fmt.Sprint(policies) != "[ephemeral// ephemeral/1h/]" {
		t.Errorf("outbound_cache_control_policies = %v", record["outbound_cache_control_policies"])
	}
	if only, _ := record["outbound_provider_only"].([]string); len(only) != 1 || only[0] != "anthropic" {
		t.Errorf("outbound_provider_only = %v", record["outbound_provider_only"])
	}
	for _, absent := range []string{"outbound_reasoning_effort", "outbound_top_level_effort", "outbound_reasoning_max_tokens"} {
		if _, present := record[absent]; present {
			t.Errorf("%s is set on a body that does not carry it", absent)
		}
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), outboundSecret) {
		t.Fatalf("content reached the record: %s", encoded)
	}
}

// A Responses body names its reasoning object, its output limit and its
// prompt cache key; parameters it omits stay off the record.
func TestOutboundParamsReadAResponsesBody(t *testing.T) {
	body := `{"model":"openai/gpt-6.1-sol","input":[],"reasoning":{"effort":"medium","max_tokens":2048},` +
		`"max_output_tokens":8000,"prompt_cache_key":"episode-1","prompt_cache_retention":"24h",` +
		`"text":{"format":{"type":"json_schema","strict":true,"schema":{"title":"SECRET"}}}}`
	dispatch := &providerDispatch{targetFormat: llmprotocol.OpenAIResponsesV1, profile: &config.ProviderProfile{Type: "openai"}}
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(body), dispatch))
	if record["outbound_reasoning_effort"] != "medium" || record["outbound_reasoning_max_tokens"] != int64(2048) ||
		record["outbound_max_tokens"] != int64(8000) || record["outbound_prompt_cache_key"] != true ||
		record["outbound_provider"] != "openai" || record["outbound_tool_count"] != 0 ||
		record["outbound_prompt_cache_retention"] != "24h" || record["outbound_output_format"] != "json_schema" ||
		record["outbound_output_format_strict"] != true {
		t.Fatalf("record = %v", record)
	}
	for _, absent := range []string{"outbound_temperature", "outbound_stream", "outbound_output_config_effort", "outbound_thinking_type",
		"outbound_top_p", "outbound_top_k", "outbound_seed", "outbound_n", "outbound_parallel_tool_calls"} {
		if _, present := record[absent]; present {
			t.Errorf("%s is set on a body that does not carry it", absent)
		}
	}
}

// Through the router, a v5 turn's record names the outbound parameters of
// the bytes that were dispatched.
func TestV5TurnRecordsItsOutboundParams(t *testing.T) {
	router, fake := v5Router(t, "anthropic")
	client := `{"model":"auto","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":"fix the failing test"}]}`
	body, ctx := v5Turn(t, router, fake, v5GLMUp, "v5-outbound", client)
	params := ctx.DispatchedOutbound
	if params == nil {
		t.Fatal("the dispatch recorded no outbound parameters")
	}
	if params.Model != "z-ai/glm-5.3-flash" || params.Protocol != string(llmprotocol.AnthropicMessagesV1) ||
		params.Provider != "openrouter" || params.Stream == nil || !*params.Stream || params.MaxTokens == nil {
		t.Fatalf("params = %+v\nbody %s", params, body)
	}
	if want := readDispatchedOutboundParams(body, nil); want.Model != params.Model || *want.MaxTokens != *params.MaxTokens ||
		want.OutputConfigEffort != params.OutputConfigEffort || want.CacheControlMarkers != params.CacheControlMarkers {
		t.Fatalf("the record does not describe the dispatched bytes: %+v vs %+v", params, want)
	}
}

// Responses reasoning.summary is recorded as a known enum or "other", never
// its raw text; Messages task_budget as a number; and a cache directive's
// evict_on_complete as part of its policy.
func TestOutboundParamsReadSummaryTaskBudgetAndEviction(t *testing.T) {
	for raw, want := range map[string]interface{}{`"auto"`: "auto", `"detailed"`: "detailed", `"` + outboundSecret + `"`: "other", `7`: "other"} {
		record := map[string]interface{}{}
		appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(`{"model":"m","reasoning":{"effort":"low","summary":`+raw+`}}`), nil))
		if record["outbound_reasoning_summary"] != want || record["outbound_reasoning_effort"] != "low" {
			t.Errorf("summary %s: record = %v", raw, record)
		}
	}
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(`{"model":"m","reasoning":{"effort":"low"}}`), nil))
	if _, present := record["outbound_reasoning_summary"]; present {
		t.Errorf("summary set on a body that does not carry it: %v", record)
	}
	body := `{"model":"m","output_config":{"effort":"high","task_budget":2048},"messages":[{"role":"user","content":[` +
		`{"type":"text","text":"x","cache_control":{"type":"ephemeral","evict_on_complete":true}},` +
		`{"type":"text","text":"y","cache_control":{"type":"ephemeral"}}]}]}`
	record = map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(body), nil))
	if record["outbound_task_budget"] != int64(2048) || record["outbound_output_config_effort"] != "high" {
		t.Errorf("record = %v", record)
	}
	if policies, _ := record["outbound_cache_control_policies"].([]string); fmt.Sprint(policies) != "[ephemeral// ephemeral///evict_on_complete=true]" {
		t.Errorf("outbound_cache_control_policies = %v", record["outbound_cache_control_policies"])
	}
	record = map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(`{"model":"m","output_config":{"task_budget":{"total":1}}}`), nil))
	if _, present := record["outbound_task_budget"]; present || record["outbound_model"] != "m" {
		t.Errorf("a non-numeric task_budget: record = %v", record)
	}
}

// A Responses body that references provider-held context says so without
// recording the ids.
func TestOutboundParamsRecordProviderContextPresence(t *testing.T) {
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(`{"model":"m","previous_response_id":"`+outboundSecret+`","conversation":{"id":"`+outboundSecret+`"}}`), nil))
	if record["outbound_previous_response_id"] != true || record["outbound_conversation"] != true {
		t.Fatalf("record = %v", record)
	}
	record = map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(`{"model":"m","store":false,"auto_store":true}`), nil))
	if record["outbound_store"] != false || record["outbound_auto_store"] != true {
		t.Fatalf("storage flags: record = %v", record)
	}
	record = map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(`{"model":"m","conversation":"conv_1"}`), nil))
	if _, present := record["outbound_previous_response_id"]; present || record["outbound_conversation"] != true {
		t.Fatalf("record = %v", record)
	}
	if encoded, _ := json.Marshal(record); strings.Contains(string(encoded), "conv_1") {
		t.Fatalf("an id reached the record: %s", encoded)
	}
}

// A client's provider object reaches the record with its slugs readable and
// any other value (a hostname, a pasted key) only as a digest.
func TestOutboundParamsRedactUnslugProviderValues(t *testing.T) {
	body := `{"model":"m","provider":{"order":["deepinfra/fp8","Anthropic"],"only":["` + outboundSecret + `.internal.example"],` +
		`"ignore":["sk-or-v1-` + outboundSecret + `-0123456789abcdef0123456789"],"data_collection":"` + outboundSecret + `",` +
		`"max_price":{"prompt":1.5,"` + outboundSecret + `":2}}}`
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(body), nil))
	encoded, _ := json.Marshal(record)
	if strings.Contains(string(encoded), outboundSecret) {
		t.Fatalf("a provider value reached the record: %s", encoded)
	}
	preferences, _ := record["outbound_provider_preferences"].(string)
	if !strings.Contains(preferences, `"order":["deepinfra/fp8","Anthropic"]`) || !strings.Contains(preferences, `"data_collection":"other"`) ||
		!strings.Contains(preferences, `"prompt":1.5`) || !strings.Contains(preferences, `"sha256:`) {
		t.Fatalf("outbound_provider_preferences = %s", preferences)
	}
	if only, _ := record["outbound_provider_only"].([]string); len(only) != 1 || !strings.HasPrefix(only[0], "sha256:") {
		t.Fatalf("outbound_provider_only = %v", record["outbound_provider_only"])
	}
}

// Every enum-shaped string a client writes is recorded verbatim only when the
// protocols define it; anything else is a digest.
func TestOutboundParamsDigestUnknownEnumValues(t *testing.T) {
	body := `{"model":"m","reasoning_effort":"` + outboundSecret + `","reasoning":{"effort":"` + outboundSecret + `"},` +
		`"output_config":{"effort":"` + outboundSecret + `","format":{"type":"` + outboundSecret + `"}},` +
		`"thinking":{"type":"` + outboundSecret + `","display":"` + outboundSecret + `"},` +
		`"chat_template_kwargs":{"reasoning_effort":"` + outboundSecret + `"},` +
		`"tool_choice":{"type":"` + outboundSecret + `","name":"` + outboundSecret + `"},` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":` +
		`{"type":"` + outboundSecret + `","ttl":"` + outboundSecret + `","scope":"` + outboundSecret + `"}}]}]}`
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(body), nil))
	encoded, _ := json.Marshal(record)
	if strings.Contains(string(encoded), outboundSecret) {
		t.Fatalf("a client string reached the record: %s", encoded)
	}
	digest := outboundDigest(outboundSecret)
	for _, key := range []string{"outbound_top_level_effort", "outbound_reasoning_effort", "outbound_output_config_effort",
		"outbound_output_format", "outbound_thinking_type", "outbound_thinking_display", "outbound_template_effort",
		"outbound_tool_choice", "outbound_tool_choice_name"} {
		if record[key] != digest {
			t.Errorf("%s = %v, want the digest", key, record[key])
		}
	}
	if policies, _ := record["outbound_cache_control_policies"].([]string); len(policies) != 1 || policies[0] != digest+"/"+digest+"/"+digest {
		t.Errorf("outbound_cache_control_policies = %v", record["outbound_cache_control_policies"])
	}
}

// Truncation, stream options and per-tool strictness reach the record.
func TestOutboundParamsReadTruncationStreamOptionsAndStrictTools(t *testing.T) {
	body := `{"model":"m","truncation":"auto","stream":true,"stream_options":{"include_usage":true,"include_obfuscation":false},` +
		`"tools":[{"type":"function","function":{"name":"a","strict":true}},{"type":"function","name":"b","strict":true},{"name":"c"}]}`
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(body), nil))
	if record["outbound_truncation"] != "auto" || record["outbound_stream_include_usage"] != true ||
		record["outbound_stream_include_obfuscation"] != false || record["outbound_strict_tool_count"] != 2 ||
		record["outbound_tool_count"] != 3 {
		t.Fatalf("record = %v", record)
	}
}

// A Chat body names its top-level reasoning budget and its chat-template
// reasoning controls.
func TestOutboundParamsReadAChatBody(t *testing.T) {
	body := `{"model":"qwen/qwen3.8-27b","messages":[],"reasoning_budget_tokens":4096,` +
		`"chat_template_kwargs":{"reasoning_effort":"low","enable_thinking":true}}`
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(body), nil))
	if record["outbound_reasoning_budget_tokens"] != int64(4096) || record["outbound_template_effort"] != "low" ||
		record["outbound_template_enable_thinking"] != true {
		t.Fatalf("record = %v", record)
	}
}

// A scalar tool_choice ("auto") is common on Chat and Responses and must not
// cost the record its other parameters.
func TestOutboundParamsReadAScalarToolChoice(t *testing.T) {
	body := `{"model":"openai/gpt-6.1-sol","tool_choice":"auto","tools":[{"type":"function"}],"parallel_tool_calls":true,"max_tokens":10}`
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(body), nil))
	if record["outbound_tool_choice"] != "auto" {
		t.Errorf("outbound_tool_choice = %v, want auto", record["outbound_tool_choice"])
	}
	if record["outbound_model"] != "openai/gpt-6.1-sol" || record["outbound_tool_count"] != 1 ||
		record["outbound_parallel_tool_calls"] != true || record["outbound_max_tokens"] != int64(10) {
		t.Fatalf("record = %v", record)
	}
}

// The record's reasoning_effort reads a Messages body's output_config.effort,
// so a Messages turn no longer logs an empty dispatched effort.
func TestDispatchedReasoningEffortReadsOutputConfig(t *testing.T) {
	ctx := &RequestContext{}
	recordDispatchedReasoningControls(ctx, []byte(`{"model":"m","output_config":{"effort":"high"},"messages":[]}`))
	if ctx.DispatchedReasoningEffort != "high" {
		t.Fatalf("dispatched effort = %q, want high", ctx.DispatchedReasoningEffort)
	}
	recordDispatchedReasoningControls(ctx, []byte(`{"model":"m","output_config":{"effort":"`+outboundSecret+`"},"messages":[]}`))
	if ctx.DispatchedReasoningEffort != outboundDigest(outboundSecret) {
		t.Fatalf("dispatched effort = %q, want the digest", ctx.DispatchedReasoningEffort)
	}
}
