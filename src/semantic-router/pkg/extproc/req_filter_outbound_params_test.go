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
		`"cache_control":{"type":"ephemeral"},` +
		`"system":[{"type":"text","text":"` + outboundSecret + `","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"` + outboundSecret + `"},` +
		`{"role":"user","content":[{"type":"text","text":"` + outboundSecret + `","cache_control":{"type":"ephemeral"}}]}],` +
		// A schema property named cache_control is data, not a directive.
		`"tools":[{"name":"Bash","description":"` + outboundSecret + `","input_schema":{"properties":{"cache_control":{"type":"string"}}}},` +
		`{"name":"Edit","input_schema":{},"cache_control":{"type":"ephemeral"}}],` +
		`"top_p":0.9,"top_k":40,"seed":7,"frequency_penalty":0.1,"presence_penalty":0.2,"n":1,"parallel_tool_calls":false,` +
		`"thinking":{"type":"enabled","budget_tokens":4096},"output_config":{"effort":"high"},` +
		`"provider":{"only":["anthropic"],"order":["anthropic"],"allow_fallbacks":false}}`
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
		"outbound_thinking_budget_tokens": int64(4096),
		"outbound_max_tokens":             int64(32000),
		"outbound_temperature":            0.5,
		"outbound_stream":                 true,
		"outbound_tool_count":             2,
		"outbound_cache_control_markers":  4,
		"outbound_top_p":                  0.9,
		"outbound_top_k":                  int64(40),
		"outbound_seed":                   int64(7),
		"outbound_frequency_penalty":      0.1,
		"outbound_presence_penalty":       0.2,
		"outbound_n":                      int64(1),
		"outbound_parallel_tool_calls":    false,
		"outbound_prompt_cache_key":       false,
	}
	for key, value := range want {
		if fmt.Sprint(record[key]) != fmt.Sprint(value) {
			t.Errorf("%s = %v, want %v", key, record[key], value)
		}
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
		`"max_output_tokens":8000,"prompt_cache_key":"episode-1"}`
	dispatch := &providerDispatch{targetFormat: llmprotocol.OpenAIResponsesV1, profile: &config.ProviderProfile{Type: "openai"}}
	record := map[string]interface{}{}
	appendOutboundParamFields(record, readDispatchedOutboundParams([]byte(body), dispatch))
	if record["outbound_reasoning_effort"] != "medium" || record["outbound_reasoning_max_tokens"] != int64(2048) ||
		record["outbound_max_tokens"] != int64(8000) || record["outbound_prompt_cache_key"] != true ||
		record["outbound_provider"] != "openai" || record["outbound_tool_count"] != 0 {
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

// The record's reasoning_effort reads a Messages body's output_config.effort,
// so a Messages turn no longer logs an empty dispatched effort.
func TestDispatchedReasoningEffortReadsOutputConfig(t *testing.T) {
	ctx := &RequestContext{}
	recordDispatchedReasoningControls(ctx, []byte(`{"model":"m","output_config":{"effort":"high"},"messages":[]}`))
	if ctx.DispatchedReasoningEffort != "high" {
		t.Fatalf("dispatched effort = %q, want high", ctx.DispatchedReasoningEffort)
	}
}
