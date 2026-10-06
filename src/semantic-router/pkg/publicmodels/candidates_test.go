package publicmodels

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A cell in the shape of the Rayline ARC data-plane cell: one alias, arms
// named as worker ids with a thinking suffix, every fact on the card, and one
// arm out of service. The figures are illustrative; the cell's own are pinned
// in its config repository.
const candidatesConfig = `
version: v0.3
providers:
  defaults:
    model: z-ai/glm-5.2@thinking-off
  models:
    - name: z-ai/glm-5.2@thinking-off
      provider_model_id: z-ai/glm-5.2
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 0.56
        cached_input_per_1m: 0.104
        cache_write_per_1m: 0.56
        completion_per_1m: 1.76
      backend_refs:
        - name: openrouter
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: PUBLICMODELS_CANDIDATES_TEST_KEY
    - name: qwen/qwen3.6-35b-a3b@thinking-off
      provider_model_id: qwen/qwen3.6-35b-a3b
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 0.15
        cached_input_per_1m: 0.05
        cache_write_per_1m: 0.15
        completion_per_1m: 1.0
      backend_refs:
        - name: openrouter
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: PUBLICMODELS_CANDIDATES_TEST_KEY
    - name: deepseek/deepseek-v4-flash@thinking-on
      provider_model_id: deepseek/deepseek-v4-flash
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 0.0983
        cached_input_per_1m: 0.0197
        cache_write_per_1m: 0.0983
        completion_per_1m: 0.1966
      backend_refs:
        - name: openrouter
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: PUBLICMODELS_CANDIDATES_TEST_KEY
routing:
  modelCards:
    - name: z-ai/glm-5.2@thinking-off
      capabilities: [chat, tools]
      context_window_size: 1048576
      max_output_tokens: 128000
      vision: false
    - name: qwen/qwen3.6-35b-a3b@thinking-off
      capabilities: [chat, tool_result_images, tools]
      context_window_size: 262144
      max_output_tokens: 65536
      vision: true
    - name: deepseek/deepseek-v4-flash@thinking-on
      capabilities: [chat, tools]
      context_window_size: 1000000
      max_output_tokens: 65536
      vision: false
      disabled: true
  decisions:
    - name: rayline-arc-dev
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: z-ai/glm-5.2@thinking-off
          use_reasoning: false
        - model: qwen/qwen3.6-35b-a3b@thinking-off
          use_reasoning: false
        - model: deepseek/deepseek-v4-flash@thinking-on
          use_reasoning: true
      adaptations:
        mode: bypass
      plugins:
        - type: router_replay
          configuration:
            enabled: false
      algorithm:
        type: rayline_arc
        on_error: fail_closed
        rayline_arc:
          routes_api:
            enabled: true
            checkpoint_label: arc-c82-dev
          artifact_dir: /var/lib/vllm-sr/rayline-arc
          artifact_revision: c82-dev-shadow-openrouter-v1
          encoder:
            base_url: https://encoder.example.test
            model: Qwen/Qwen3.5-0.8B
            model_revision: 2fc06364715b967f1860aea9cf38778875588b17
            expected_build_id: vllm@test
            expected_io_plugin_version: rayline-arc-io@0.1.0
            serializer_version: mtrouter-token-blocks-v2
            serving_rung: B
            required_pooling_capabilities:
              - chunked_causal_mean
            connect_timeout_seconds: 1
            total_timeout_seconds: 5
            max_retries: 0
            max_inflight_encoder_calls: 4
          episode:
            id_header: x-rayline-session
            backend: memory
            key_prefix: "vsr:rayline-arc-test:"
            acquire_timeout_seconds: 2
            lease_ttl_seconds: 3
            idle_ttl_seconds: 900
            max_in_memory_episodes: 16
            development_mode: true
global:
  router:
    auto_model_names: ["rayline/arc-dev"]
`

func candidatesListing(t *testing.T) map[string]interface{} {
	t.Helper()
	t.Setenv("PUBLICMODELS_CANDIDATES_TEST_KEY", "public-candidates-test-key")
	cfg, err := config.ParseYAMLBytes([]byte(candidatesConfig))
	if err != nil {
		t.Fatalf("the candidates config did not load: %v", err)
	}
	return marshalListing(t, NewOpenAIModelList(cfg, 123))
}

func marshalListing(t *testing.T, listing OpenAIModelList) map[string]interface{} {
	t.Helper()
	encoded, err := json.Marshal(listing)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	return body
}

func modelEntry(t *testing.T, listing map[string]interface{}, id string) map[string]interface{} {
	t.Helper()
	for _, entry := range listing["data"].([]interface{}) {
		model := entry.(map[string]interface{})
		if model["id"] == id {
			return model
		}
	}
	t.Fatalf("model %q is not listed in %v", id, listing["data"])
	return nil
}

func arcDecision(name string, refs ...config.ModelRef) config.Decision {
	return config.Decision{
		Name:      name,
		ModelRefs: refs,
		Algorithm: &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType, OnError: "fail_closed", RaylineARC: &config.RaylineARCAlgorithmConfig{}},
	}
}

func candidatesOf(t *testing.T, listing map[string]interface{}, id string) []interface{} {
	t.Helper()
	candidates, _ := modelEntry(t, listing, id)["routing"].(map[string]interface{})["candidates"].([]interface{})
	return candidates
}

// Every fact is read off the loaded config: the card's limits, verdicts and
// claims, the access binding's provider and model id, the rate card, and the
// decision's thinking mode per arm. Nothing is restated, the arms are in
// artifact order, and they are not listed as ids of their own.
func TestCandidatesAreTheARCArmsUnderTheAlias(t *testing.T) {
	listing := candidatesListing(t)
	if data := listing["data"].([]interface{}); len(data) != 1 {
		t.Fatalf("data = %v, want the alias alone: the arms are not ids a caller may send", data)
	}
	candidates := candidatesOf(t, listing, "rayline/arc-dev")
	if len(candidates) != 3 {
		t.Fatalf("candidates = %v, want the decision's three arms in order", candidates)
	}
	type expectation struct {
		model, providerModel, thinking string
		vision, tools, disabled        bool
		contextWindow, maxOutput       float64
		inputPerMTok                   float64
	}
	expected := []expectation{
		{"z-ai/glm-5.2@thinking-off", "z-ai/glm-5.2", "off", false, true, false, 1048576, 128000, 0.56},
		{"qwen/qwen3.6-35b-a3b@thinking-off", "qwen/qwen3.6-35b-a3b", "off", true, true, false, 262144, 65536, 0.15},
		{"deepseek/deepseek-v4-flash@thinking-on", "deepseek/deepseek-v4-flash", "on", false, true, true, 1000000, 65536, 0.0983},
	}
	for index, want := range expected {
		candidate := candidates[index].(map[string]interface{})
		if candidate["model"] != want.model || candidate["provider_model"] != want.providerModel || candidate["provider"] != "openrouter" {
			t.Fatalf("candidates[%d] identity = %v/%v/%v, want %s dispatched as %s by openrouter", index, candidate["model"], candidate["provider_model"], candidate["provider"], want.model, want.providerModel)
		}
		if candidate["thinking"].(map[string]interface{})["mode"] != want.thinking {
			t.Fatalf("candidates[%d].thinking = %v, want mode %s", index, candidate["thinking"], want.thinking)
		}
		if candidate["vision"] != want.vision || candidate["tools"] != want.tools || candidate["disabled"] != want.disabled {
			t.Fatalf("candidates[%d] verdicts = vision %v tools %v disabled %v, want %v %v %v", index, candidate["vision"], candidate["tools"], candidate["disabled"], want.vision, want.tools, want.disabled)
		}
		if candidate["context_window"] != want.contextWindow || candidate["max_output_tokens"] != want.maxOutput {
			t.Fatalf("candidates[%d] limits = %v/%v, want %v/%v", index, candidate["context_window"], candidate["max_output_tokens"], want.contextWindow, want.maxOutput)
		}
		pricing := candidate["pricing"].(map[string]interface{})
		if pricing["input_per_mtok"] != want.inputPerMTok || pricing["currency"] != "USD" || pricing["cache_write_per_mtok"] != want.inputPerMTok {
			t.Fatalf("candidates[%d].pricing = %v, want %v USD per input mtok and the same cache write", index, pricing, want.inputPerMTok)
		}
		for _, field := range []string{"decision", "index", "base_model"} {
			if _, present := candidate[field]; present {
				t.Fatalf("candidates[%d] publishes %s, which is not part of the contract", index, field)
			}
		}
	}
	for _, field := range []string{"decisions", "algorithm", "checkpoint"} {
		if _, present := modelEntry(t, listing, "rayline/arc-dev")["routing"].(map[string]interface{})[field]; present {
			t.Fatalf("%s is published, but nothing reads it", field)
		}
	}
}

// Only a rayline_arc decision is described. A plain decision, a Looper
// decision and a direct Looper alias resolve through paths the refs alone do
// not describe, so they carry no candidates key at all, and neither does a
// passthrough id.
func TestOnlyARCDecisionsAreDescribed(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}, IncludeConfigModelsInList: true},
		Looper:        config.LooperConfig{Endpoint: "looper:50051", Fusion: config.FusionRuntimeConfig{ModelNames: []string{"router/fusion"}}},
		Entrypoints:   []config.EntrypointMapping{{ModelNames: []string{"partner/arc"}, Recipe: "arc"}},
		Recipes: []config.RoutingRecipe{
			{Name: "default", Profile: config.RoutingProfile{Decisions: []config.Decision{
				{Name: "plain", ModelRefs: []config.ModelRef{{Model: "cheap"}}},
				{Name: "fused", Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{Model: "judge"}}, ModelRefs: []config.ModelRef{{Model: "cheap"}}},
			}}},
			{Name: "arc", Profile: config.RoutingProfile{Decisions: []config.Decision{arcDecision("arm-set", config.ModelRef{Model: "arm-0"}, config.ModelRef{Model: "arm-1"})}}},
		},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"cheap": {}, "judge": {}, "arm-0": {}, "arm-1": {}}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	for _, id := range []string{"router/auto", "router/fusion", "cheap"} {
		if _, present := modelEntry(t, listing, id)["routing"].(map[string]interface{})["candidates"]; present {
			t.Fatalf("%s publishes candidates, but it resolves through no ARC decision", id)
		}
	}
	candidates := candidatesOf(t, listing, "partner/arc")
	if len(candidates) != 2 || candidates[0].(map[string]interface{})["model"] != "arm-0" || candidates[1].(map[string]interface{})["model"] != "arm-1" {
		t.Fatalf("partner/arc candidates = %v, want the recipe's ARC arms in order", candidates)
	}
}

// A card that declares nothing reports what selection would read off it --
// capable of vision, claiming no tools, in service -- and null for a limit
// it does not state, never a zero that reads as measured. An arm no backend
// serves names no provider, and one no backend renames repeats no id.
func TestCandidateReportsNullForWhatTheCardDoesNotSay(t *testing.T) {
	t.Parallel()
	cfg := &config.RouterConfig{
		RouterOptions:      config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{arcDecision("bare", config.ModelRef{Model: "bare"})}},
		BackendModels:      config.BackendModels{ModelConfig: map[string]config.ModelParams{"bare": {}}},
	}
	candidate := candidatesOf(t, marshalListing(t, NewOpenAIModelList(cfg, 123)), "router/auto")[0].(map[string]interface{})
	for _, field := range []string{"context_window", "max_output_tokens", "pricing"} {
		value, present := candidate[field]
		if !present || value != nil {
			t.Fatalf("%s = %v, want an explicit null", field, value)
		}
	}
	if candidate["vision"] != true || candidate["tools"] != false || candidate["disabled"] != false {
		t.Fatalf("verdicts = %v, want the unmarked card's: vision true, tools false, disabled false", candidate)
	}
	for _, field := range []string{"provider", "provider_model"} {
		if _, present := candidate[field]; present {
			t.Fatalf("%s is published for an arm with no backend", field)
		}
	}
	if candidate["thinking"].(map[string]interface{})["mode"] != "off" {
		t.Fatalf("thinking = %v, want off when use_reasoning is unset", candidate["thinking"])
	}
}

// The provider facts are the primary backend's, the highest-weight endpoint
// dispatch takes, not the first listed one; an endpoint that declares no
// type is dispatched as vLLM and says so.
func TestCandidateProviderIsThePrimaryBackend(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions:      config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{arcDecision("arms", config.ModelRef{Model: "m"}, config.ModelRef{Model: "local"})}},
		BackendModels: config.BackendModels{
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "backup", Address: "127.0.0.1", Port: 8001, Weight: 1, Type: "openai"},
				{Name: "main", Address: "127.0.0.1", Port: 8002, Weight: 9, Type: "openrouter"},
				{Name: "vllm", Address: "127.0.0.1", Port: 8003, Weight: 1},
			},
			ModelConfig: map[string]config.ModelParams{
				"m":     {PreferredEndpoints: []string{"backup", "main"}, ExternalModelIDs: map[string]string{"openai": "vendor/m-mini", "openrouter": "vendor/m"}},
				"local": {PreferredEndpoints: []string{"vllm"}, ExternalModelIDs: map[string]string{"vllm": "org/local-served"}},
			},
		},
	}
	candidates := candidatesOf(t, marshalListing(t, NewOpenAIModelList(cfg, 123)), "router/auto")
	if m := candidates[0].(map[string]interface{}); m["provider"] != "openrouter" || m["provider_model"] != "vendor/m" {
		t.Fatalf("m = %v, want the primary backend's provider openrouter and id vendor/m, not the first listed", m)
	}
	if local := candidates[1].(map[string]interface{}); local["provider"] != "vllm" || local["provider_model"] != "org/local-served" {
		t.Fatalf("local = %v, want provider vllm and the vllm-resolved id", local)
	}
}

// The output limit is what the router dispatches: the card's limit capped by
// the decision's max_tokens_limit. A card with no limit stays null.
func TestMaxOutputTokensHonoursTheDecisionCap(t *testing.T) {
	decision := arcDecision("arms", config.ModelRef{Model: "m"}, config.ModelRef{Model: "unbounded"})
	decision.Plugins = []config.DecisionPlugin{{
		Type: "request_params", Configuration: config.MustStructuredPayload(map[string]interface{}{"max_tokens_limit": 4000}),
	}}
	cfg := &config.RouterConfig{
		RouterOptions:      config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{decision}},
		BackendModels:      config.BackendModels{ModelConfig: map[string]config.ModelParams{"m": {MaxOutputTokens: 32000}, "unbounded": {}}},
	}
	candidates := candidatesOf(t, marshalListing(t, NewOpenAIModelList(cfg, 123)), "router/auto")
	if m := candidates[0].(map[string]interface{}); m["max_output_tokens"] != float64(4000) {
		t.Fatalf("m.max_output_tokens = %v, want the decision cap 4000", m["max_output_tokens"])
	}
	if unbounded := candidates[1].(map[string]interface{}); unbounded["max_output_tokens"] != nil {
		t.Fatalf("unbounded.max_output_tokens = %v, want null: a cap does not bound a card that declares no limit", unbounded["max_output_tokens"])
	}
}

// The claim and the card share one spelling, so a card that claims tool
// calling under the catalog's name is read as such and nothing else is.
func TestCandidateReadsToolsUnderTheCatalogSpelling(t *testing.T) {
	t.Parallel()
	if llmprotocol.RoutingCapabilityTools != "tools" {
		t.Fatalf("RoutingCapabilityTools = %q, want the catalog's spelling", llmprotocol.RoutingCapabilityTools)
	}
	cfg := &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
		"claims":   {Capabilities: []string{"chat", "tools"}},
		"misspelt": {Capabilities: []string{"chat", "tool_calling"}},
	}}}
	claims := routingCandidateOf(cfg, config.ModelRef{Model: "claims"}, 0)
	misspelt := routingCandidateOf(cfg, config.ModelRef{Model: "misspelt"}, 0)
	if !claims.Tools || misspelt.Tools {
		t.Fatalf("tools = %v/%v, want true for the catalog spelling and false for any other", claims.Tools, misspelt.Tools)
	}
}
