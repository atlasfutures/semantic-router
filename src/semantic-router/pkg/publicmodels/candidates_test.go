package publicmodels

import (
	"encoding/json"
	"reflect"
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
      max_output_tokens: 943718
      vision: false
    - name: qwen/qwen3.6-35b-a3b@thinking-off
      capabilities: [chat, tool_result_images, tools]
      context_window_size: 262144
      max_output_tokens: 235929
      vision: true
    - name: deepseek/deepseek-v4-flash@thinking-on
      capabilities: [chat, tools]
      context_window_size: 1024000
      max_output_tokens: 384000
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

// Every fact is read off the loaded config: the card's limits, verdicts and
// claims, the access binding's provider and model id, the rate card, and the
// decision's thinking mode per candidate. Nothing is restated, and the arms
// are not listed as ids of their own.
func TestCandidatesAreListedUnderTheAliasAndNotAsIDs(t *testing.T) {
	listing := candidatesListing(t)
	if data := listing["data"].([]interface{}); len(data) != 1 {
		t.Fatalf("data = %v, want the alias alone: the arms are not ids a caller may send", data)
	}
	routing := modelEntry(t, listing, "rayline/arc-dev")["routing"].(map[string]interface{})
	candidates, _ := routing["candidates"].([]interface{})
	if len(candidates) != 3 {
		t.Fatalf("candidates = %v, want the decision's three model refs in order", routing["candidates"])
	}
	type expectation struct {
		model, providerModel, thinking string
		vision, tools, disabled        bool
		contextWindow, maxOutput       float64
		inputPerMTok                   float64
	}
	expected := []expectation{
		{"z-ai/glm-5.2@thinking-off", "z-ai/glm-5.2", "off", false, true, false, 1048576, 943718, 0.56},
		{"qwen/qwen3.6-35b-a3b@thinking-off", "qwen/qwen3.6-35b-a3b", "off", true, true, false, 262144, 235929, 0.15},
		{"deepseek/deepseek-v4-flash@thinking-on", "deepseek/deepseek-v4-flash", "on", false, true, true, 1024000, 384000, 0.0983},
	}
	for index, want := range expected {
		candidate := candidates[index].(map[string]interface{})
		for _, field := range []string{"decision", "index"} {
			if _, present := candidate[field]; present {
				t.Fatalf("candidates[%d] publishes %s: a candidate is a model, not a decision's slot", index, field)
			}
		}
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
	}
	// The candidates are listed and nothing else: no decision block, no
	// algorithm, no artifact pin. A gateway derives its document from the
	// candidates, and when the set changes the candidates change with it.
	for _, field := range []string{"decisions", "algorithm", "checkpoint"} {
		if _, present := routing[field]; present {
			t.Fatalf("%s is published, but nothing reads it", field)
		}
	}
}

// A card that declares nothing reports what selection would read off it --
// capable of vision, claiming no tools, in service -- and null for a limit
// it does not state, never a zero that reads as measured. A model no backend
// renames repeats no provider_model.
func TestCandidateReportsNullForWhatTheCardDoesNotSay(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
			Name:      "bare",
			ModelRefs: []config.ModelRef{{Model: "bare"}},
		}}},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"bare": {}}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	routing := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})
	candidate := routing["candidates"].([]interface{})[0].(map[string]interface{})
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
			t.Fatalf("%s is published for a candidate with no backend", field)
		}
	}
	if candidate["thinking"].(map[string]interface{})["mode"] != "off" {
		t.Fatalf("thinking = %v, want off when use_reasoning is unset", candidate["thinking"])
	}
}

// An alias with nothing behind it carries no candidates key at all, and a
// passthrough id never does: the list describes a virtual resolution.
func TestCandidatesAreAbsentWhereNothingResolves(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{
			AutoModelNames:            []string{"router/auto"},
			IncludeConfigModelsInList: true,
		},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"partner/backend": {}}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	for _, id := range []string{"router/auto", "partner/backend"} {
		routing := modelEntry(t, listing, id)["routing"].(map[string]interface{})
		if _, present := routing["candidates"]; present {
			t.Fatalf("%s publishes candidates %v, want none", id, routing["candidates"])
		}
	}
}

// An entrypoint alias lists its own recipe's candidates, not the default's,
// each distinct model once in first-declared order: a model two decisions
// declare is one candidate, and a model one decision reasons with and
// another does not is two.
func TestEntrypointAliasesListTheirRecipesCandidates(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		Entrypoints:   []config.EntrypointMapping{{ModelNames: []string{"partner/balanced"}, Recipe: "balanced"}},
		Recipes: []config.RoutingRecipe{
			{Name: "default", Profile: config.RoutingProfile{Decisions: []config.Decision{{
				Name: "default-decision", ModelRefs: []config.ModelRef{{Model: "cheap"}},
			}}}},
			{Name: "balanced", Profile: config.RoutingProfile{Decisions: []config.Decision{
				{Name: "code", ModelRefs: []config.ModelRef{{Model: "strong", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPointer(true)}}}},
				{Name: "chat", ModelRefs: []config.ModelRef{{Model: "cheap"}, {Model: "strong"}}},
				{Name: "math", ModelRefs: []config.ModelRef{{Model: "strong", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPointer(true)}}, {Model: "cheap"}}},
			}}},
		},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"cheap": {}, "strong": {}}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	balanced := modelEntry(t, listing, "partner/balanced")["routing"].(map[string]interface{})
	type entry struct{ model, mode string }
	var got []entry
	for _, raw := range balanced["candidates"].([]interface{}) {
		candidate := raw.(map[string]interface{})
		got = append(got, entry{candidate["model"].(string), candidate["thinking"].(map[string]interface{})["mode"].(string)})
	}
	want := []entry{{"strong", "on"}, {"cheap", "off"}, {"strong", "off"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("partner/balanced candidates = %v, want %v: each model and mode once, in first-declared order", got, want)
	}
	auto := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})
	if candidates := auto["candidates"].([]interface{}); len(candidates) != 1 || candidates[0].(map[string]interface{})["model"] != "cheap" {
		t.Fatalf("router/auto candidates = %v, want the default recipe's one", candidates)
	}
}

// A LoRA ref selects the adapter, so the candidate is the adapter: its name
// is the one selection picks and dispatch sends, and the base is named
// beside it. Its facts resolve the way the runtime resolves them, field by
// field: vision, tools, disabled and the context window are read off the
// base card, which is what ARC gating and the context filter read; the
// output limit and the pricing are the adapter's own when it states them and
// the base's otherwise. An adapter with a partial card must not read as
// claiming nothing on the fields it omits.
func TestLoRACandidatesAreTheAdapter(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
			Name: "tuned",
			ModelRefs: []config.ModelRef{
				{Model: "base", LoRAName: "base-sql"},
				{Model: "base", LoRAName: "base-legal"},
				{Model: "base"},
			},
		}}},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"base": {
				LoRAs:             []config.LoRAAdapter{{Name: "base-sql"}, {Name: "base-legal"}},
				ContextWindowSize: 32768,
				MaxOutputTokens:   12000,
				Capabilities:      []string{"chat", "tools"},
				Pricing:           config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 2},
			},
			// An adapter with a partial card of its own: priced and output-
			// limited apart from its base, declaring a window of its own
			// that nothing at runtime reads for an adapter, and silent on
			// everything else.
			"base-legal": {
				ContextWindowSize: 8192,
				MaxOutputTokens:   4000,
				Pricing:           config.ModelPricing{Currency: "USD", PromptPer1M: 3, CompletionPer1M: 6},
			},
		}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	candidates := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})["candidates"].([]interface{})
	if len(candidates) != 3 {
		t.Fatalf("candidates = %v, want the three refs", candidates)
	}
	sql := candidates[0].(map[string]interface{})
	if sql["model"] != "base-sql" || sql["base_model"] != "base" {
		t.Fatalf("candidates[0] = %v, want the adapter base-sql served under base", sql)
	}
	if sql["context_window"] != float64(32768) || sql["max_output_tokens"] != float64(12000) || sql["tools"] != true || sql["pricing"].(map[string]interface{})["input_per_mtok"] != float64(1) {
		t.Fatalf("candidates[0] = %v, want the base card's facts for an adapter with no card of its own", sql)
	}
	legal := candidates[1].(map[string]interface{})
	if legal["model"] != "base-legal" || legal["base_model"] != "base" {
		t.Fatalf("candidates[1] = %v, want the adapter base-legal served under base", legal)
	}
	if legal["max_output_tokens"] != float64(4000) || legal["pricing"].(map[string]interface{})["input_per_mtok"] != float64(3) {
		t.Fatalf("candidates[1] = %v, want the adapter's own output limit and price to win over the base, as dispatch resolves them", legal)
	}
	// The context filter and ARC gating read the ref's model, the base card,
	// for an adapter, so the listing does too: the adapter's own window is
	// not what the runtime checks a turn against.
	if legal["context_window"] != float64(32768) || legal["tools"] != true {
		t.Fatalf("candidates[1] = %v, want the base's window and tools claim, which are what the runtime reads for an adapter", legal)
	}
	plain := candidates[2].(map[string]interface{})
	if plain["model"] != "base" {
		t.Fatalf("candidates[2] = %v, want the base itself", plain)
	}
	if _, present := plain["base_model"]; present {
		t.Fatalf("candidates[2] = %v, base_model is published for a candidate that is not an adapter", plain)
	}
}

// A direct Looper alias is served by the default profile's decisions of its
// algorithm, filtered by type rather than by recipe, so it lists exactly
// those candidates. Without this the listing described auto and entrypoint
// aliases and left the orchestrated ones blank.
func TestLooperAliasesListTheirAlgorithmsCandidates(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		Looper: config.LooperConfig{
			Endpoint: "looper:50051",
			ReMoM:    config.ReMoMRuntimeConfig{ModelNames: []string{"router/remom"}},
		},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{
			{Name: "plain", ModelRefs: []config.ModelRef{{Model: "cheap"}}},
			{Name: "ensemble", Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM}, ModelRefs: []config.ModelRef{{Model: "strong"}, {Model: "cheap"}}},
		}},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"cheap": {}, "strong": {}}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	remom := modelEntry(t, listing, "router/remom")["routing"].(map[string]interface{})
	candidates := remom["candidates"].([]interface{})
	if len(candidates) != 2 {
		t.Fatalf("router/remom candidates = %v, want the remom decision's two", candidates)
	}
	if candidates[0].(map[string]interface{})["model"] != "strong" || candidates[1].(map[string]interface{})["model"] != "cheap" {
		t.Fatalf("router/remom candidates = %v, want the remom decision's strong then cheap", candidates)
	}
	// The auto alias still lists every default-profile decision, the
	// orchestrated one included: a request to it may be routed there. The
	// model both declare is listed once.
	auto := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})
	if candidates := auto["candidates"].([]interface{}); len(candidates) != 2 {
		t.Fatalf("router/auto candidates = %v, want cheap and strong once each", candidates)
	}
}

// The provider facts are the primary backend's, the highest-weight endpoint
// dispatch takes, not the first listed one. A config that lists a light
// backup first would otherwise describe the candidate by a provider the
// request never goes to.
func TestCandidateProviderIsThePrimaryBackend(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
			Name: "d", ModelRefs: []config.ModelRef{{Model: "m"}},
		}}},
		BackendModels: config.BackendModels{
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "backup", Address: "127.0.0.1", Port: 8001, Weight: 1, Type: "openai"},
				{Name: "main", Address: "127.0.0.1", Port: 8002, Weight: 9, Type: "openrouter"},
			},
			ModelConfig: map[string]config.ModelParams{"m": {
				PreferredEndpoints: []string{"backup", "main"},
				ExternalModelIDs:   map[string]string{"openai": "vendor/m-mini", "openrouter": "vendor/m"},
			}},
		},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	candidate := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})["candidates"].([]interface{})[0].(map[string]interface{})
	if candidate["provider"] != "openrouter" || candidate["provider_model"] != "vendor/m" {
		t.Fatalf("candidate = %v, want the primary backend's provider openrouter and id vendor/m, not the first listed", candidate)
	}
}

// Fusion and Flow aliases list their own algorithm's candidates, and only
// those: the three Looper surfaces are filtered by type, not pooled.
func TestFusionAndFlowAliasesListTheirOwnAlgorithmsCandidates(t *testing.T) {
	cfg := &config.RouterConfig{
		Looper: config.LooperConfig{
			Endpoint: "looper:50051",
			Fusion:   config.FusionRuntimeConfig{ModelNames: []string{"router/fusion"}},
			Flow:     config.FlowRuntimeConfig{ModelNames: []string{"router/flow"}},
		},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{
			{Name: "fused", Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion}, ModelRefs: []config.ModelRef{{Model: "a"}, {Model: "b"}}},
			{Name: "flowed", Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmWorkflows}, ModelRefs: []config.ModelRef{{Model: "c"}}},
		}},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"a": {}, "b": {}, "c": {}}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	models := func(id string) []string {
		var names []string
		for _, raw := range modelEntry(t, listing, id)["routing"].(map[string]interface{})["candidates"].([]interface{}) {
			names = append(names, raw.(map[string]interface{})["model"].(string))
		}
		return names
	}
	if got := models("router/fusion"); !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("router/fusion candidates = %v, want the fusion decision's a and b", got)
	}
	if got := models("router/flow"); !reflect.DeepEqual(got, []string{"c"}) {
		t.Fatalf("router/flow candidates = %v, want the workflows decision's c", got)
	}
}

// A route action resolves straight to its destination, so the destination
// is listed ahead of its decision's refs. The router's default_model, the
// fallback for an unmatched request, is not listed: on an ARC cell the one
// decision matches every turn, so the fallback is never reached.
func TestCandidatesIncludeRouteActionDestinationsButNotTheDefaultModel(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{
			{Name: "guard", Action: &config.DecisionAction{Type: config.DecisionActionRoute, Destination: "safe"}, ModelRefs: []config.ModelRef{{Model: "cheap"}}},
			{Name: "plain", ModelRefs: []config.ModelRef{{Model: "strong"}}},
		}},
		BackendModels: config.BackendModels{
			DefaultModel: "fallback",
			ModelConfig:  map[string]config.ModelParams{"safe": {}, "cheap": {}, "strong": {}, "fallback": {}},
		},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	var got []string
	for _, raw := range modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})["candidates"].([]interface{}) {
		got = append(got, raw.(map[string]interface{})["model"].(string))
	}
	if want := []string{"safe", "cheap", "strong"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("candidates = %v, want %v: the destination first and no default model", got, want)
	}
}

// An alias with no decisions behind it lists nothing, default model or not.
func TestDefaultModelIsNeverListed(t *testing.T) {
	for name, cfg := range map[string]*config.RouterConfig{
		"no decision anywhere": {
			RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
			BackendModels: config.BackendModels{DefaultModel: "cheap", ModelConfig: map[string]config.ModelParams{"cheap": {}}},
		},
		"decisions in another recipe": {
			RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
			Recipes: []config.RoutingRecipe{
				{Name: "default"},
				{Name: "named", Profile: config.RoutingProfile{Decisions: []config.Decision{{Name: "d", ModelRefs: []config.ModelRef{{Model: "strong"}}}}}},
			},
			BackendModels: config.BackendModels{DefaultModel: "cheap", ModelConfig: map[string]config.ModelParams{"cheap": {}, "strong": {}}},
		},
	} {
		listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
		if _, present := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})["candidates"]; present {
			t.Fatalf("%s: candidates are published for an alias with no decision behind it", name)
		}
	}
}

// An algorithm that executes models beside its refs -- a Fusion judge, a
// ReMoM synthesis model, a workflow planner -- answers the request with them,
// so they are listed after the refs, each once.
func TestCandidatesIncludeAlgorithmOwnedModels(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		Looper:        config.LooperConfig{Endpoint: "looper:50051", Fusion: config.FusionRuntimeConfig{ModelNames: []string{"router/fusion"}}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
			Name:      "fused",
			Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{Model: "judge", AnalysisModels: []string{"a", "judge"}}},
			ModelRefs: []config.ModelRef{{Model: "a"}, {Model: "b"}},
		}}},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"a": {}, "b": {}, "judge": {}}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	for _, id := range []string{"router/auto", "router/fusion"} {
		var got []string
		for _, raw := range modelEntry(t, listing, id)["routing"].(map[string]interface{})["candidates"].([]interface{}) {
			got = append(got, raw.(map[string]interface{})["model"].(string))
		}
		if want := []string{"a", "b", "judge"}; !reflect.DeepEqual(got, want) {
			t.Fatalf("%s candidates = %v, want %v: the judge listed once after the refs", id, got, want)
		}
	}
}

// An endpoint that declares no type is dispatched as vLLM, and the listing
// names that provider rather than omitting it.
func TestCandidateProviderDefaultsToVLLM(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions:      config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{Name: "d", ModelRefs: []config.ModelRef{{Model: "m"}}}}},
		BackendModels: config.BackendModels{
			VLLMEndpoints: []config.VLLMEndpoint{{Name: "local", Address: "127.0.0.1", Port: 8000, Weight: 1}},
			ModelConfig: map[string]config.ModelParams{"m": {
				PreferredEndpoints: []string{"local"},
				ExternalModelIDs:   map[string]string{"vllm": "org/m-served"},
			}},
		},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	candidate := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})["candidates"].([]interface{})[0].(map[string]interface{})
	if candidate["provider"] != "vllm" || candidate["provider_model"] != "org/m-served" {
		t.Fatalf("candidate = %v, want provider vllm and the vllm-resolved id", candidate)
	}
}

// An algorithm-owned model that is also a ref runs under that ref's reasoning
// control, so it is one candidate with that mode, not a second reasoning-off
// twin.
func TestAlgorithmOwnedModelsKeepTheirRefsReasoning(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
			Name:      "ensemble",
			Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM, ReMoM: &config.ReMoMAlgorithmConfig{SynthesisModel: "strong"}},
			ModelRefs: []config.ModelRef{{Model: "cheap"}, {Model: "strong", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPointer(true)}}},
		}}},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"cheap": {}, "strong": {}}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	candidates := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})["candidates"].([]interface{})
	if len(candidates) != 2 {
		t.Fatalf("candidates = %v, want cheap and strong once each", candidates)
	}
	strong := candidates[1].(map[string]interface{})
	if strong["model"] != "strong" || strong["thinking"].(map[string]interface{})["mode"] != "on" {
		t.Fatalf("candidates[1] = %v, want strong with reasoning on, as its ref declares", strong)
	}
}

// An algorithm-owned model whose ref names an adapter still runs the model
// itself under that ref's reasoning, as the Looper resolves it, so it is
// listed as the model with that mode and no second reasoning-off twin.
func TestAlgorithmOwnedModelsTakeReasoningFromAnAdapterRef(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
			Name:      "ensemble",
			Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM, ReMoM: &config.ReMoMAlgorithmConfig{SynthesisModel: "base"}},
			ModelRefs: []config.ModelRef{{Model: "base", LoRAName: "base-sql", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: boolPointer(true)}}},
		}}},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"base": {LoRAs: []config.LoRAAdapter{{Name: "base-sql"}}}}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	candidates := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})["candidates"].([]interface{})
	if len(candidates) != 2 {
		t.Fatalf("candidates = %v, want the adapter and the base once each", candidates)
	}
	base := candidates[1].(map[string]interface{})
	if base["model"] != "base" || base["thinking"].(map[string]interface{})["mode"] != "on" {
		t.Fatalf("candidates[1] = %v, want the base with reasoning on, as its adapter ref declares", base)
	}
	if _, present := base["base_model"]; present {
		t.Fatalf("candidates[1] = %v, the synthesis model is executed as itself, not as the adapter", base)
	}
}

// A model both an ARC decision and a plain decision declare is reachable
// through the plain one whatever its card says, so it is not reported
// disabled, in either declaration order.
func TestDisabledMergesAcrossDecisionsRegardlessOfOrder(t *testing.T) {
	disabled := true
	arc := config.Decision{Name: "arc", Algorithm: &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType, OnError: "fail_closed", RaylineARC: &config.RaylineARCAlgorithmConfig{}}, ModelRefs: []config.ModelRef{{Model: "shared"}}}
	plain := config.Decision{Name: "plain", ModelRefs: []config.ModelRef{{Model: "shared"}}}
	for name, decisions := range map[string][]config.Decision{"arc first": {arc, plain}, "plain first": {plain, arc}} {
		cfg := &config.RouterConfig{
			RouterOptions:      config.RouterOptions{AutoModelNames: []string{"router/auto"}},
			IntelligentRouting: config.IntelligentRouting{Decisions: decisions},
			BackendModels:      config.BackendModels{ModelConfig: map[string]config.ModelParams{"shared": {Disabled: &disabled}}},
		}
		listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
		candidates := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})["candidates"].([]interface{})
		if len(candidates) != 1 || candidates[0].(map[string]interface{})["disabled"] != false {
			t.Fatalf("%s: candidates = %v, want shared once and not disabled", name, candidates)
		}
	}
}

// The disabled flag is reported as selection enforces it: a rayline_arc
// decision masks a disabled arm, so its candidate says so; a plain decision
// still dispatches the model, so its candidate says false.
func TestDisabledIsReportedOnlyWhereItIsEnforced(t *testing.T) {
	disabled := true
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{
			{Name: "plain", ModelRefs: []config.ModelRef{{Model: "retired"}}},
			{Name: "arc", Algorithm: &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType, OnError: "fail_closed", RaylineARC: &config.RaylineARCAlgorithmConfig{}}, ModelRefs: []config.ModelRef{{Model: "retired-arm"}}},
		}},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{"retired": {Disabled: &disabled}, "retired-arm": {Disabled: &disabled}}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	candidates := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})["candidates"].([]interface{})
	if candidates[0].(map[string]interface{})["disabled"] != false || candidates[1].(map[string]interface{})["disabled"] != true {
		t.Fatalf("candidates = %v, want disabled false under the plain decision and true under the ARC decision", candidates)
	}
}

// A route action falls back to a ref's model, not an adapter the ref names,
// so its refs are described as the model.
func TestRouteActionRefsAreDescribedAsTheModelTheyDispatch(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{"router/auto"}},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
			Name:      "guard",
			Action:    &config.DecisionAction{Type: config.DecisionActionRoute, Destination: "safe"},
			ModelRefs: []config.ModelRef{{Model: "base", LoRAName: "base-sql"}},
		}}},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"safe": {}, "base": {LoRAs: []config.LoRAAdapter{{Name: "base-sql"}}},
		}},
	}
	listing := marshalListing(t, NewOpenAIModelList(cfg, 123))
	candidates := modelEntry(t, listing, "router/auto")["routing"].(map[string]interface{})["candidates"].([]interface{})
	fallback := candidates[1].(map[string]interface{})
	if fallback["model"] != "base" {
		t.Fatalf("candidates[1] = %v, want the base model the fallback dispatches, not the adapter", fallback)
	}
	if _, present := fallback["base_model"]; present {
		t.Fatalf("candidates[1] = %v, base_model is published for a ref described as its model", fallback)
	}
}

// The claim and the card share one spelling, so a card that claims tool
// calling under the catalog's name is read as such and nothing else is.
func TestCandidateReadsToolsUnderTheCatalogSpelling(t *testing.T) {
	if llmprotocol.RoutingCapabilityTools != "tools" {
		t.Fatalf("RoutingCapabilityTools = %q, want the catalog's spelling", llmprotocol.RoutingCapabilityTools)
	}
	cfg := &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
		"claims":   {Capabilities: []string{"chat", "tools"}},
		"misspelt": {Capabilities: []string{"chat", "tool_calling"}},
	}}}
	claims := routingCandidateOf(cfg, config.ModelRef{Model: "claims"}, false)
	misspelt := routingCandidateOf(cfg, config.ModelRef{Model: "misspelt"}, false)
	if !claims.Tools || misspelt.Tools {
		t.Fatalf("tools = %v/%v, want true for the catalog spelling and false for any other", claims.Tools, misspelt.Tools)
	}
}

func boolPointer(value bool) *bool { return &value }
