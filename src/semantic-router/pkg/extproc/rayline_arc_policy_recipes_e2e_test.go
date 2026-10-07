//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const (
	keepAlias   = "rayline/arc-keep-test"
	stripAlias  = "rayline/arc-strip-test"
	keepModel   = "rayline/arc/keep"
	stripModel  = "rayline/arc/strip"
	stripRecipe = "strip"
)

// Several policy packages in one cell, one per recipe, chosen by the
// request's model name: each recipe arms on its own service, decides from its
// own package, and keeps its own episodes, even when two sessions share a
// session id.
func TestPolicyPackagesServeSideBySideByModelName(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := []config.RaylineARCPolicyBinding{
		policyAction("think", "none", "think-trained", policyTestEffort("high"), nil, ""),
		policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
	}
	catalog := []string{actions[0].ActionID, actions[1].ActionID}
	keepSHA, stripSHA := strings.Repeat("1", 64), strings.Repeat("2", 64)
	keep := newFakePolicyService(t, keepAlias, keepSHA, catalog)
	strip := newFakePolicyService(t, stripAlias, stripSHA, catalog)
	strip.cold.Store(true)
	keep.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions[0].ActionID })
	strip.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions[1].ActionID })

	router, err := NewOpenAIRouter(writePolicyRecipesConfig(t, keep, strip, actions))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitRecipeSelectorArmed(t, router, config.DefaultRecipeName)

	// The strip service is still loading: its recipe fails closed while the
	// keep recipe serves.
	if status := recipeRequestStatus(t, router, stripModel, "shared-session"); status != 503 {
		t.Fatalf("a request for the unarmed recipe answered %d, want 503", status)
	}
	strip.cold.Store(false)
	awaitRecipeSelectorArmed(t, router, stripRecipe)

	// Interleave two turns of the same session id on both recipes.
	turn := func(model string, messages string) *RequestContext {
		body, ctx := dispatchRecipeRequest(t, router, model, "shared-session", messages)
		if _, err := router.handleResponseHeaders(arcResponseHeaders("200"), ctx); err != nil {
			t.Fatalf("commit %s: %v", model, err)
		}
		completeTestResponse(t, ctx)
		finalizeSelectionProcessTerminal(ctx)
		want := "vendor/think"
		if model == stripModel {
			want = "vendor/off"
		}
		if got := string(body["model"]); got != `"`+want+`"` {
			t.Fatalf("%s dispatched %s, want %s", model, got, want)
		}
		return ctx
	}
	first := `{"role":"user","content":"fix it"}`
	second := first + `,{"role":"assistant","content":"done"},{"role":"user","content":"next"}`
	turn(keepModel, first)
	turn(stripModel, first)
	turn(keepModel, second)
	turn(stripModel, second)

	for name, fake := range map[string]*fakePolicyService{"keep": keep, "strip": strip} {
		requests := fake.received()
		if len(requests) != 2 {
			t.Fatalf("%s service decided %d turns, want 2", name, len(requests))
		}
		want := actions[0].ActionID
		if name == "strip" {
			want = actions[1].ActionID
		}
		for _, request := range requests {
			if request.Package.Alias != fake.alias || request.Package.PackageSHA256 != fake.sha256 {
				t.Fatalf("%s service was asked for package %+v", name, request.Package)
			}
		}
		// Each recipe's second turn sees only its own first reply, in its
		// own epoch: the episodes did not cross.
		if attribution := requests[1].Attribution; len(attribution) != 1 || attribution[0].ActionID != want ||
			requests[1].ContextEpoch != "0" {
			t.Fatalf("%s second turn: attribution %+v epoch %s", name, attribution, requests[1].ContextEpoch)
		}
	}
}

func awaitRecipeSelectorArmed(t *testing.T, router *OpenAIRouter, recipe config.RecipeName) {
	t.Helper()
	registry := router.RecipeModelSelectors[recipe]
	if registry == nil {
		t.Fatalf("recipe %s has no selector registry", recipe)
	}
	registered, ok := registry.Get(selection.MethodRaylineARC)
	if !ok {
		t.Fatalf("recipe %s has no ARC selector", recipe)
	}
	selector := registered.(*raylineARCSelector)
	deadline := time.Now().Add(20 * time.Second)
	for selector.armedComponents() == nil {
		if time.Now().After(deadline) {
			t.Fatalf("recipe %s never armed", recipe)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func recipeRequestHeaders(model, session string) *ext_proc.ProcessingRequest_RequestHeaders {
	return &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/messages"},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: session},
		}},
	}}
}

func recipeRequestBody(model, messages string) *ext_proc.ProcessingRequest_RequestBody {
	body := `{"model":"` + model + `","max_tokens":1024,"messages":[` + messages + `]}`
	return &ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true}}
}

func newRecipeRequestContext(model string) *RequestContext {
	return &RequestContext{
		Headers: map[string]string{}, RequestID: "recipes-" + model + "-" + fmt.Sprint(time.Now().UnixNano()),
		StartTime: time.Now(), TraceContext: context.Background(),
	}
}

func recipeRequestStatus(t *testing.T, router *OpenAIRouter, model, session string) int {
	t.Helper()
	ctx := newRecipeRequestContext(model)
	if _, err := router.handleRequestHeaders(recipeRequestHeaders(model, session), ctx); err != nil {
		t.Fatal(err)
	}
	response, err := router.handleRequestBody(recipeRequestBody(model, `{"role":"user","content":"go"}`), ctx)
	if err != nil {
		t.Fatal(err)
	}
	finalizeSelectionProcessTerminal(ctx)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		return int(immediate.GetStatus().GetCode())
	}
	return 200
}

func dispatchRecipeRequest(t *testing.T, router *OpenAIRouter, model, session, messages string) (map[string]json.RawMessage, *RequestContext) {
	t.Helper()
	ctx := newRecipeRequestContext(model)
	if _, err := router.handleRequestHeaders(recipeRequestHeaders(model, session), ctx); err != nil {
		t.Fatal(err)
	}
	response, err := router.handleRequestBody(recipeRequestBody(model, messages), ctx)
	if err != nil {
		t.Fatal(err)
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		t.Fatalf("%s refused: %d %s", model, immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatal(err)
	}
	return body, ctx
}

func writePolicyRecipesConfig(t *testing.T, keep, strip *fakePolicyService, actions []config.RaylineARCPolicyBinding) string {
	t.Helper()
	bindings := func(indent string) string {
		var out strings.Builder
		for _, action := range actions {
			fmt.Fprintf(&out, "%s- action_id: %s\n%s  worker: %s\n%s  level: none\n%s  model: %s\n%s  effort: %s\n",
				indent, action.ActionID, indent, action.Worker, indent, indent, action.Model, indent, *action.Effort)
		}
		return strings.TrimRight(out.String(), "\n")
	}
	decision := func(name, indent string, fake *fakePolicyService, prefix string) string {
		block := `- name: NAME
  description: Policy-service ARC route
  priority: 100
  rules:
    operator: AND
    conditions: []
  modelRefs:
    - model: think
      use_reasoning: true
    - model: off
      use_reasoning: false
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
      readiness_wait_seconds: -1
      policy_service:
        base_url: URL
        total_timeout_seconds: 5
        package_alias: ALIAS
        package_sha256: "SHA"
        bindings:
BINDINGS
      episode:
        id_header: x-rayline-session
        backend: memory
        key_prefix: "PREFIX"
        acquire_timeout_seconds: 2
        lease_ttl_seconds: 3
        idle_ttl_seconds: 900
        max_in_memory_episodes: 16
        development_mode: true`
		block = strings.NewReplacer("NAME", name, "URL", fake.URL(), "ALIAS", fake.alias, "SHA", fake.sha256,
			"PREFIX", prefix, "BINDINGS", bindings("          ")).Replace(block)
		lines := strings.Split(block, "\n")
		for index := range lines {
			if lines[index] != "" {
				lines[index] = indent + lines[index]
			}
		}
		return strings.Join(lines, "\n")
	}
	rendered := strings.NewReplacer(
		"{{KEEP_DECISION}}", decision("rayline-arc-keep", "    ", keep, "vsr:keep:"),
		"{{STRIP_DECISION}}", decision("rayline-arc-strip", "        ", strip, "vsr:strip:"),
	).Replace(policyRecipesConfigTemplate)
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

const policyRecipesConfigTemplate = `version: v0.3

providers:
  defaults:
    model: think
  models:
    - name: think
      provider_model_id: vendor/think
      api_format: openai
      pricing: {currency: USD, prompt_per_1m: 1, cached_input_per_1m: 0.1, cache_write_per_1m: 1.25, completion_per_1m: 5}
      backend_refs:
        - name: openrouter-think
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: POLICY_E2E_PROVIDER_KEY
    - name: off
      provider_model_id: vendor/off
      api_format: openai
      pricing: {currency: USD, prompt_per_1m: 1, cached_input_per_1m: 0.1, cache_write_per_1m: 1.25, completion_per_1m: 5}
      backend_refs:
        - name: openrouter-off
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: POLICY_E2E_PROVIDER_KEY

routing:
  modelCards:
    - name: think
      modality: text
    - name: off
      modality: text
  decisions:
{{KEEP_DECISION}}

entrypoints:
  - model_names: ["rayline/arc/strip"]
    recipe: strip

recipes:
  - name: strip
    routing:
      decisions:
{{STRIP_DECISION}}

global:
  router:
    auto_model_names: ["rayline/arc/keep"]
  stores:
    semantic_cache:
      enabled: false
  model_catalog:
    embeddings:
      semantic:
        mmbert_model_path: ""
        qwen3_model_path: ""
        gemma_model_path: ""
        bert_model_path: ""
        multimodal_model_path: ""
    modules:
      prompt_guard:
        enabled: false
        model_ref: ""
        model_id: ""
        jailbreak_mapping_path: ""
        use_mmbert_32k: false
      classifier:
        domain:
          model_ref: ""
          model_id: ""
          category_mapping_path: ""
          use_mmbert_32k: false
        pii:
          model_ref: ""
          model_id: ""
          pii_mapping_path: ""
          use_mmbert_32k: false
      feedback_detector:
        enabled: false
        model_ref: ""
        model_id: ""
        use_mmbert_32k: false
`

// Only the policy-service mode is recipe-scoped, and two recipes may not share
// a Redis episode namespace; memory stores are private to their selector.
func TestRaylineARCRecipeRefusal(t *testing.T) {
	policy := &config.RaylineARCAlgorithmConfig{PolicyService: &config.RaylineARCPolicyServiceConfig{}}
	redis := func(prefix string) config.RaylineARCEpisodeConfig {
		return config.RaylineARCEpisodeConfig{Backend: config.RaylineARCBackendRedis, KeyPrefix: prefix,
			Redis: config.RaylineARCRedisConfig{Address: "10.0.0.1:6379"}}
	}
	taken := map[string]config.RecipeName{raylineARCEpisodeNamespace(redis("vsr:keep:")): config.DefaultRecipeName}
	cases := map[string]struct {
		arc     *config.RaylineARCAlgorithmConfig
		episode config.RaylineARCEpisodeConfig
		want    string
	}{
		"a policy recipe on its own prefix": {policy, redis("vsr:strip:"), ""},
		"a policy recipe on memory":         {policy, config.RaylineARCEpisodeConfig{Backend: config.RaylineARCBackendMemory}, ""},
		"a policy recipe on a taken prefix": {policy, redis("vsr:keep:"), "episode_namespace_shared_with_default"},
		"an artifact-mode recipe":           {&config.RaylineARCAlgorithmConfig{}, redis("vsr:other:"), "recipe_requires_policy_service"},
	}
	for name, test := range cases {
		if got := raylineARCRecipeRefusal(test.arc, raylineARCEpisodeNamespace(test.episode), taken); got != test.want {
			t.Fatalf("%s: refusal = %q, want %q", name, got, test.want)
		}
	}
}

func TestRaylineARCEpisodeStoreFollowsTheRecipe(t *testing.T) {
	defaultStore, _ := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 1, IdleTTL: time.Minute})
	stripStore, _ := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 1, IdleTTL: time.Minute})
	router := &OpenAIRouter{
		RaylineARCEpisodeStore:        defaultStore,
		RaylineARCRecipeEpisodeStores: map[config.RecipeName]raylinearc.EpisodeStore{stripRecipe: stripStore},
	}
	if router.raylineARCEpisodeStoreFor(&RequestContext{}) != defaultStore {
		t.Fatal("a default-recipe request did not get the default store")
	}
	ctx := &RequestContext{}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: stripRecipe})
	if router.raylineARCEpisodeStoreFor(ctx) != stripStore {
		t.Fatal("a strip-recipe request did not get the strip store")
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "unregistered"})
	if router.raylineARCEpisodeStoreFor(ctx) != nil {
		t.Fatal("a recipe with no store borrowed another's")
	}
}
