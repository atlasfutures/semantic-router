package extproc

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A format the policy service cannot read is refused before the episode lease
// (Responses is read, as openai_responses),
// so the episode is untouched and nothing waits on a service call.
func TestPolicyModeRefusesAnUnreadableFormatBeforeTheLease(t *testing.T) {
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{RaylineARCEpisodeStore: store}
	requestContext := &RequestContext{
		Headers:      map[string]string{"x-rayline-session": t.Name()},
		SourceFormat: llmprotocol.WireFormat("unsupported.format.v1"),
		SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "go"}}},
		}},
		TraceContext: context.Background(),
	}
	algorithm := &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType, RaylineARC: &config.RaylineARCAlgorithmConfig{
		PolicyService: &config.RaylineARCPolicyServiceConfig{},
		Episode:       config.RaylineARCEpisodeConfig{IDHeader: "x-rayline-session", AcquireTimeoutSeconds: 1, LeaseTTLSeconds: 30},
	}}
	selectionContext := router.buildRaylineARCSelectionContext(algorithm, requestContext,
		[]config.ModelRef{{Model: "arm-0"}, {Model: "arm-1"}}, raylineARCEpisodeRequired)
	if selectionContext.PreparationFailure != arcFailurePolicyRequestFormat || requestContext.RaylineARCTransaction != nil {
		t.Fatalf("failure %q, transaction %v", selectionContext.PreparationFailure, requestContext.RaylineARCTransaction)
	}
	// The episode was never leased: a second request acquires it at once.
	lease, _, err := store.Prepare(context.Background(), raylinearc.HashEpisodeID(t.Name()), 2)
	if err != nil {
		t.Fatalf("the refused request left the episode leased: %v", err)
	}
	_ = store.Abort(context.Background(), lease)
}

// A service error code becomes a metric label, so only the contract's codes
// pass through, and the service's own back-pressure answers 429.
func TestPolicyServiceFailureClassesAreBounded(t *testing.T) {
	cases := map[string]string{
		"session_busy":                     "policy_service_session_busy",
		"backend_unavailable":              "policy_service_backend_unavailable",
		"a-new-code\nwith text":            "policy_service_service_error",
		"context_exceeds_encoder_capacity": "policy_service_context_exceeds_encoder_capacity",
	}
	for code, want := range cases {
		fixture := newPolicySelectorFixture(t, "")
		bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
		fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
		fixture.fake.failNext(code)
		state, _ := raylinearc.NewEpisodeState(2)
		_, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
		var failure *raylineARCSelectionFailure
		if !errors.As(err, &failure) || failure.class != want {
			t.Fatalf("%q: error = %v, want class %s", code, err, want)
		}
	}
	for class, contended := range map[string]bool{
		"policy_service_session_busy":        true,
		"policy_service_session_capacity":    true,
		"policy_service_backend_unavailable": false,
	} {
		if selectionFailureIsContended(class) != contended {
			t.Fatalf("%s contended = %v", class, !contended)
		}
	}
}

func TestPolicySessionActionIsBounded(t *testing.T) {
	for action, want := range map[string]string{"appended": "appended", "reused": "reused", "evicted-by-gc": ""} {
		if got := boundedPolicySessionAction(action); got != want {
			t.Fatalf("%q -> %q, want %q", action, got, want)
		}
	}
}

// A response naming another package than the one requested is not served:
// readiness armed against the configured package, and nothing re-checks it.
func TestPolicySelectorRefusesAResponseFromAnotherPackage(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	fixture.fake.answerAs = &raylinearc.PolicyPackageRef{Alias: policyTestAlias, PackageSHA256: strings.Repeat("b", 64)}
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "policy_package_mismatch" {
		t.Fatalf("error = %v, want class policy_package_mismatch", err)
	}
}

func policyReadinessConfig(mutate func(*config.RouterConfig)) (*config.RouterConfig, *config.Decision) {
	write := 1.25
	cfg := &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"think": {
				PreferredEndpoints: []string{"openrouter"},
				Pricing:            config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 5, CachedInputPer1M: 0.1, CacheWritePer1M: &write},
			},
		},
		VLLMEndpoints: []config.VLLMEndpoint{{
			Name: "openrouter", Address: "openrouter.ai", Port: 443, Type: "openai",
			APIKey: "resolved-key", APIKeyEnvName: "OPENROUTER_API_KEY", ProviderProfileName: "openrouter",
		}},
		ProviderProfiles: map[string]config.ProviderProfile{
			"openrouter": {Type: "openai", BaseURL: "https://openrouter.ai/api/v1"},
		},
	}}
	if mutate != nil {
		mutate(cfg)
	}
	return cfg, &config.Decision{ModelRefs: []config.ModelRef{{Model: "think"}}}
}

// The policy mode keeps the artifact mode's dispatch contract where it needs
// no manifest: an owned credential, the default auth shape, https, and USD
// pricing with a cache-write rate.
func TestPolicyDispatchReadiness(t *testing.T) {
	if cfg, decision := policyReadinessConfig(nil); !raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("a sound worker was refused")
	}
	cases := map[string]func(*config.RouterConfig){
		"an inline key": func(c *config.RouterConfig) { c.VLLMEndpoints[0].APIKeyInline = true },
		"a missing key": func(c *config.RouterConfig) { c.VLLMEndpoints[0].APIKey = "" },
		"no key env":    func(c *config.RouterConfig) { c.VLLMEndpoints[0].APIKeyEnvName = "" },
		"plain http": func(c *config.RouterConfig) {
			c.ProviderProfiles["openrouter"] = config.ProviderProfile{Type: "openai", BaseURL: "http://openrouter.ai/api/v1"}
		},
		"a custom chat path": func(c *config.RouterConfig) {
			c.ProviderProfiles["openrouter"] = config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1", ChatPath: "/x"}
		},
		"a custom auth header": func(c *config.RouterConfig) {
			c.ProviderProfiles["openrouter"] = config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1", AuthHeader: "x-caller-key"}
		},
		"an unsupported provider type": func(c *config.RouterConfig) { c.VLLMEndpoints[0].Type = "vllm" },
		"no pricing": func(c *config.RouterConfig) {
			params := c.ModelConfig["think"]
			params.Pricing = config.ModelPricing{}
			c.ModelConfig["think"] = params
		},
		"no cache-write rate": func(c *config.RouterConfig) {
			params := c.ModelConfig["think"]
			params.Pricing.CacheWritePer1M = nil
			c.ModelConfig["think"] = params
		},
	}
	for name, mutate := range cases {
		if cfg, decision := policyReadinessConfig(mutate); raylineARCPolicyDispatchReady(cfg, decision) {
			t.Fatalf("%s: readiness passed", name)
		}
	}
}

// With the cap spent, a decide call is shed before it reaches the service,
// and answers 429 like the artifact mode's encoder admission.
func TestPolicySelectorShedsBeyondItsAdmissionCap(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	armed := fixture.selector.armedComponents()
	gate := raylinearc.NewAdmissionGate(1)
	fixture.selector.arm(&raylineARCArmedComponents{scorer: armed.scorer, admission: gate, policy: armed.policy})
	release, err := gate.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	state, _ := raylinearc.NewEpisodeState(2)
	_, err = fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || !selectionFailureIsContended(failure.class) {
		t.Fatalf("error = %v, want a contended class", err)
	}
	if calls := len(fixture.fake.received()); calls != 0 {
		t.Fatalf("a shed request reached the service %d times", calls)
	}
	release()
	if _, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"})); err != nil {
		t.Fatalf("after release: %v", err)
	}
}

// encoder_latency keeps its artifact-mode meaning: the service's encode time,
// not the decide round trip, which is reported on its own.
func TestPolicySelectorReportsEncodeAndRoundTripSeparately(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	calls := 0
	fixture.selector.now = func() time.Time {
		calls++
		return time.Unix(0, 0).Add(time.Duration(calls) * 300 * time.Millisecond)
	}
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatal(err)
	}
	// The shared fixture's timing_ms.encode is 41.5.
	if got := result.RaylineARC.EncoderLatency; got != 41500*time.Microsecond {
		t.Fatalf("encoder latency = %v, want the service's encode time", got)
	}
	if result.RaylineARC.PolicyLatency != 300*time.Millisecond {
		t.Fatalf("policy latency = %v, want the timed round trip", result.RaylineARC.PolicyLatency)
	}
}

// The worker manifest carries what the artifact manifest would: the provider
// pin, the thinking mode and the dispatch backend.
func TestPolicyWorkerManifestCarriesTheCardsFacts(t *testing.T) {
	on, fallbacks := true, false
	cfg, _ := policyReadinessConfig(func(c *config.RouterConfig) {
		params := c.ModelConfig["think"]
		params.ProviderPreferences = &config.OpenRouterProviderPreferences{Order: []string{"z-ai", "chutes"}, AllowFallbacks: &fallbacks}
		c.ModelConfig["think"] = params
	})
	worker := policyWorkerManifest(cfg, config.ModelRef{Model: "think", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &on}})
	if worker.OpenRouterProviderSlug != "z-ai" || len(worker.OpenRouterProviderOrder) != 2 || worker.OpenRouterAllowFallbacks ||
		worker.ThinkingMode != "on" || worker.EffectiveDispatchBackend() != raylinearc.DispatchOpenRouter {
		t.Fatalf("worker = %+v", worker)
	}
	// With fallbacks allowed the first preference need not serve, so no
	// provider is claimed for the worker.
	allowed := true
	cfg.ModelConfig["think"] = func() config.ModelParams {
		params := cfg.ModelConfig["think"]
		params.ProviderPreferences = &config.OpenRouterProviderPreferences{Order: []string{"z-ai"}, AllowFallbacks: &allowed}
		return params
	}()
	if worker := policyWorkerManifest(cfg, config.ModelRef{Model: "think"}); worker.OpenRouterProviderSlug != "" {
		t.Fatalf("a fallback-allowed pin claimed provider %q", worker.OpenRouterProviderSlug)
	}
	cfg, _ = policyReadinessConfig(func(c *config.RouterConfig) {
		c.ProviderProfiles["openrouter"] = config.ProviderProfile{Type: "anthropic", BaseURL: "https://api.anthropic.com"}
	})
	if worker := policyWorkerManifest(cfg, config.ModelRef{Model: "think"}); worker.EffectiveDispatchBackend() == raylinearc.DispatchOpenRouter ||
		worker.ThinkingMode != "off" {
		t.Fatalf("a non-OpenRouter worker = %+v", worker)
	}
}

// A service that does not report its encode time leaves encoder latency
// unknown: the selection log omits the field rather than logging 0.
func TestPolicySelectorLeavesAnUnreportedEncodeTimeUnknown(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	fixture.fake.encodeUnreported = true
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatal(err)
	}
	if !result.RaylineARC.EncoderLatencyUnknown {
		t.Fatal("an unreported encode time was not marked unknown")
	}
	logs := captureLogs(t)
	observeRaylineARCSelection(&RequestContext{RequestID: "r"}, result.RaylineARC)
	fields := findLogEvent(t, logs, "rayline_arc_selection")
	if _, present := fields["encoder_latency_millis"]; present {
		t.Fatalf("encoder_latency_millis logged for an unknown encode time: %v", fields["encoder_latency_millis"])
	}
	if _, present := fields["policy_latency_millis"]; !present {
		t.Fatal("policy_latency_millis missing")
	}

	// A reported time is logged, as in the artifact mode.
	fixture.fake.encodeUnreported = false
	result, err = fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatal(err)
	}
	logs = captureLogs(t)
	observeRaylineARCSelection(&RequestContext{RequestID: "r"}, result.RaylineARC)
	if fields := findLogEvent(t, logs, "rayline_arc_selection"); fields["encoder_latency_millis"] != int64(41) {
		t.Fatalf("encoder_latency_millis = %v, want 41", fields["encoder_latency_millis"])
	}
}

// An Anthropic worker must keep its provider's own credential header and
// prefix, as an OpenAI-compatible one must keep the default shape.
func TestPolicyDispatchReadinessHoldsAnthropicToItsDefaultAuth(t *testing.T) {
	anthropic := func(profile config.ProviderProfile) (*config.RouterConfig, *config.Decision) {
		return policyReadinessConfig(func(c *config.RouterConfig) {
			c.VLLMEndpoints[0].Type = "anthropic"
			c.ProviderProfiles["openrouter"] = profile
		})
	}
	if cfg, decision := anthropic(config.ProviderProfile{Type: "anthropic", BaseURL: "https://api.anthropic.com"}); !raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("an Anthropic worker on its default auth was refused")
	}
	if cfg, decision := anthropic(config.ProviderProfile{Type: "anthropic", BaseURL: "https://api.anthropic.com", AuthHeader: "x-caller-key"}); raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("an Anthropic worker with a custom auth header passed readiness")
	}
}

// The manifest describes the route dispatch takes: the primary backend.
func TestPolicyWorkerManifestFollowsThePrimaryBackend(t *testing.T) {
	cfg, _ := policyReadinessConfig(func(c *config.RouterConfig) {
		params := c.ModelConfig["think"]
		params.PreferredEndpoints = []string{"direct", "openrouter"}
		params.ExternalModelIDs = map[string]string{"openai": "vendor/think"}
		c.ModelConfig["think"] = params
		c.VLLMEndpoints[0].Weight = 10
		c.VLLMEndpoints = append([]config.VLLMEndpoint{{
			Name: "direct", Address: "api.example.com", Port: 443, Type: "openai", Weight: 1,
			APIKey: "resolved-key", APIKeyEnvName: "DIRECT_KEY", ProviderProfileName: "direct",
		}}, c.VLLMEndpoints...)
		c.ProviderProfiles["direct"] = config.ProviderProfile{Type: "openai", BaseURL: "https://api.example.com/v1"}
	})
	if _, primary, _, _ := cfg.ResolvePrimaryBackendForModel("think"); primary != "openrouter" {
		t.Fatalf("the higher-weight endpoint is not primary: %q", primary)
	}
	// The first-listed endpoint is a direct OpenAI one; the primary is
	// OpenRouter, and the manifest must say so.
	if worker := policyWorkerManifest(cfg, config.ModelRef{Model: "think"}); worker.EffectiveDispatchBackend() != raylinearc.DispatchOpenRouter {
		t.Fatalf("manifest backend %q, want the primary's openrouter", worker.EffectiveDispatchBackend())
	}
}

// An artifact-mode selection logs no policy-service facts, not zeros.
func TestArtifactSelectionLogsNoPolicyFields(t *testing.T) {
	logs := captureLogs(t)
	observeRaylineARCSelection(&RequestContext{RequestID: "r"}, &selection.RaylineARCTrace{SelectedArm: 0})
	fields := findLogEvent(t, logs, "rayline_arc_selection")
	for _, field := range []string{"policy_latency_millis", "policy_action_id", "thinking_level", "policy_action_model", "worker_provider_model"} {
		if _, present := fields[field]; present {
			t.Fatalf("an artifact-mode selection logged %s", field)
		}
	}
}
