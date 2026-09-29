package extproc

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A format the policy service cannot read is refused before the episode lease,
// so the episode is untouched and nothing waits on a service call.
func TestPolicyModeRefusesAnUnreadableFormatBeforeTheLease(t *testing.T) {
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{RaylineARCEpisodeStore: store}
	requestContext := &RequestContext{
		Headers:      map[string]string{"x-rayline-session": t.Name()},
		SourceFormat: llmprotocol.OpenAIResponsesV1,
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
