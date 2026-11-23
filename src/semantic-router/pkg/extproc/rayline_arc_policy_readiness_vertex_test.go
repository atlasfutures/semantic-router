package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// vertexReadinessConfig is policyReadinessConfig with its worker on Vertex AI
// and no static key: the google-access-token provider supplies the credential.
func vertexReadinessConfig(mutate func(*config.RouterConfig)) (*config.RouterConfig, *config.Decision) {
	return policyReadinessConfig(func(c *config.RouterConfig) {
		c.VLLMEndpoints[0] = config.VLLMEndpoint{
			Name: "openrouter", Address: "aiplatform.googleapis.com", Port: 443, Type: "vertex-ai",
			ProviderProfileName: "openrouter",
		}
		c.ProviderProfiles["openrouter"] = config.ProviderProfile{
			Type: "vertex-ai", BaseURL: "https://aiplatform.googleapis.com/v1/projects/p/locations/global",
		}
		c.Authz.Providers = []config.AuthzProviderConfig{{Type: "google-access-token"}}
		if mutate != nil {
			mutate(c)
		}
	})
}

func TestPolicyDispatchReadinessAcceptsAVertexWorkerOnItsOwnIdentity(t *testing.T) {
	if cfg, decision := vertexReadinessConfig(nil); !raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("a keyless Vertex AI worker with the google-access-token provider was refused")
	}
	cases := map[string]func(*config.RouterConfig){
		"no google-access-token provider": func(c *config.RouterConfig) {
			c.Authz.Providers = []config.AuthzProviderConfig{{Type: "static-config"}}
		},
		"the default chain (reads caller headers)": func(c *config.RouterConfig) { c.Authz.Providers = nil },
		"header injection beside it": func(c *config.RouterConfig) {
			c.Authz.Providers = append([]config.AuthzProviderConfig{{Type: "header-injection"}}, c.Authz.Providers...)
		},
		"a fail-open chain": func(c *config.RouterConfig) { c.Authz.FailOpen = true },
		"an unknown provider type in the chain": func(c *config.RouterConfig) {
			c.Authz.Providers = append(c.Authz.Providers, config.AuthzProviderConfig{Type: "nope"})
		},
		"an inline key": func(c *config.RouterConfig) { c.VLLMEndpoints[0].APIKeyInline = true },
		"a custom auth header": func(c *config.RouterConfig) {
			profile := c.ProviderProfiles["openrouter"]
			profile.AuthHeader = "x-caller-key"
			c.ProviderProfiles["openrouter"] = profile
		},
		"plain http": func(c *config.RouterConfig) {
			profile := c.ProviderProfiles["openrouter"]
			profile.BaseURL = "http://aiplatform.googleapis.com/v1/projects/p/locations/global"
			c.ProviderProfiles["openrouter"] = profile
		},
		"another provider without a key": func(c *config.RouterConfig) {
			c.VLLMEndpoints[0].Type = "openai"
			c.ProviderProfiles["openrouter"] = config.ProviderProfile{Type: "openai", BaseURL: "https://api.example.com/v1"}
		},
	}
	for name, mutate := range cases {
		if cfg, decision := vertexReadinessConfig(mutate); raylineARCPolicyDispatchReady(cfg, decision) {
			t.Fatalf("%s: readiness passed", name)
		}
	}
}

// A disabled worker is never sent a turn, so an install that does not serve
// it (and so carries no key for it) stays ready; a disabled worker that an
// in-service arm is dispatched on is still checked.
func TestPolicyDispatchReadinessSkipsOutOfServiceWorkers(t *testing.T) {
	disabled := true
	withOff := func(overrides map[string]string, offArm bool) (*config.RouterConfig, *config.Decision) {
		cfg, decision := vertexReadinessConfig(func(c *config.RouterConfig) {
			write := 1.25
			c.ModelConfig["off"] = config.ModelParams{
				PreferredEndpoints: []string{"keyless-openrouter"},
				Pricing:            config.ModelPricing{Currency: "USD", PromptPer1M: 1, CompletionPer1M: 5, CacheWritePer1M: &write},
				Disabled:           &disabled,
			}
			if offArm {
				think := c.ModelConfig["think"]
				think.Disabled = &disabled
				c.ModelConfig["think"] = think
			}
			c.VLLMEndpoints = append(c.VLLMEndpoints, config.VLLMEndpoint{
				Name: "keyless-openrouter", Address: "openrouter.ai", Port: 443, Type: "openai",
				APIKeyEnvName: "OPENROUTER_API_KEY", ProviderProfileName: "plain-openrouter",
			})
			c.ProviderProfiles["plain-openrouter"] = config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"}
		})
		decision.ModelRefs = append(decision.ModelRefs, config.ModelRef{Model: "off"})
		decision.Algorithm = &config.AlgorithmConfig{RaylineARC: &config.RaylineARCAlgorithmConfig{
			PolicyService: &config.RaylineARCPolicyServiceConfig{DispatchModelOverrides: overrides},
		}}
		return cfg, decision
	}
	if cfg, decision := withOff(nil, false); !raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("a disabled worker without a key held readiness back")
	}
	if cfg, decision := withOff(map[string]string{"think": "off"}, false); raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("a disabled worker that an in-service arm is sent on skipped readiness")
	}
	if cfg, decision := withOff(map[string]string{"think": "off"}, true); !raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("a disabled worker sent on only by a disabled arm held readiness back")
	}
	if cfg, _ := withOff(nil, false); raylineARCPolicyWorkerReady(cfg, "off") {
		t.Fatal("the keyless worker itself is not ready; only its being out of service skips it")
	}
}
