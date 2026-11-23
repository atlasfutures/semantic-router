package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// bedrockReadinessConfig is policyReadinessConfig with its worker on Amazon
// Bedrock and no static key: the bedrock-short-term-key provider supplies the
// credential.
func bedrockReadinessConfig(mutate func(*config.RouterConfig)) (*config.RouterConfig, *config.Decision) {
	return policyReadinessConfig(func(c *config.RouterConfig) {
		c.VLLMEndpoints[0] = config.VLLMEndpoint{
			Name: "openrouter", Address: "bedrock-runtime.us-east-1.amazonaws.com", Port: 443, Type: "bedrock",
			ProviderProfileName: "openrouter",
		}
		c.ProviderProfiles["openrouter"] = config.ProviderProfile{
			Type: "bedrock", BaseURL: "https://bedrock-runtime.us-east-1.amazonaws.com/anthropic",
		}
		c.Authz.Providers = []config.AuthzProviderConfig{{Type: "bedrock-short-term-key"}}
		if mutate != nil {
			mutate(c)
		}
	})
}

func TestPolicyDispatchReadinessAcceptsABedrockWorkerOnItsOwnIdentity(t *testing.T) {
	if cfg, decision := bedrockReadinessConfig(nil); !raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("a keyless Bedrock worker with the bedrock-short-term-key provider was refused")
	}
	cases := map[string]func(*config.RouterConfig){
		"no bedrock-short-term-key provider": func(c *config.RouterConfig) {
			c.Authz.Providers = []config.AuthzProviderConfig{{Type: "static-config"}}
		},
		"only the Vertex AI minter": func(c *config.RouterConfig) {
			c.Authz.Providers = []config.AuthzProviderConfig{{Type: "google-access-token"}}
		},
		"the default chain (reads caller headers)": func(c *config.RouterConfig) { c.Authz.Providers = nil },
		"header injection beside it": func(c *config.RouterConfig) {
			c.Authz.Providers = append([]config.AuthzProviderConfig{{Type: "header-injection"}}, c.Authz.Providers...)
		},
		"a fail-open chain": func(c *config.RouterConfig) { c.Authz.FailOpen = true },
		"an inline key":     func(c *config.RouterConfig) { c.VLLMEndpoints[0].APIKeyInline = true },
		"a custom auth header": func(c *config.RouterConfig) {
			profile := c.ProviderProfiles["openrouter"]
			profile.AuthHeader = "x-caller-key"
			c.ProviderProfiles["openrouter"] = profile
		},
		"plain http": func(c *config.RouterConfig) {
			profile := c.ProviderProfiles["openrouter"]
			profile.BaseURL = "http://bedrock-runtime.us-east-1.amazonaws.com/anthropic"
			c.ProviderProfiles["openrouter"] = profile
		},
		"an endpoint type that is not its profile's": func(c *config.RouterConfig) { c.VLLMEndpoints[0].Type = "vertex-ai" },
	}
	for name, mutate := range cases {
		if cfg, decision := bedrockReadinessConfig(mutate); raylineARCPolicyDispatchReady(cfg, decision) {
			t.Fatalf("%s: readiness passed", name)
		}
	}
	// A Vertex AI worker is not keyed by the Bedrock minter either.
	if cfg, decision := vertexReadinessConfig(func(c *config.RouterConfig) {
		c.Authz.Providers = []config.AuthzProviderConfig{{Type: "bedrock-short-term-key"}}
	}); raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("a Vertex AI worker passed readiness on the Bedrock minter")
	}
}

// A Bedrock worker's effort override must have a field to go in, as on
// Vertex AI.
func TestPolicyDispatchReadinessRefusesAnUncarriableBedrockEffortOverride(t *testing.T) {
	withOverride := func(apiFormat, baseURL string) (*config.RouterConfig, *config.Decision) {
		cfg, decision := bedrockReadinessConfig(func(c *config.RouterConfig) {
			params := c.ModelConfig["think"]
			params.APIFormat = apiFormat
			c.ModelConfig["think"] = params
			profile := c.ProviderProfiles["openrouter"]
			profile.BaseURL = baseURL
			c.ProviderProfiles["openrouter"] = profile
		})
		decision.Algorithm = &config.AlgorithmConfig{RaylineARC: &config.RaylineARCAlgorithmConfig{
			PolicyService: &config.RaylineARCPolicyServiceConfig{DispatchEffortOverrides: map[string]string{"think": "high"}},
		}}
		return cfg, decision
	}
	if cfg, decision := withOverride(config.APIFormatAnthropic, "https://bedrock-runtime.us-east-1.amazonaws.com/anthropic"); !raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("an effort override on a Bedrock Messages worker was refused")
	}
	if cfg, decision := withOverride(config.APIFormatOpenAI, "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"); raylineARCPolicyDispatchReady(cfg, decision) {
		t.Fatal("an effort override on a Bedrock Chat worker, which has no field for it, passed readiness")
	}
}
