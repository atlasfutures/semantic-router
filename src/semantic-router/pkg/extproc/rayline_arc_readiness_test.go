/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestRaylineARCDispatchContractsRejectConfigurationDrift(t *testing.T) {
	cfg, workers, decisions := validARCDispatchReadinessFixture()
	if !raylineARCDispatchContractsMatch(cfg, workers, decisions) {
		t.Fatal("valid dispatch contract was rejected")
	}
	tests := []arcDispatchDriftCase{
		{"provider model", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			cfg.ModelConfig["worker"].ExternalModelIDs["openai"] = "other/model"
		}},
		{"provider type", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			cfg.VLLMEndpoints[0].Type = "vllm"
		}},
		{"provider host", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			profile := cfg.ProviderProfiles["openrouter"]
			profile.BaseURL = "https://example.com/api/v1"
			cfg.ProviderProfiles["openrouter"] = profile
		}},
		{"reasoning", func(
			_ *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			decisions []*config.Decision,
		) {
			disabled := false
			decisions[0].ModelRefs[0].UseReasoning = &disabled
		}},
		{"prompt price", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			params := cfg.ModelConfig["worker"]
			params.Pricing.PromptPer1M++
			cfg.ModelConfig["worker"] = params
		}},
		{"cache read price", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			params := cfg.ModelConfig["worker"]
			params.Pricing.CachedInputPer1M++
			cfg.ModelConfig["worker"] = params
		}},
		{"cache write price", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			params := cfg.ModelConfig["worker"]
			params.Pricing.CacheWritePer1M = nil
			cfg.ModelConfig["worker"] = params
		}},
		{"completion price", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			params := cfg.ModelConfig["worker"]
			params.Pricing.CompletionPer1M++
			cfg.ModelConfig["worker"] = params
		}},
		{"currency", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			params := cfg.ModelConfig["worker"]
			params.Pricing.Currency = "EUR"
			cfg.ModelConfig["worker"] = params
		}},
	}
	runARCDispatchDriftCases(t, tests)
}

// TestRaylineARCDispatchContractsRejectCredentialDrift proves the artifact's
// credential identity is enforced: the endpoint key must come from exactly the
// worker's declared environment variable and must be present.
func TestRaylineARCDispatchContractsRejectCredentialDrift(t *testing.T) {
	tests := []arcDispatchDriftCase{
		{"credential env identity", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			cfg.VLLMEndpoints[0].APIKeyEnvName = "OTHER_PROVIDER_KEY"
		}},
		{"credential missing", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			cfg.VLLMEndpoints[0].APIKey = ""
		}},
		{"credential literal not env", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			cfg.VLLMEndpoints[0].APIKeyEnvName = ""
		}},
		{"auth header collision", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			profile := cfg.ProviderProfiles["openrouter"]
			profile.AuthHeader = "content-length"
			cfg.ProviderProfiles["openrouter"] = profile
		}},
		{"auth prefix drift", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			profile := cfg.ProviderProfiles["openrouter"]
			profile.AuthPrefix = "Token"
			cfg.ProviderProfiles["openrouter"] = profile
		}},
		{"custom chat path", func(
			cfg *config.RouterConfig,
			_ []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			profile := cfg.ProviderProfiles["openrouter"]
			profile.ChatPath = "/custom/chat"
			cfg.ProviderProfiles["openrouter"] = profile
		}},
		{"worker credential env unset", func(
			_ *config.RouterConfig,
			workers []raylinearc.WorkerManifest,
			_ []*config.Decision,
		) {
			workers[0].APIKeyEnv = ""
		}},
	}
	runARCDispatchDriftCases(t, tests)
}

func TestRaylineARCDispatchContractsAcceptPinnedOpenAICompatibleURL(t *testing.T) {
	cfg, workers, decisions := validARCDispatchReadinessFixture()
	workers[0].DispatchBackend = raylinearc.DispatchOpenAICompat
	workers[0].ProviderBaseURL = "https://worker.example/v1"
	profile := cfg.ProviderProfiles["openrouter"]
	profile.BaseURL = workers[0].ProviderBaseURL
	cfg.ProviderProfiles["openrouter"] = profile
	if !raylineARCDispatchContractsMatch(cfg, workers, decisions) {
		t.Fatal("pinned openai-compatible dispatch contract was rejected")
	}

	profile.BaseURL = "https://other-worker.example/v1"
	cfg.ProviderProfiles["openrouter"] = profile
	if raylineARCDispatchContractsMatch(cfg, workers, decisions) {
		t.Fatal("openai-compatible base URL drift was accepted")
	}
}

type arcDispatchDriftCase struct {
	name   string
	mutate func(*config.RouterConfig, []raylinearc.WorkerManifest, []*config.Decision)
}

func runARCDispatchDriftCases(t *testing.T, tests []arcDispatchDriftCase) {
	t.Helper()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg, workers, decisions := validARCDispatchReadinessFixture()
			test.mutate(cfg, workers, decisions)
			if raylineARCDispatchContractsMatch(cfg, workers, decisions) {
				t.Fatal("configuration drift was accepted")
			}
		})
	}
}

func validARCDispatchReadinessFixture() (
	*config.RouterConfig,
	[]raylinearc.WorkerManifest,
	[]*config.Decision,
) {
	cacheWrite := 3.0
	enabled := true
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"worker": {
					PreferredEndpoints: []string{"openrouter-endpoint"},
					ExternalModelIDs: map[string]string{
						"openai": "provider/model",
					},
					APIFormat: config.APIFormatOpenAI,
					Pricing: config.ModelPricing{
						Currency:         "USD",
						PromptPer1M:      1,
						CachedInputPer1M: 2,
						CacheWritePer1M:  &cacheWrite,
						CompletionPer1M:  4,
					},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:                "openrouter-endpoint",
					Type:                "openai",
					ProviderProfileName: "openrouter",
					APIKey:              "synthetic-provider-credential",
					APIKeyEnvName:       "ARC_TEST_PROVIDER_KEY",
				},
			},
			ProviderProfiles: map[string]config.ProviderProfile{
				"openrouter": {
					Type:    "openai",
					BaseURL: "https://openrouter.ai/api/v1",
				},
			},
		},
	}
	workers := []raylinearc.WorkerManifest{
		{
			ID:                              "worker",
			Model:                           "provider/model",
			ThinkingMode:                    "on",
			APIKeyEnv:                       "ARC_TEST_PROVIDER_KEY",
			EstimatedInputCostPerToken:      0.000001,
			EstimatedCacheReadCostPerToken:  0.000002,
			EstimatedCacheWriteCostPerToken: 0.000003,
			EstimatedOutputCostPerToken:     0.000004,
		},
	}
	decision := &config.Decision{
		ModelRefs: []config.ModelRef{
			{
				Model: "worker",
				ModelReasoningControl: config.ModelReasoningControl{
					UseReasoning: &enabled,
				},
			},
		},
	}
	return cfg, workers, []*config.Decision{decision}
}
