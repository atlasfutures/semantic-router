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
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestOpenRouterCanaryConfigMatchesThreeArmDispatchContract(t *testing.T) {
	t.Setenv("SYNTHETIC_API_KEY", "test-openrouter-key")
	t.Setenv("RAYLINE_ARC_ENCODER_BASE_URL", "https://encoder.example")
	t.Setenv("RAYLINE_ARC_ENCODER_BUILD_ID", "test-build")
	t.Setenv("RAYLINE_ARC_REDIS_PASSWORD", "test-redis-password")

	path := filepath.Join(
		"..", "..", "..", "..", "deploy", "compose", "rayline-arc",
		"config-openrouter.yaml",
	)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read OpenRouter canary config: %v", err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse OpenRouter canary config: %v", err)
	}

	workers := []raylinearc.WorkerManifest{
		openRouterCanaryWorker(
			"worker-a", "deepseek/deepseek-v4-flash",
			0.00000014, 0.000000028, 0.00000014, 0.00000028,
		),
		openRouterCanaryWorker(
			"worker-b", "openai/gpt-5.6-luna",
			0.0000001, 0.00000001, 0.000000125, 0.0000006,
		),
		openRouterCanaryWorker(
			"worker-c", "z-ai/glm-5.2",
			0.0000014, 0.00000014, 0.0000014, 0.0000044,
		),
	}
	decisions := configuredRaylineARCDecisions(cfg)
	if !raylineARCDispatchContractsMatch(cfg, workers, decisions) {
		t.Fatal("OpenRouter canary config drifted from its artifact dispatch contract")
	}
}

func openRouterCanaryWorker(
	id string,
	model string,
	inputCost float64,
	cacheReadCost float64,
	cacheWriteCost float64,
	outputCost float64,
) raylinearc.WorkerManifest {
	return raylinearc.WorkerManifest{
		ID:                              id,
		Model:                           model,
		APIKeyEnv:                       "SYNTHETIC_API_KEY",
		ThinkingMode:                    "off",
		EstimatedInputCostPerToken:      inputCost,
		EstimatedCacheReadCostPerToken:  cacheReadCost,
		EstimatedCacheWriteCostPerToken: cacheWriteCost,
		EstimatedOutputCostPerToken:     outputCost,
	}
}
