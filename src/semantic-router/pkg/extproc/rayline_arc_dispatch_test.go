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
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestApplyRaylineARCDispatchOwnsProviderAndThinkingContract(t *testing.T) {
	maxCompletion := uint64(70_000)
	worker := &raylinearc.WorkerManifest{
		Model:                       "provider/private-thinking-model",
		OpenRouterProviderOrder:     []string{"pinned-provider"},
		OpenRouterAllowFallbacks:    false,
		OpenRouterRequireParameters: true,
		ThinkingMode:                "on",
		ReasoningBudgetTokens:       32_768,
		MinimumCompletionTokens:     65_536,
		MaxCompletionTokens:         &maxCompletion,
		ExtraBody: json.RawMessage(
			`{"reasoning":{"enabled":true,"max_tokens":32768}}`,
		),
	}
	body, err := applyRaylineARCDispatch(
		[]byte(`{
			"model":"client-model",
			"messages":[{"role":"user","content":"private"}],
			"provider":{"order":["client-provider"],"allow_fallbacks":true},
			"reasoning":{"enabled":false},
			"reasoning_effort":"low",
			"chat_template_kwargs":{"enable_thinking":false},
			"thinking":{"type":"adaptive"},
			"context_management":{"edits":[]},
			"diagnostics":{"enabled":true},
			"output_config":{"effort":"low"},
			"max_completion_tokens":1024,
			"temperature":1.7,
			"tools":[{
				"name":"run",
				"defer_loading":true,
				"allowed_callers":["code_execution"],
				"eager_input_streaming":true,
				"custom":"kept"
			}]
		}`),
		worker,
	)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]any
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatal(err)
	}
	assertARCProviderBody(t, result, worker)
	assertARCThinkingBody(t, result)
	assertARCClientFieldsRemoved(t, result)
	assertARCToolMaterialized(t, result)
}

func assertARCProviderBody(
	t *testing.T,
	result map[string]any,
	worker *raylinearc.WorkerManifest,
) {
	t.Helper()
	if result["model"] != worker.Model {
		t.Fatalf("model = %#v", result["model"])
	}
	provider := result["provider"].(map[string]any)
	order := provider["order"].([]any)
	if len(order) != 1 || order[0] != "pinned-provider" ||
		provider["allow_fallbacks"] != false ||
		provider["require_parameters"] != true {
		t.Fatalf("provider = %#v", provider)
	}
}

func assertARCThinkingBody(t *testing.T, result map[string]any) {
	t.Helper()
	reasoning := result["reasoning"].(map[string]any)
	if reasoning["enabled"] != true ||
		reasoning["max_tokens"] != float64(32_768) {
		t.Fatalf("reasoning = %#v", reasoning)
	}
	if result["max_tokens"] != float64(65_536) {
		t.Fatalf("max_tokens = %#v", result["max_tokens"])
	}
}

func assertARCClientFieldsRemoved(t *testing.T, result map[string]any) {
	t.Helper()
	for _, absent := range []string{
		"max_completion_tokens",
		"temperature",
		"reasoning_effort",
		"chat_template_kwargs",
		"thinking",
		"context_management",
		"diagnostics",
		"output_config",
	} {
		if _, exists := result[absent]; exists {
			t.Fatalf("client-owned field %q survived", absent)
		}
	}
}

func assertARCToolMaterialized(t *testing.T, result map[string]any) {
	t.Helper()
	tool := result["tools"].([]any)[0].(map[string]any)
	if tool["custom"] != "kept" || len(tool) != 2 {
		t.Fatalf("tool = %#v", tool)
	}
}

func TestApplyRaylineARCDispatchUsesSelectedArmLimits(t *testing.T) {
	maxCompletion := uint64(70_000)
	temperature := 0.2
	worker := &raylinearc.WorkerManifest{
		Model:                       "provider/private-no-thinking-model",
		OpenRouterProviderOrder:     []string{"second-provider"},
		OpenRouterRequireParameters: true,
		ThinkingMode:                "off",
		MaxCompletionTokens:         &maxCompletion,
		Temperature:                 &temperature,
		ExtraBody: json.RawMessage(
			`{
				"reasoning":{"enabled":false,"effort":"none"},
				"reasoning_effort":"none",
				"max_tokens":68000
			}`,
		),
	}
	body, err := applyRaylineARCDispatch(
		[]byte(`{"model":"client","messages":[],"max_tokens":90000}`),
		worker,
	)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]any
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatal(err)
	}
	if result["model"] != worker.Model ||
		result["max_tokens"] != float64(68_000) ||
		result["temperature"] != 0.2 {
		t.Fatalf("selected arm contract was not applied: %#v", result)
	}
	reasoning := result["reasoning"].(map[string]any)
	if reasoning["enabled"] != false {
		t.Fatalf("reasoning = %#v", reasoning)
	}
}

func TestApplyRaylineARCDispatchRejectsMalformedInput(t *testing.T) {
	worker := &raylinearc.WorkerManifest{
		Model:                       "provider/model",
		OpenRouterProviderOrder:     []string{"provider"},
		OpenRouterRequireParameters: true,
		ExtraBody: json.RawMessage(
			`{"reasoning":{"enabled":false}}`,
		),
	}
	for _, body := range [][]byte{
		[]byte(`[]`),
		[]byte(`{"max_tokens":"many"}`),
		[]byte(`{"max_tokens":-1}`),
		[]byte(`{"max_tokens":1} {}`),
	} {
		if _, err := applyRaylineARCDispatch(body, worker); err == nil {
			t.Fatalf("malformed body was accepted: %s", body)
		}
	}
	worker.ExtraBody = json.RawMessage(`[]`)
	if _, err := applyRaylineARCDispatch([]byte(`{}`), worker); err == nil {
		t.Fatal("malformed extra body was accepted")
	}
}
