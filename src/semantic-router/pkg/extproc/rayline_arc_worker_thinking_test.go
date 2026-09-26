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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// derivedThinkingBody is what a claude-code turn looks like on the Chat seam
// before a base applies: the bound derived from max_tokens, and no effort.
const derivedThinkingBody = `{"model":"m","max_completion_tokens":32000,"reasoning":{"max_tokens":32000},"reasoning_effort":"high","messages":[]}`

func workerThinkingFixture(base *config.RaylineARCWorkerThinkingConfig, useReasoning bool, baseURL string) (*providerDispatch, *RequestContext) {
	arc := &config.RaylineARCAlgorithmConfig{}
	if base != nil {
		arc.WorkerThinking = map[string]config.RaylineARCWorkerThinkingConfig{"worker@on": *base}
	}
	dispatch := &providerDispatch{
		logicalModel: "worker@on", targetFormat: llmprotocol.OpenAIChatV1, useReasoning: useReasoning,
		profile: &config.ProviderProfile{Type: "openai", BaseURL: baseURL},
	}
	ctx := &RequestContext{
		VSRSelectedDecision: &config.Decision{Algorithm: &config.AlgorithmConfig{RaylineARC: arc}},
		RaylineARCDispatch:  &raylinearc.WorkerManifest{ID: "worker@on"},
	}
	return dispatch, ctx
}

func reasoningControls(t *testing.T, body []byte) (json.RawMessage, bool) {
	t.Helper()
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	_, effort := wire["reasoning_effort"]
	return wire["reasoning"], effort
}

func TestWorkerThinkingBaseReplacesTheDerivedBound(t *testing.T) {
	const openRouter = "https://openrouter.ai/api/v1"
	cases := map[string]struct {
		base config.RaylineARCWorkerThinkingConfig
		want string
	}{
		"named effort reaches the model": {
			config.RaylineARCWorkerThinkingConfig{Level: "high", Wire: "effort", Effort: "high"}, `{"effort":"high"}`},
		"simulated low is a budget alone": {
			config.RaylineARCWorkerThinkingConfig{Level: "cap_4096", Wire: "budget", MaxTokens: 4096}, `{"max_tokens":4096}`},
		"provider default is uncapped": {
			config.RaylineARCWorkerThinkingConfig{Level: "default", Wire: "provider_default"}, ``},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			base := test.base
			dispatch, ctx := workerThinkingFixture(&base, true, openRouter)
			body, err := applyRaylineARCWorkerThinking([]byte(derivedThinkingBody), dispatch, ctx)
			if err != nil {
				t.Fatal(err)
			}
			reasoning, effort := reasoningControls(t, body)
			if string(reasoning) != test.want || effort {
				t.Fatalf("reasoning = %s, top-level effort present = %v", reasoning, effort)
			}
			record := map[string]interface{}{}
			appendRaylineARCThinkingFields(record, ctx)
			if record["thinking_base_level"] != base.Level || record["thinking_base_wire"] != base.Wire {
				t.Fatalf("routing record = %v", record)
			}
		})
	}
}

func TestWorkerThinkingBaseAppliesOnlyWhereItCan(t *testing.T) {
	base := config.RaylineARCWorkerThinkingConfig{Level: "high", Wire: "effort", Effort: "high"}
	cases := map[string]func() (*providerDispatch, *RequestContext){
		"no base": func() (*providerDispatch, *RequestContext) {
			return workerThinkingFixture(nil, true, "https://openrouter.ai/api/v1")
		},
		"thinking-off turn": func() (*providerDispatch, *RequestContext) {
			return workerThinkingFixture(&base, false, "https://openrouter.ai/api/v1")
		},
		"not OpenRouter": func() (*providerDispatch, *RequestContext) {
			return workerThinkingFixture(&base, true, "http://vllm:8000/v1")
		},
	}
	for name, fixture := range cases {
		t.Run(name, func(t *testing.T) {
			dispatch, ctx := fixture()
			body, err := applyRaylineARCWorkerThinking([]byte(derivedThinkingBody), dispatch, ctx)
			if err != nil || string(body) != derivedThinkingBody || ctx.RaylineARCWorkerThinking != nil {
				t.Fatalf("body changed: %s (err %v)", body, err)
			}
		})
	}
}

// Through the real provider boundary: the arm's configured effort is dropped
// for the bound derived from the client's allowance, and the worker's base
// then puts the registry's wire back. The routing record reads the body after
// both, so it names the effort that travelled.
func TestWorkerThinkingBaseWinsAtTheProviderBoundary(t *testing.T) {
	router := newArmReasoningRouter()
	decision := router.Config.GetDecisionByName("arc")
	decision.Algorithm = &config.AlgorithmConfig{RaylineARC: &config.RaylineARCAlgorithmConfig{
		WorkerThinking: map[string]config.RaylineARCWorkerThinkingConfig{
			"gpt-5-mini": {Level: "high", Wire: config.RaylineARCWorkerThinkingEffort, Effort: "high"},
		},
	}}
	dispatch := &providerDispatch{
		logicalModel: "gpt-5-mini", decisionName: "arc", useReasoning: true,
		targetFormat: llmprotocol.OpenAIChatV1, profile: openRouterProviderProfile(),
	}
	ctx := &RequestContext{
		VSRSelectedDecision: decision,
		RaylineARCDispatch:  &raylinearc.WorkerManifest{ID: "gpt-5-mini"},
	}
	body, err := router.adaptProviderRequest(
		[]byte(`{"model":"gpt-5-mini","max_completion_tokens":32000,"messages":[{"role":"user","content":"hi"}]}`),
		dispatch, ctx,
	)
	if err != nil {
		t.Fatal(err)
	}
	reasoning, effort := reasoningControls(t, body)
	if string(reasoning) != `{"effort":"high"}` || effort {
		t.Fatalf("reasoning = %s, top-level effort present = %v", reasoning, effort)
	}
	if ctx.DispatchedReasoningBound != nil || ctx.DispatchedReasoningEffort != "high" {
		t.Fatalf("routing record reads effort %q bound %v", ctx.DispatchedReasoningEffort, ctx.DispatchedReasoningBound)
	}
}
