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
	"slices"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

// TestRaylineARCCredentialIgnoresCallerSuppliedKey proves an armed ARC
// dispatch injects the artifact-owned credential and never a caller header.
func TestRaylineARCCredentialIgnoresCallerSuppliedKey(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"worker": {PreferredEndpoints: []string{"openrouter-endpoint"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:          "openrouter-endpoint",
					Type:          "openai",
					Model:         "worker",
					APIKey:        "artifact-owned-key",
					APIKeyEnvName: "ARC_TEST_PROVIDER_KEY",
				},
			},
		},
	}
	router := &OpenAIRouter{Config: cfg}
	worker := raylinearc.WorkerManifest{
		ID:        "worker",
		Model:     "provider/model",
		APIKeyEnv: "ARC_TEST_PROVIDER_KEY",
	}
	ctx := &RequestContext{
		Headers:            map[string]string{"x-user-openai-key": "attacker-key"},
		RaylineARCDispatch: &worker,
	}
	state := &routeHeaderState{}

	if response := router.appendRaylineARCCredentialHeader(
		state,
		"worker",
		"Authorization",
		"Bearer",
		ctx,
	); response != nil {
		t.Fatalf("artifact credential injection failed: %#v", response)
	}

	if len(state.setHeaders) != 1 {
		t.Fatalf("expected one credential header, got %d", len(state.setHeaders))
	}
	value := string(state.setHeaders[0].GetHeader().GetRawValue())
	if value != "Bearer artifact-owned-key" {
		t.Fatalf("injected credential = %q", value)
	}
}

// TestRaylineARCCredentialFailsClosedWhenMissing proves a missing artifact
// credential returns the ARC 503 rather than forwarding the caller's header.
func TestRaylineARCCredentialFailsClosedWhenMissing(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"worker": {PreferredEndpoints: []string{"openrouter-endpoint"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:          "openrouter-endpoint",
					Type:          "openai",
					Model:         "worker",
					APIKeyEnvName: "OTHER_KEY",
				},
			},
		},
	}
	router := &OpenAIRouter{Config: cfg}
	worker := raylinearc.WorkerManifest{ID: "worker", APIKeyEnv: "ARC_TEST_PROVIDER_KEY"}
	ctx := &RequestContext{
		Headers:            map[string]string{"x-user-openai-key": "attacker-key"},
		RaylineARCDispatch: &worker,
	}

	response := router.appendRaylineARCCredentialHeader(
		&routeHeaderState{},
		"worker",
		"Authorization",
		"Bearer",
		ctx,
	)
	if response == nil ||
		int(response.GetImmediateResponse().GetStatus().GetCode()) != 503 {
		t.Fatalf("expected fail-closed 503, got %#v", response)
	}
}

// TestRaylineARCCredentialSurvivesLaterHeaderMutations proves profile extra
// headers and decision mutations cannot add a second Authorization value or
// delete the artifact-owned one. Envoy's default append action would
// otherwise combine them into a multi-valued header.
func TestRaylineARCCredentialSurvivesLaterHeaderMutations(t *testing.T) {
	ctx := &RequestContext{
		Headers:              map[string]string{},
		RaylineARCDispatch:   &raylinearc.WorkerManifest{ID: "worker"},
		RaylineARCAuthHeader: "Authorization",
	}
	state := &routeHeaderState{
		setHeaders: []*core.HeaderValueOption{
			{
				Header: &core.HeaderValue{
					Key:      "Authorization",
					RawValue: []byte("Bearer artifact-owned-key"),
				},
				AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
			},
			{Header: &core.HeaderValue{Key: "x-keep", RawValue: []byte("yes")}},
			{
				Header: &core.HeaderValue{
					Key:      "authorization",
					RawValue: []byte("Bearer profile-extra-header"),
				},
			},
			{
				Header: &core.HeaderValue{
					Key:      "Authorization",
					RawValue: []byte("Bearer decision-mutation"),
				},
			},
		},
		removeHeaders: []string{"authorization", "x-drop"},
	}

	enforceRaylineARCCredentialHeader(state, ctx)

	authValues := []string{}
	for _, option := range state.setHeaders {
		if strings.EqualFold(option.GetHeader().GetKey(), "Authorization") {
			authValues = append(authValues, string(option.GetHeader().GetRawValue()))
		}
	}
	if len(authValues) != 1 || authValues[0] != "Bearer artifact-owned-key" {
		t.Fatalf("auth header values = %#v", authValues)
	}
	if len(state.setHeaders) != 2 {
		t.Fatalf("unrelated headers were dropped: %#v", state.setHeaders)
	}
	if slices.Contains(state.removeHeaders, "authorization") {
		t.Fatalf("artifact credential could be deleted: %#v", state.removeHeaders)
	}
	if !slices.Contains(state.removeHeaders, "x-drop") {
		t.Fatalf("unrelated deletion was dropped: %#v", state.removeHeaders)
	}
}

// TestToolMutationCannotEraseARCDispatch proves a later tool-selection body
// rewrite cannot revert the artifact-owned model, provider policy, or limits.
func TestToolMutationCannotEraseARCDispatch(t *testing.T) {
	worker := &raylinearc.WorkerManifest{
		Model:                       "artifact/provider-model",
		OpenRouterProviderOrder:     []string{"pinned-provider"},
		OpenRouterRequireParameters: true,
		ThinkingMode:                "off",
	}
	ctx := &RequestContext{Headers: map[string]string{}, RaylineARCDispatch: worker}
	// A tool mutation reserializes the client request, losing ARC shaping.
	clientBody := []byte(`{"model":"MoM","messages":[],"provider":{"order":["attacker"]}}`)
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{Body: clientBody},
					},
				},
			},
		},
	}

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	router.reapplyRaylineARCDispatch(response, ctx)

	final := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var object map[string]any
	if err := json.Unmarshal(final, &object); err != nil {
		t.Fatal(err)
	}
	if object["model"] != "artifact/provider-model" {
		t.Fatalf("artifact model was lost: %#v", object["model"])
	}
	provider, ok := object["provider"].(map[string]any)
	if !ok {
		t.Fatalf("artifact provider policy was lost: %#v", object["provider"])
	}
	order, ok := provider["order"].([]any)
	if !ok || len(order) != 1 || order[0] != "pinned-provider" {
		t.Fatalf("client provider order survived: %#v", provider["order"])
	}
	if provider["require_parameters"] != true {
		t.Fatalf("artifact parameter policy was lost: %#v", provider)
	}
}

// TestReapplyFailsClosedOnUnshapeableBody proves a body the artifact contract
// cannot shape is cleared rather than forwarded unshaped.
func TestReapplyFailsClosedOnUnshapeableBody(t *testing.T) {
	ctx := &RequestContext{
		Headers:            map[string]string{},
		RaylineARCDispatch: &raylinearc.WorkerManifest{Model: "artifact/model"},
	}
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{
							Body: []byte("not json"),
						},
					},
				},
			},
		},
	}

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	router.reapplyRaylineARCDispatch(response, ctx)

	if body := response.GetRequestBody().GetResponse().
		GetBodyMutation().GetBody(); len(body) != 0 {
		t.Fatalf("unshapeable body was forwarded: %q", body)
	}
}
