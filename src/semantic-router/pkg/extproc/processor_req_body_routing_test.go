package extproc

import (
	"context"
	"encoding/json"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func TestHandleAutoModelRoutingPreservesSelectedModelHeaderAndRewritesUpstreamModel(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "qwen14b-dev",
			ModelConfig: map[string]config.ModelParams{
				"qwen14b-dev": {
					PreferredEndpoints: []string{"qwen14b-dev_vllm"},
					ExternalModelIDs: map[string]string{
						"vllm": "Qwen/Qwen2.5-14B-Instruct",
					},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:    "qwen14b-dev_vllm",
					Address: "127.0.0.1",
					Port:    8000,
					Type:    "vllm",
					Weight:  1,
				},
			},
		},
	}

	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
	ctx := &RequestContext{
		Headers:      map[string]string{},
		TraceContext: context.Background(),
	}
	openAIRequest := &openai.ChatCompletionNewParams{
		Model: "MoM",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("hello from routed model"),
		},
	}
	baseResponse := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}

	response, err := router.handleAutoModelRouting(
		openAIRequest,
		"MoM",
		"",
		entropy.ReasoningDecision{},
		"qwen14b-dev",
		ctx,
		baseResponse,
	)
	if err != nil {
		t.Fatalf("handleAutoModelRouting returned error: %v", err)
	}

	requestBodyResponse := response.GetRequestBody()
	if requestBodyResponse == nil {
		t.Fatal("expected request body response")
	}
	headerMap := headerValuesByName(requestBodyResponse.Response.HeaderMutation.SetHeaders)
	if got := headerMap[headers.SelectedModel]; got != "qwen14b-dev" {
		t.Fatalf("expected %s header to preserve router alias, got %q", headers.SelectedModel, got)
	}
	if got := headerMap["x-vsr-destination-endpoint"]; got != "" {
		t.Fatalf("router must not emit endpoint destination header, got %q", got)
	}

	var body map[string]any
	if err := json.Unmarshal(requestBodyResponse.Response.BodyMutation.GetBody(), &body); err != nil {
		t.Fatalf("failed to decode mutated body: %v", err)
	}
	if got := body["model"]; got != "Qwen/Qwen2.5-14B-Instruct" {
		t.Fatalf("expected upstream body model rewrite, got %#v", got)
	}
}

func TestHandleAutoModelRoutingAppliesArtifactOwnedARCDispatch(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"arc-worker": {
					PreferredEndpoints: []string{"openrouter"},
					ExternalModelIDs: map[string]string{
						"openai": "configured/model-must-not-win",
					},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:                "openrouter",
					Type:                "openai",
					ProviderProfileName: "openrouter",
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
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
	requestContext := &RequestContext{
		Headers:      map[string]string{},
		TraceContext: context.Background(),
		RaylineARCDispatch: &raylinearc.WorkerManifest{
			ID:                              "arc-worker",
			Model:                           "artifact/provider-model",
			OpenRouterProviderOrder:         []string{"artifact-provider"},
			OpenRouterRequireParameters:     true,
			ThinkingMode:                    "on",
			ReasoningBudgetTokens:           32_768,
			EstimatedInputCostPerToken:      0.000001,
			EstimatedCacheReadCostPerToken:  0.000001,
			EstimatedCacheWriteCostPerToken: 0.000001,
			EstimatedOutputCostPerToken:     0.000001,
			ExtraBody: json.RawMessage(
				`{"reasoning":{"enabled":true,"max_tokens":32768}}`,
			),
		},
	}
	openAIRequest := &openai.ChatCompletionNewParams{
		Model: "MoM",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("hello"),
		},
	}
	response, err := router.handleAutoModelRouting(
		openAIRequest,
		"MoM",
		"arc",
		entropy.ReasoningDecision{},
		"arc-worker",
		requestContext,
		&ext_proc.ProcessingResponse{},
	)
	if err != nil {
		t.Fatal(err)
	}
	requestBodyResponse := response.GetRequestBody()
	if requestBodyResponse == nil {
		t.Fatal("missing request-body response")
	}
	headerMap := headerValuesByName(
		requestBodyResponse.Response.HeaderMutation.SetHeaders,
	)
	if headerMap[headers.SelectedModel] != "arc-worker" {
		t.Fatalf("selected model header = %q", headerMap[headers.SelectedModel])
	}
	var body map[string]any
	if err := json.Unmarshal(
		requestBodyResponse.Response.BodyMutation.GetBody(),
		&body,
	); err != nil {
		t.Fatal(err)
	}
	if body["model"] != "artifact/provider-model" {
		t.Fatalf("body model = %#v", body["model"])
	}
	provider := body["provider"].(map[string]any)
	if provider["allow_fallbacks"] != false ||
		provider["require_parameters"] != true {
		t.Fatalf("provider = %#v", provider)
	}
	if requestContext.VSRReasoningMode != "on" {
		t.Fatalf("reasoning telemetry = %q", requestContext.VSRReasoningMode)
	}
}

func TestHandleAutoModelRoutingFailsClosedOnARCDispatchMutation(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"arc-worker": {
					PreferredEndpoints: []string{"openrouter"},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "openrouter", Address: "127.0.0.1", Port: 8000},
			},
		},
	}
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
	_, requestContext, store, episode := newTestARCEpisodeTransaction(t)
	requestContext.TraceContext = context.Background()
	requestContext.RaylineARCTransaction.markSelection(0, 10)
	requestContext.RaylineARCDispatch = &raylinearc.WorkerManifest{
		ID:        "arc-worker",
		Model:     "artifact/model",
		ExtraBody: json.RawMessage(`[]`),
	}
	response, err := router.handleAutoModelRouting(
		&openai.ChatCompletionNewParams{
			Model: "MoM",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("hello"),
			},
		},
		"MoM",
		"arc",
		entropy.ReasoningDecision{},
		"arc-worker",
		requestContext,
		&ext_proc.ProcessingResponse{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.GetImmediateResponse() == nil ||
		int(response.GetImmediateResponse().GetStatus().GetCode()) != 503 {
		t.Fatalf("response = %#v", response)
	}
	if got := string(response.GetImmediateResponse().Body); got == "" ||
		got == "ARC extra body is invalid" {
		t.Fatalf("response leaked dispatch detail: %q", got)
	}
	assertARCEpisodeNotAdvanced(t, store, episode)
}
