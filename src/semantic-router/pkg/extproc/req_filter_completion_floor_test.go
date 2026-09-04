//go:build !windows && cgo

package extproc

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const (
	completionFloorThinkingArm = "arc/model@thinking-on"
	completionFloorPlainArm    = "arc/model@thinking-off"
	completionFloorTokens      = 65536
)

// A thinking arm that answers with a truncated chain of thought is worse than
// an arm that does not answer at all, so the arm declares a completion floor.
// The floor is per model, because the same decision also routes to arms that
// must keep the client budget.
func TestCompletionFloorRaisesThinkingArmBudget(t *testing.T) {
	router := completionFloorTestRouter()
	request := testNeutralRequest("auto", "think about this")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(1024)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = completionFloorDecision(map[string]interface{}{
		completionFloorThinkingArm: completionFloorTokens,
	})

	body := routeAndDecodeBody(t, router, request, completionFloorThinkingArm, ctx)

	if got := body["max_completion_tokens"]; got != float64(completionFloorTokens) {
		t.Fatalf("completion budget = %#v, want the configured floor %d", got, completionFloorTokens)
	}
}

// The old fork read the floor off the worker manifest and applied it with
// max(requested, floor), so a request that names no budget also gets the floor.
func TestCompletionFloorAppliesWhenClientNamesNoBudget(t *testing.T) {
	router := completionFloorTestRouter()
	request := testNeutralRequest("auto", "think about this")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = completionFloorDecision(map[string]interface{}{
		completionFloorThinkingArm: completionFloorTokens,
	})

	body := routeAndDecodeBody(t, router, request, completionFloorThinkingArm, ctx)

	if got := body["max_completion_tokens"]; got != float64(completionFloorTokens) {
		t.Fatalf("completion budget = %#v, want the configured floor %d", got, completionFloorTokens)
	}
}

func TestCompletionFloorLeavesUnlistedArmUntouched(t *testing.T) {
	router := completionFloorTestRouter()
	request := testNeutralRequest("auto", "answer briefly")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(1024)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = completionFloorDecision(map[string]interface{}{
		completionFloorThinkingArm: completionFloorTokens,
	})

	body := routeAndDecodeBody(t, router, request, completionFloorPlainArm, ctx)

	if got := body["max_completion_tokens"]; got != float64(1024) {
		t.Fatalf("completion budget on an unlisted arm = %#v, want the client value 1024", got)
	}
}

func TestCompletionFloorLeavesLargerClientBudgetUntouched(t *testing.T) {
	router := completionFloorTestRouter()
	request := testNeutralRequest("auto", "think about this")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(completionFloorTokens * 2)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = completionFloorDecision(map[string]interface{}{
		completionFloorThinkingArm: completionFloorTokens,
	})

	body := routeAndDecodeBody(t, router, request, completionFloorThinkingArm, ctx)

	if got := body["max_completion_tokens"]; got != float64(completionFloorTokens*2) {
		t.Fatalf("completion budget = %#v, want the larger client value untouched", got)
	}
}

// Default off: a decision that configures no floor must serialize exactly the
// bytes it serialized before the floor existed.
func TestCompletionFloorLeavesUnconfiguredRequestsByteIdentical(t *testing.T) {
	withoutPlugin := completionFloorRoutedBody(t, nil)
	withEmptyFloor := completionFloorRoutedBody(t, map[string]interface{}{})
	withOtherArm := completionFloorRoutedBody(t, map[string]interface{}{
		completionFloorPlainArm: completionFloorTokens,
	})

	if !bytes.Equal(withoutPlugin, withEmptyFloor) {
		t.Fatalf("empty floor changed the body:\n%s\n%s", withoutPlugin, withEmptyFloor)
	}
	if !bytes.Equal(withoutPlugin, withOtherArm) {
		t.Fatalf("floor on another arm changed the body:\n%s\n%s", withoutPlugin, withOtherArm)
	}
}

func completionFloorRoutedBody(t *testing.T, floors map[string]interface{}) []byte {
	t.Helper()
	router := completionFloorTestRouter()
	request := testNeutralRequest("auto", "think about this")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(1024)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	if floors != nil {
		ctx.VSRSelectedDecision = completionFloorDecision(floors)
	}
	return completionFloorRouteBytes(t, router, request, completionFloorThinkingArm, ctx)
}

func completionFloorDecision(floors map[string]interface{}) *config.Decision {
	configuration := map[string]interface{}{}
	if len(floors) > 0 {
		configuration["min_completion_tokens_by_model"] = floors
	}
	return &config.Decision{
		Name: "rayline-arc-dev",
		Plugins: []config.DecisionPlugin{
			{Type: "request_params", Configuration: config.MustStructuredPayload(configuration)},
		},
	}
}

func completionFloorRouteBytes(
	t *testing.T,
	router *OpenAIRouter,
	request *llmprotocol.Request,
	selectedModel string,
	ctx *RequestContext,
) []byte {
	t.Helper()
	response, err := router.handleEntrypointModelRouting(
		request, "auto", "rayline-arc-dev", entropy.ReasoningDecision{}, selectedModel, ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting returned error: %v", err)
	}
	return response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
}

func routeAndDecodeBody(
	t *testing.T,
	router *OpenAIRouter,
	request *llmprotocol.Request,
	selectedModel string,
	ctx *RequestContext,
) map[string]any {
	t.Helper()
	raw := completionFloorRouteBytes(t, router, request, selectedModel, ctx)
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode routed request: %v", err)
	}
	return body
}

func completionFloorTestRouter() *OpenAIRouter {
	arm := config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatOpenAI,
	}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: completionFloorPlainArm,
			ModelConfig: map[string]config.ModelParams{
				completionFloorThinkingArm: arm,
				completionFloorPlainArm:    arm,
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "backend", Address: "127.0.0.1", Port: 8000,
				ProviderProfileName: "provider",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"provider": {Type: "openai", BaseURL: "http://127.0.0.1:8000/v1"},
			},
		},
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
}

// The floor exists so an answer has room beside the thinking, not so the turn
// may think longer. The reasoning bound is derived from what the client
// allowed, and the floor raises the allowance after that number is taken.
//
// Measured on the dev cell 2026-09-04: both ARC configs floor the thinking
// arms at 65,536, so a client asking for 512 dispatched max_completion_tokens
// 65,536, the bound was derived from the floored number, and the turn was
// told it could spend 65,536 tokens reasoning. Three of 17 thinking-arm turns
// ran to the 630 s platform deadline. The bound never bound anything.
func TestCompletionFloorDoesNotRaiseTheReasoningBound(t *testing.T) {
	router := completionFloorTestRouter()
	request := testNeutralRequest("auto", "think about this")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(512)
	request.ReasoningMode = llmprotocol.ReasoningModeAdaptive
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = completionFloorDecision(map[string]interface{}{
		completionFloorThinkingArm: completionFloorTokens,
	})

	body := routeAndDecodeBody(t, router, request, completionFloorThinkingArm, ctx)

	if got := body["max_completion_tokens"]; got != float64(completionFloorTokens) {
		t.Fatalf("completion budget = %#v, want the configured floor %d", got, completionFloorTokens)
	}
	reasoning, present := body["reasoning"].(map[string]any)
	if !present {
		t.Fatalf("a reasoning turn was left unbounded: %#v", body)
	}
	// 512 is below the smallest budget any reasoning API accepts, so the
	// client's allowance is floored to 1024 -- not raised to the arm's floor.
	if got := reasoning["max_tokens"]; got != float64(1024) {
		t.Fatalf("reasoning.max_tokens = %#v, want 1024 from the client's own 512", got)
	}
}
