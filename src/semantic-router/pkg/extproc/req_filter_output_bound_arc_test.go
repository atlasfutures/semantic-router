package extproc

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkingcontrol"
)

// A v5 control renders Messages thinking after encoding, so the output bound
// plans the planned control's budget, and the control raise that follows has
// nothing left to add. Under a cap the total never exceeds it: the control's
// budget is lowered to fit, and a cap that cannot hold the 1024 minimum fails
// the turn rather than change the control's base or exceed the cap. A limit
// the caller stated is left to the existing raise.
func TestDispatchOutputBoundPlansTheV5ControlBudget(t *testing.T) {
	for name, test := range map[string]struct {
		limit      int
		stated     *int64
		wantTokens int64
		wantBudget int64
		wantErr    bool
	}{
		"cap 5000":           {limit: 5000, wantTokens: 5000, wantBudget: 3000},
		"no cap":             {wantTokens: 6000, wantBudget: 4000},
		"fits under the cap": {limit: 10000, wantTokens: 6000, wantBudget: 4000},
		"cap below 1025":     {limit: 1000, wantErr: true},
		"a caller's limit":   {stated: llmprotocol.Int64(1000), wantTokens: 5000, wantBudget: 4000},
	} {
		t.Run(name, func(t *testing.T) {
			router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
			params := router.Config.ModelConfig[model]
			params.MaxOutputTokens = 2000
			router.Config.ModelConfig[model] = params
			budget := int64(4000)
			planned := &plannedThinkingControl{control: thinkingcontrol.Control{Native: "enabled", BudgetTokens: &budget}}
			ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, nil)
			ctx.RaylineARCThinkingControl = planned
			if test.limit > 0 {
				ctx.VSRSelectedDecision = maxTokensLimitDecision(t, test.limit)
			}
			request := testNeutralRequest(model, "hi")
			request.Sampling.MaxOutputTokens = test.stated
			ctx.SemanticRequest = request
			_, err := router.prepareProviderDispatch(request, model, "", false, ctx)
			if test.wantErr {
				if !errors.Is(err, errThinkingControlOverCap) {
					t.Fatalf("dispatch = %v, want the over-cap refusal", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			if got := request.Sampling.MaxOutputTokens; got == nil || *got != test.wantTokens {
				t.Fatalf("max_tokens = %v, want %d", got, test.wantTokens)
			}
			if got := planned.control.BudgetTokens; got == nil || *got != test.wantBudget {
				t.Fatalf("control budget = %v, want %d", got, test.wantBudget)
			}
			if budget != 4000 {
				t.Fatal("the action's own budget was changed; only the planned copy may be")
			}
		})
	}
}

// Compression reserves the output bound dispatch will send, including a
// policy action's reasoning_max_tokens, which dispatch copies into the
// request only after compression has run.
func TestCompressionReserveMatchesDispatchWithAPolicyAction(t *testing.T) {
	budget := int64(12000)
	for name, effort := range map[string]*string{
		"budget action": nil,
		"effort action": policyTestEffort("high"),
		"off action":    policyTestEffort("none"),
	} {
		t.Run(name, func(t *testing.T) {
			router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
			params := router.Config.ModelConfig[model]
			params.MaxOutputTokens = 4000
			router.Config.ModelConfig[model] = params
			actionBudget := &budget
			if effort != nil {
				actionBudget = nil
			}
			action := policyAction("think", "none", model, effort, actionBudget, "")
			decision := policyDecisionWithActions(action)
			ctx := policyDispatchContext(decision, action)
			ctx.SourceFormat, ctx.VSRSelectedModel = llmprotocol.OpenAIResponsesV1, model
			ctx.RequestID, ctx.TraceContext = "reserve-policy-action", context.Background()
			request := testNeutralRequest(model, "hi")
			ctx.SemanticRequest = request

			reserve := router.compressionOutputReserve(model, ctx, request)
			if _, err := router.prepareProviderDispatch(request, model, decision.Name, true, ctx); err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			sent := request.Sampling.MaxOutputTokens
			if sent == nil || *sent != reserve {
				t.Fatalf("compression reserved %d, dispatch sent %v", reserve, sent)
			}
			if effort == nil && reserve != 4000+budget {
				t.Fatalf("reserve = %d, want the card bound on top of the action's budget (%d)", reserve, 4000+budget)
			}
		})
	}
}
