package extproc

import (
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
