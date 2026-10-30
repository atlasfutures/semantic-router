package extproc

import (
	"errors"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// armCeilingDecision is a rayline_arc decision, under which the router chose
// the arm, with the given request_params max_tokens_limit when one is set.
func armCeilingDecision(t *testing.T, limit int) *config.Decision {
	t.Helper()
	decision := &config.Decision{
		Name: "arc",
		Algorithm: &config.AlgorithmConfig{
			Type: config.RaylineARCAlgorithmType, RaylineARC: &config.RaylineARCAlgorithmConfig{},
		},
	}
	if limit > 0 {
		decision.Plugins = maxTokensLimitDecision(t, limit).Plugins
	}
	return decision
}

// plainDecision routes by an ordinary algorithm, where the caller's own
// modelRef picks the model, so the "a stated limit is never touched"
// invariant holds as it always has.
func plainDecision() *config.Decision {
	return &config.Decision{Name: "plain", Algorithm: &config.AlgorithmConfig{Type: "static"}}
}

// arcTypeWithoutItsBlock is the predicate's other side: the rayline_arc type
// with no rayline_arc block, which the loader rejects but which must not be
// read as "the router chose the arm" if it ever reaches dispatch.
func arcTypeWithoutItsBlock() *config.Decision {
	return &config.Decision{Name: "arc-typed", Algorithm: &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType}}
}

const armCeilingBody = `{"model":"m","input":"write the long tool call","max_output_tokens":100000}`

// On a rayline_arc decision the caller never named the arm, so a stated limit
// above the chosen arm's card max_output_tokens is lowered to it rather than
// forwarded for the provider to refuse. A limit already at or under the
// ceiling is the caller's own and stands, and an arm whose card declares no
// ceiling clamps nothing.
func TestARCArmCeilingLowersAStatedLimitToTheArm(t *testing.T) {
	for name, test := range map[string]struct {
		card int
		body string
		want string
	}{
		"above the ceiling is lowered": {16384, armCeilingBody, "max_output_tokens=16384"},
		"below the ceiling stands": {
			16384, `{"model":"m","input":"write the long tool call","max_output_tokens":8000}`, "max_output_tokens=8000",
		},
		"at the ceiling stands": {
			16384, `{"model":"m","input":"write the long tool call","max_output_tokens":16384}`, "max_output_tokens=16384",
		},
		"no declared ceiling clamps nothing": {0, armCeilingBody, "max_output_tokens=100000"},
	} {
		t.Run(name, func(t *testing.T) {
			request, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
				target: llmprotocol.OpenAIResponsesV1, card: test.card, body: test.body,
				decision: armCeilingDecision(t, 0),
			})
			if got := wireOutputLimit(wire); got != test.want {
				t.Fatalf("dispatched %q, want %q", got, test.want)
			}
			// The lowered limit is still the caller's allowance, not a limit the
			// router invented for a caller that stated none, so a reasoning
			// bound keeps deriving from it.
			if request.RouterSetMaxOutputTokens {
				t.Fatal("a clamped caller limit was marked as the router's own")
			}
			// Setting this would have the Chat encoder re-assert the caller's
			// original number over the clamp (ClientOutputAllowance).
			if request.ClientMaxOutputTokens != nil {
				t.Fatalf("ClientMaxOutputTokens = %v, want it left unset", *request.ClientMaxOutputTokens)
			}
		})
	}
}

// The clamp reaches every target format the arms accept, and on Chat it
// survives the encoder, which re-asserts ClientMaxOutputTokens when a plugin
// raised the allowance above the caller's number.
func TestARCArmCeilingLowersOnEveryTargetFormat(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
	} {
		_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
			target: target, card: 16384, body: armCeilingBody, decision: armCeilingDecision(t, 0),
		})
		got := wireOutputLimit(wire)
		if got == "" || got[len(got)-6:] != "=16384" {
			t.Fatalf("-> %s: dispatched %q, want the arm's 16384", target, got)
		}
	}
}

// A plain decision leaves a stated limit alone, whatever the card says: the
// caller picked that model themselves, so the limit is between them and it.
func TestARCArmCeilingLeavesAPlainDecisionAlone(t *testing.T) {
	for name, decision := range map[string]*config.Decision{
		"no decision at all":             nil,
		"a decision with no algorithm":   maxTokensLimitDecision(t, 200000),
		"a plain static decision":        plainDecision(),
		"an arc type without its config": arcTypeWithoutItsBlock(),
	} {
		t.Run(name, func(t *testing.T) {
			_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
				target: llmprotocol.OpenAIResponsesV1, card: 16384, body: armCeilingBody, decision: decision,
			})
			if got := wireOutputLimit(wire); got != "max_output_tokens=100000" {
				t.Fatalf("dispatched %q, want the caller's 100000 untouched", got)
			}
		})
	}
}

// The arm's ceiling, the decision's max_tokens_limit and the room the prompt
// leaves in the arm's context window compose as one rule, so the limit
// dispatched is the least of them and the caller's own number. Reading them
// from a single cap is what stops one clamp undoing another: a ceiling above
// the room cannot raise the limit back over the room, nor a room above the
// ceiling over the ceiling.
func TestARCArmCeilingComposesWithTheContextAndDecisionClamps(t *testing.T) {
	for name, test := range map[string]struct {
		card          int
		limit         int
		window        int
		contextTokens int
		want          string
	}{
		"the ceiling is tightest":      {16384, 0, 1000000, 100, "max_output_tokens=16384"},
		"the context room is tightest": {16384, 0, 20000, 10000, "max_output_tokens=10000"},
		// The request_params plugin has already capped the caller to 4000 by
		// the time dispatch runs, so this one checks the two agree rather than
		// the planner's own cap arithmetic; the case below does that.
		"the decision cap is tightest": {16384, 4000, 1000000, 100, "max_output_tokens=4000"},
		// A cap under the ceiling with the room tighter still: the planner, not
		// the earlier cap, has to take the least of the three.
		"the room beats a cap under the ceiling": {16384, 9000, 20000, 18000, "max_output_tokens=2000"},
		"the room is above the ceiling":          {16384, 0, 1000000, 900000, "max_output_tokens=16384"},
		"no room leaves the ceiling":             {16384, 0, 10000, 10000, "max_output_tokens=16384"},
		"no window leaves the ceiling":           {16384, 0, 0, 10000, "max_output_tokens=16384"},
	} {
		t.Run(name, func(t *testing.T) {
			_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
				target: llmprotocol.OpenAIResponsesV1, card: test.card, body: armCeilingBody,
				decision: armCeilingDecision(t, test.limit),
				window:   test.window, contextTokens: test.contextTokens,
			})
			if got := wireOutputLimit(wire); got != test.want {
				t.Fatalf("dispatched %q, want %q", got, test.want)
			}
		})
	}
}

// A request that states no limit still takes the card-default path: the clamp
// only ever lowers a number the caller stated, and adds nothing where the
// router already supplies one.
func TestARCArmCeilingLeavesTheCardDefaultPathAlone(t *testing.T) {
	request, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
		target: llmprotocol.AnthropicMessagesV1, card: 16384, body: unboundedResponsesBody,
		decision: armCeilingDecision(t, 0),
	})
	if got := wireOutputLimit(wire); got != "max_tokens=16384" {
		t.Fatalf("dispatched %q, want the card's 16384", got)
	}
	if !request.RouterSetMaxOutputTokens {
		t.Fatal("the router's own bound is not marked as the router's")
	}
}

// Messages refuses a thinking budget that is not below max_tokens, so a budget
// the lowered limit cannot hold comes down with it, as it does for a card
// limit. The arm would refuse the pair otherwise.
func TestARCArmCeilingLowersAThinkingBudgetItCannotHold(t *testing.T) {
	budget := int64(40000)
	request, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
		target: llmprotocol.AnthropicMessagesV1, card: 8000,
		body:     `{"model":"m","input":"write the long tool call","max_output_tokens":100000}`,
		decision: armCeilingDecision(t, 10000),
		mutate: func(request *llmprotocol.Request) {
			request.ReasoningMode = llmprotocol.ReasoningModeEnabled
			request.ReasoningBudgetTokens = &budget
		},
	})
	if got := wireOutputLimit(wire); got != "max_tokens=8000" {
		t.Fatalf("dispatched %q, want the arm's 8000", got)
	}
	// The budget is reconciled against the limit the request leaves with, not
	// the one it arrived with: the arm validates the pair it is sent.
	if got := request.ReasoningBudgetTokens; got == nil || *got != minimumAnthropicThinkingBudget {
		t.Fatalf("thinking budget = %v, want it lowered to %d under the arm's 8000", got, minimumAnthropicThinkingBudget)
	}
}

// An arm whose ceiling cannot hold the minimum thinking budget and an output
// token still gets the clamp: the limit comes down to the arm and the thinking
// that does not fit beside it is turned off, rather than sent as a pair the arm
// refuses.
func TestARCArmCeilingTurnsOffThinkingTheArmCannotHold(t *testing.T) {
	budget := int64(40000)
	request, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
		target: llmprotocol.AnthropicMessagesV1, card: 1000,
		body:     `{"model":"m","input":"write the long tool call","max_output_tokens":100000}`,
		decision: armCeilingDecision(t, 0),
		mutate: func(request *llmprotocol.Request) {
			request.ReasoningMode = llmprotocol.ReasoningModeEnabled
			request.ReasoningBudgetTokens = &budget
		},
	})
	if got := wireOutputLimit(wire); got != "max_tokens=1000" {
		t.Fatalf("dispatched %q, want the arm's 1000", got)
	}
	if request.ReasoningBudgetTokens != nil || request.ReasoningMode != llmprotocol.ReasoningModeDisabled {
		t.Fatalf("thinking = %v/%v, want it turned off", request.ReasoningMode, request.ReasoningBudgetTokens)
	}
}

// The context room binds the clamped limit even when the thinking beside it is
// turned off for not fitting. An earlier revision re-derived the limit from the
// arm's ceiling once thinking went, which sent the ceiling whole into a window
// that had no room for it -- the very refusal the context clamp exists to stop.
func TestARCArmCeilingKeepsTheContextRoomWhenThinkingIsTurnedOff(t *testing.T) {
	budget := int64(40000)
	request, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
		target: llmprotocol.AnthropicMessagesV1, card: 8000, body: armCeilingBody,
		decision: armCeilingDecision(t, 0),
		// 500 tokens of room, far under the arm's 8000 ceiling.
		window: 20000, contextTokens: 19500,
		mutate: func(request *llmprotocol.Request) {
			request.ReasoningMode = llmprotocol.ReasoningModeEnabled
			request.ReasoningBudgetTokens = &budget
		},
	})
	if got := wireOutputLimit(wire); got != "max_tokens=500" {
		t.Fatalf("dispatched %q, want the context room of 500, not the arm's 8000", got)
	}
	if request.ReasoningMode != llmprotocol.ReasoningModeDisabled {
		t.Fatalf("thinking = %v, want it turned off under a 500-token room", request.ReasoningMode)
	}
}

// A caller already at or under the arm's ceiling still has the context room
// applied to it. An earlier revision returned early on that comparison, which
// skipped the context clamp for every caller-stated limit on the ARC path.
func TestARCArmCeilingAppliesTheContextRoomUnderTheCeiling(t *testing.T) {
	_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
		target: llmprotocol.OpenAIResponsesV1, card: 16384,
		body: `{"model":"m","input":"write the long tool call","max_output_tokens":8000}`,
		// 2000 tokens of room: under the caller's 8000 and under the ceiling.
		decision: armCeilingDecision(t, 0), window: 20000, contextTokens: 18000,
	})
	if got := wireOutputLimit(wire); got != "max_output_tokens=2000" {
		t.Fatalf("dispatched %q, want the context room of 2000", got)
	}
}

// A room too small for the target's own minimum sends the caller's number
// rather than a limit the target would refuse, as the card path does. Responses
// refuses max_output_tokens below 16.
func TestARCArmCeilingLeavesTheCallerWhenTheRoomIsUnderTheTargetMinimum(t *testing.T) {
	_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
		target: llmprotocol.OpenAIResponsesV1, card: 16384, body: armCeilingBody,
		decision: armCeilingDecision(t, 0), window: 20000, contextTokens: 19988,
	})
	if got := wireOutputLimit(wire); got != "max_output_tokens=100000" {
		t.Fatalf("dispatched %q, want the caller's 100000 where no limit is sendable", got)
	}
}

// A v5 control the arm's ceiling cannot hold names the arm, not an operator's
// max_tokens_limit. The planner only knows a cap as the decision's, so the
// clamp relabels it: an operator sent after a cap they never set has nothing
// to change.
func TestARCArmCeilingBlamesTheArmNotTheDecisionForAV5Refusal(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	params := router.Config.ModelConfig[model]
	params.MaxOutputTokens = 1000
	router.Config.ModelConfig[model] = params
	budget := int64(40000)
	ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, nil)
	// No request_params plugin at all, so no max_tokens_limit is configured.
	ctx.VSRSelectedDecision = armCeilingDecision(t, 0)
	ctx.RaylineARCThinkingControl = &plannedThinkingControl{
		control: thinkingcontrol.Control{Native: "enabled", BudgetTokens: &budget},
	}
	request := testNeutralRequest(model, "hi")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(100000)
	ctx.SemanticRequest = request
	dispatch, err := router.prepareProviderDispatch(request, model, "", false, ctx)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	_, err = router.boundDispatchRequest(request, dispatch, ctx)
	if !errors.Is(err, errThinkingControlOverArm) {
		t.Fatalf("bound = %v, want the arm's own refusal", err)
	}
}

// A caller already at the arm's ceiling needs no lowering of the number, but a
// v5 thinking budget the ceiling cannot seat still has to come down: Messages
// counts the budget inside max_tokens, and RaiseMessagesAllowance would
// otherwise raise max_tokens over the budget and so back over the ceiling.
// Measured before the fix: caller 8000, ceiling 8000, budget 40000 dispatched
// max_tokens=48000, six times the arm's figure.
func TestARCArmCeilingHoldsAV5BudgetWhenTheNumberStands(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	params := router.Config.ModelConfig[model]
	params.MaxOutputTokens = 8000
	router.Config.ModelConfig[model] = params
	budget := int64(40000)
	ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, nil)
	ctx.VSRSelectedDecision = armCeilingDecision(t, 0)
	ctx.RaylineARCThinkingControl = &plannedThinkingControl{
		control: thinkingcontrol.Control{Native: "enabled", BudgetTokens: &budget},
	}
	request := testNeutralRequest(model, "hi")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(8000)
	ctx.SemanticRequest = request
	dispatch, err := router.prepareProviderDispatch(request, model, "", false, ctx)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if _, err := router.boundDispatchRequest(request, dispatch, ctx); err != nil {
		t.Fatalf("bound: %v", err)
	}
	if got := *request.Sampling.MaxOutputTokens; got > 8000 {
		t.Fatalf("max_tokens = %d, want it to stay within the arm's 8000", got)
	}
	if got := ctx.RaylineARCThinkingControl.control.BudgetTokens; got == nil || *got != minimumAnthropicThinkingBudget {
		t.Fatalf("v5 budget = %v, want it lowered to %d under the arm's ceiling", got, minimumAnthropicThinkingBudget)
	}
	if budget != 40000 {
		t.Fatal("the action's own budget was changed; only the planned copy may be")
	}
}

// A completion floor the request_params plugin raised the allowance to is the
// Router's number, and a floor above the chosen arm's ceiling is a limit that
// arm refuses just as a card limit would be. The floor branch returned before
// the ceiling was ever consulted, so a 65536 floor dispatched whole against a
// 16384 arm.
func TestARCArmCeilingHoldsACompletionFloorToTheArm(t *testing.T) {
	logicalModel := "target-" + string(llmprotocol.OpenAIChatV1)
	for name, test := range map[string]struct {
		window        int
		contextTokens int
		want          string
	}{
		// The floor fits the window, so the old code returned early.
		"a floor the window has room for": {200000, 1000, "max_completion_tokens=16384"},
		// The floor is clamped to the room first, and the room is still above
		// the arm's ceiling.
		"a floor clamped to a room above the ceiling": {100000, 60000, "max_completion_tokens=16384"},
		// The room is under the ceiling, so the room is what binds.
		"a room under the ceiling binds": {20000, 12000, "max_completion_tokens=8000"},
	} {
		t.Run(name, func(t *testing.T) {
			decision := completionFloorDecision(map[string]interface{}{logicalModel: 65536})
			decision.Algorithm = &config.AlgorithmConfig{
				Type: config.RaylineARCAlgorithmType, RaylineARC: &config.RaylineARCAlgorithmConfig{},
			}
			_, wire := dispatchWithOutputBound(t, dispatchOutputBoundCase{
				target: llmprotocol.OpenAIChatV1, card: 16384, body: unboundedResponsesBody,
				decision: decision, window: test.window, contextTokens: test.contextTokens,
			})
			if got := wireOutputLimit(wire); got != test.want {
				t.Fatalf("dispatched %q, want %q", got, test.want)
			}
		})
	}
}

// Context compression reserves no more than the arm's ceiling, since dispatch
// will lower the caller's limit to it. Reserving the caller's whole number
// frees room the turn never sends: on a 128k window a 100k caller limit
// against a 16k arm discarded input down to 28k when 112k could have stayed.
func TestARCArmCeilingBoundsTheCompressionReserve(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	params := router.Config.ModelConfig[model]
	params.MaxOutputTokens = 16384
	params.ContextWindowSize = 128000
	router.Config.ModelConfig[model] = params
	request := testNeutralRequest(model, "hi")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(100000)

	arc := routingTestContext(llmprotocol.OpenAIChatV1, nil)
	arc.VSRSelectedDecision, arc.VSRSelectedModel = armCeilingDecision(t, 0), model
	arc.SemanticRequest = request
	if got := router.compressionOutputReserve(model, arc, request); got != 16384 {
		t.Fatalf("ARC reserve = %d, want the arm's 16384", got)
	}
	// A plain decision keeps reserving the caller's own number: nothing will
	// lower it later, so freeing less room would truncate the answer.
	plain := routingTestContext(llmprotocol.OpenAIChatV1, nil)
	plain.VSRSelectedDecision, plain.VSRSelectedModel = plainDecision(), model
	plain.SemanticRequest = request
	if got := router.compressionOutputReserve(model, plain, request); got != 100000 {
		t.Fatalf("plain reserve = %d, want the caller's 100000", got)
	}
}
