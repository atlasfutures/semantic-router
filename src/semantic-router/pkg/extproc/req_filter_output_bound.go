package extproc

import (
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// dispatchFallbackMaxOutputTokens is the output limit a request that states
// none is sent with when its target requires one (Anthropic Messages) and the
// worker's model card declares none. It is valid for every current Claude
// model and matches what Claude Code sends by default. At a slow arm's rate it
// can outlast the response stream deadline, so slow arms set their card's
// max_output_tokens instead.
const dispatchFallbackMaxOutputTokens int64 = 32000

// Where a dispatched output limit came from, as logged.
const (
	outputBoundSourceCard     = "card"
	outputBoundSourceFallback = "fallback"
)

// errThinkingControlOverCap is a v5 control whose thinking budget the
// decision's max_tokens_limit cannot hold at Anthropic's minimum.
var errThinkingControlOverCap = errors.New(
	"the decision's max_tokens_limit cannot hold the v5 control's thinking budget at its 1024-token minimum",
)

// applyDispatchOutputBound sets the output limit of a request whose caller
// stated none, as planDispatchOutputBound decides it, and logs the plan as the
// dispatch_output_bound event. A limit the caller stated is never touched.
//
// It runs after every other output-allowance rule (request_params cap and
// floor, a policy action's budget), so it only fills what they left unset.
//
// On a v5 turn the planned control owns thinking, so its budget is the one
// planned. A lowered budget is written to the planned control, which the
// placer renders after encoding. A control the cap cannot hold fails the
// turn: turning its thinking off would change the control's base, which the
// placer fixes per episode and which would serve an action the policy did
// not choose, and sending it whole would exceed the operator's cap.
func (r *OpenAIRouter) applyDispatchOutputBound(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) (bool, error) {
	if request == nil || dispatch == nil || ctx == nil || request.Sampling.MaxOutputTokens != nil {
		return false, nil
	}
	control := plannedMessagesControl(ctx, dispatch.targetFormat)
	budget := messagesThinkingBudget(request, dispatch.targetFormat)
	if control != nil {
		budget = control.control.BudgetTokens
	}
	plan := r.planDispatchOutputBound(
		dispatch.logicalModel, dispatch.targetFormat, decisionMaxTokensLimit(ctx), budget,
	)
	if plan.source == "" {
		return false, nil
	}
	event := map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       dispatch.logicalModel,
		"wire_format": dispatch.targetFormat,
		"source":      plan.source,
	}
	if plan.belowMinimum != "" {
		event["below_target_minimum"] = plan.belowMinimum
	}
	if plan.thinking != "" {
		event["thinking"] = plan.thinking
	}
	if control != nil {
		event["thinking_source"] = "v5_control"
	}
	switch plan.thinking {
	case outputBoundThinkingBudgetLowered:
		lowered := plan.thinkingBudget
		event["thinking_budget_tokens"] = lowered
		if control != nil {
			control.control.BudgetTokens = &lowered
		} else {
			request.ReasoningBudgetTokens = &lowered
		}
	case outputBoundThinkingDisabled:
		if control != nil {
			event["max_output_tokens"] = nil
			logging.ComponentWarnEvent("extproc", "dispatch_output_bound", event)
			return false, errThinkingControlOverCap
		}
		request.ReasoningMode, request.ReasoningBudgetTokens = llmprotocol.ReasoningModeDisabled, nil
	}
	if plan.maxTokens == 0 {
		event["max_output_tokens"] = nil
		logging.ComponentEvent("extproc", "dispatch_output_bound", event)
		return false, nil
	}
	bound := plan.maxTokens
	request.Sampling.MaxOutputTokens = &bound
	request.RouterSetMaxOutputTokens = true
	event["max_output_tokens"] = bound
	logging.ComponentEvent("extproc", "dispatch_output_bound", event)
	return true, nil
}

// plannedMessagesControl is this dispatch's v5 control when it renders
// Messages thinking, or nil.
func plannedMessagesControl(ctx *RequestContext, format llmprotocol.WireFormat) *plannedThinkingControl {
	if format != llmprotocol.AnthropicMessagesV1 {
		return nil
	}
	return ctx.RaylineARCThinkingControl
}

// How a planned bound met the target's minimum, as logged.
const (
	outputBoundRaisedToMinimum = "raised_to_minimum"
	outputBoundLeftUnbounded   = "left_unbounded"
)

// How a planned bound met a Messages thinking budget, as logged.
const (
	outputBoundThinkingOnTop         = "budget_on_top"
	outputBoundThinkingBudgetLowered = "budget_lowered"
	outputBoundThinkingDisabled      = "thinking_disabled"
)

// minimumAnthropicThinkingBudget is the smallest thinking.budget_tokens
// Anthropic accepts.
const minimumAnthropicThinkingBudget int64 = 1024

// outputBoundPlan is the output limit a request that states none is sent
// with. A zero maxTokens sends none; an empty source means no rule applied.
type outputBoundPlan struct {
	maxTokens      int64
	source         string
	belowMinimum   string
	thinking       string
	thinkingBudget int64
}

// planDispatchOutputBound decides the output limit for a request that states
// none, to model on format. Dispatch applies it, and context compression
// reserves it before dispatch, so the two cannot disagree.
//
//   - The bound is the model card's operator-declared max_output_tokens, or,
//     only for a target that requires a limit (Messages), the fallback. Chat
//     and Responses targets with no card value stay unbounded.
//   - It stays within the decision's max_tokens_limit.
//   - It is never a limit the target refuses. A bound below the target's
//     minimum is raised to it, which keeps a small card value or cap as close
//     as the target allows; turning it into no limit would discard the
//     operator's intent. Only when the minimum itself is above the cap is the
//     request left unbounded, as its caller sent it: the router does not send
//     a limit above the operator's cap.
//   - On Messages, which counts thinking inside max_tokens and refuses a
//     budget that is not below it, an enabled budget at or above the bound
//     gets the bound on top of it, the way a caller's allowance is kept on
//     top. Under a cap the total never exceeds it: max_tokens becomes the cap
//     and the budget is lowered to what the cap leaves beside the bound, never
//     below Anthropic's 1024 minimum. A cap that cannot hold the minimum
//     budget and one output token turns thinking off for the dispatch. No
//     existing rule reconciles a caller's capped limit with a budget, so this
//     one is the router's own.
func (r *OpenAIRouter) planDispatchOutputBound(
	model string,
	format llmprotocol.WireFormat,
	decisionLimit int64,
	thinkingBudget *int64,
) outputBoundPlan {
	bound, source := r.dispatchOutputBound(model, format)
	if source == "" {
		return outputBoundPlan{}
	}
	plan := outputBoundPlan{source: source}
	if decisionLimit > 0 && bound > decisionLimit {
		bound = decisionLimit
	}
	if minimum := minimumOutputLimit(format); bound < minimum {
		if decisionLimit > 0 && minimum > decisionLimit {
			plan.belowMinimum = outputBoundLeftUnbounded
			return plan
		}
		plan.belowMinimum = outputBoundRaisedToMinimum
		bound = minimum
	}
	plan.maxTokens = bound
	if thinkingBudget == nil || *thinkingBudget < bound {
		return plan
	}
	if total := bound + *thinkingBudget; decisionLimit <= 0 || total <= decisionLimit {
		plan.maxTokens, plan.thinking = total, outputBoundThinkingOnTop
		return plan
	}
	lowered := max(decisionLimit-bound, minimumAnthropicThinkingBudget)
	if lowered >= decisionLimit {
		plan.thinking = outputBoundThinkingDisabled
		return plan
	}
	plan.maxTokens, plan.thinking, plan.thinkingBudget = decisionLimit, outputBoundThinkingBudgetLowered, lowered
	return plan
}

// minimumOutputLimit is the smallest output limit a target accepts.
// Responses refuses max_output_tokens below 16. Messages requires max_tokens
// of at least 1, and Chat accepts any non-negative max_completion_tokens; a
// card value is positive and a non-positive max_tokens_limit is ignored, so
// a bound for either is never below its minimum and never left unbounded by
// it (Messages would then get the codec's 32000, above the cap).
func minimumOutputLimit(format llmprotocol.WireFormat) int64 {
	if format == llmprotocol.OpenAIResponsesV1 {
		return 16
	}
	return 1
}

// dispatchOutputBound is the uncapped limit for a request that states none,
// and its source; an empty source means the request is sent without one.
func (r *OpenAIRouter) dispatchOutputBound(model string, format llmprotocol.WireFormat) (int64, string) {
	if r != nil && r.Config != nil {
		// A LoRA adapter dispatches under its own name and takes its base
		// model's card.
		if limit := r.Config.GetModelMaxOutputTokens(model); limit > 0 {
			return int64(limit), outputBoundSourceCard
		}
	}
	if format == llmprotocol.AnthropicMessagesV1 {
		return dispatchFallbackMaxOutputTokens, outputBoundSourceFallback
	}
	return 0, ""
}

// decisionMaxTokensLimit is the selected decision's request_params
// max_tokens_limit, or zero when it sets none.
func decisionMaxTokensLimit(ctx *RequestContext) int64 {
	if ctx.VSRSelectedDecision == nil {
		return 0
	}
	params := ctx.VSRSelectedDecision.GetRequestParamsConfig()
	if params == nil || params.MaxTokensLimit == nil || *params.MaxTokensLimit <= 0 {
		return 0
	}
	return int64(*params.MaxTokensLimit)
}

// messagesThinkingBudget is the thinking budget the Messages encoder will
// validate against max_tokens, or nil when there is none to keep room for.
func messagesThinkingBudget(request *llmprotocol.Request, format llmprotocol.WireFormat) *int64 {
	if format != llmprotocol.AnthropicMessagesV1 || request.ReasoningBudgetTokens == nil {
		return nil
	}
	if request.ReasoningMode != "" && request.ReasoningMode != llmprotocol.ReasoningModeEnabled {
		return nil
	}
	return request.ReasoningBudgetTokens
}
