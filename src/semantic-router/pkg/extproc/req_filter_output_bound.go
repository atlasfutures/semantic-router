package extproc

import (
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

// applyDispatchOutputBound sets the output limit of a request whose caller
// stated none: the worker's card max_output_tokens when the operator declared
// one, otherwise, only for a target that requires a limit, the fallback.
// Chat and Responses targets with no card value stay unbounded, as the caller
// asked. A limit the caller stated is never touched.
//
// It runs after every other output-allowance rule (request_params cap and
// floor, a policy action's budget), so it only fills what they left unset.
// The bound stays within the decision's max_tokens_limit and at or above the
// target's minimum (minimumOutputLimit). On Messages, which counts thinking
// inside max_tokens and refuses a budget that leaves no room below it, an
// enabled thinking budget at or above the bound gets the bound on top of it,
// the way a caller's allowance is kept on top.
func (r *OpenAIRouter) applyDispatchOutputBound(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) bool {
	if request == nil || dispatch == nil || ctx == nil || request.Sampling.MaxOutputTokens != nil {
		return false
	}
	bound, source := r.dispatchOutputBound(dispatch)
	if source == "" {
		return false
	}
	decisionLimit := decisionMaxTokensLimit(ctx)
	if decisionLimit > 0 && bound > decisionLimit {
		bound = decisionLimit
	}
	event := map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       dispatch.logicalModel,
		"wire_format": dispatch.targetFormat,
		"source":      source,
	}
	// Never synthesize a limit the target refuses. A bound below the
	// target's minimum is raised to it, which keeps a small card value or
	// cap as close as the target allows; turning it into no limit would
	// discard the operator's intent entirely. Only when the minimum itself
	// is above the decision's max_tokens_limit is the request left
	// unbounded, as its caller sent it: the router does not send a limit
	// above the operator's cap.
	if minimum := minimumOutputLimit(dispatch.targetFormat); bound < minimum {
		if decisionLimit > 0 && minimum > decisionLimit {
			event["below_target_minimum"] = "left_unbounded"
			event["max_output_tokens"] = nil
			logging.ComponentEvent("extproc", "dispatch_output_bound", event)
			return false
		}
		event["below_target_minimum"] = "raised_to_minimum"
		bound = minimum
	}
	if budget := messagesThinkingBudget(request, dispatch.targetFormat); budget != nil && *budget >= bound {
		bound += *budget
	}
	request.Sampling.MaxOutputTokens = &bound
	request.RouterSetMaxOutputTokens = true
	event["max_output_tokens"] = bound
	logging.ComponentEvent("extproc", "dispatch_output_bound", event)
	return true
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

// dispatchOutputBound is the limit for a request that states none, and its
// source; an empty source means the request is sent without one.
func (r *OpenAIRouter) dispatchOutputBound(dispatch *providerDispatch) (int64, string) {
	if r != nil && r.Config != nil {
		// A LoRA adapter dispatches under its own name and takes its base
		// model's card.
		if limit := r.Config.GetModelMaxOutputTokens(dispatch.logicalModel); limit > 0 {
			return int64(limit), outputBoundSourceCard
		}
	}
	if dispatch.targetFormat == llmprotocol.AnthropicMessagesV1 {
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
