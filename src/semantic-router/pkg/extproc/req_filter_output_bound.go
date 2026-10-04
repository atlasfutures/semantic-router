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
// The bound stays within the decision's max_tokens_limit, and on Messages,
// which counts thinking inside max_tokens and refuses a budget that leaves no
// room below it, an enabled thinking budget at or above the bound gets the
// bound on top of it, the way a caller's allowance is kept on top.
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
	bound = capOutputBoundAtDecisionLimit(bound, ctx)
	if budget := messagesThinkingBudget(request, dispatch.targetFormat); budget != nil && *budget >= bound {
		bound += *budget
	}
	request.Sampling.MaxOutputTokens = &bound
	request.RouterSetMaxOutputTokens = true
	logging.ComponentEvent("extproc", "dispatch_output_bound", map[string]interface{}{
		"request_id":        ctx.RequestID,
		"model":             dispatch.logicalModel,
		"wire_format":       dispatch.targetFormat,
		"max_output_tokens": bound,
		"source":            source,
	})
	return true
}

// dispatchOutputBound is the limit for a request that states none, and its
// source; an empty source means the request is sent without one.
func (r *OpenAIRouter) dispatchOutputBound(dispatch *providerDispatch) (int64, string) {
	if r != nil && r.Config != nil {
		if params, ok := r.Config.ModelConfig[dispatch.logicalModel]; ok && params.MaxOutputTokens > 0 {
			return int64(params.MaxOutputTokens), outputBoundSourceCard
		}
	}
	if dispatch.targetFormat == llmprotocol.AnthropicMessagesV1 {
		return dispatchFallbackMaxOutputTokens, outputBoundSourceFallback
	}
	return 0, ""
}

func capOutputBoundAtDecisionLimit(bound int64, ctx *RequestContext) int64 {
	if ctx.VSRSelectedDecision == nil {
		return bound
	}
	params := ctx.VSRSelectedDecision.GetRequestParamsConfig()
	if params == nil || params.MaxTokensLimit == nil || *params.MaxTokensLimit <= 0 {
		return bound
	}
	if limit := int64(*params.MaxTokensLimit); bound > limit {
		return limit
	}
	return bound
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
