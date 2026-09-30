package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (r *OpenAIRouter) applyDispatchRequestParams(
	request *llmprotocol.Request,
	ctx *RequestContext,
) (bool, error) {
	if ctx.VSRSelectedDecision != nil && ctx.VSRSelectedDecision.GetRequestParamsConfig() != nil {
		return r.applySemanticRequestParams(
			ctx.VSRSelectedDecision, request, ctx.Routing.RecipeName(),
		)
	}
	return false, nil
}

// applyDispatchCompletionFloor raises the output allowance to the floor the
// dispatched model declares, after the request_params plugin has blocked and
// capped. It is its own seam so the upstream request_params step keeps its
// signature; the floor needs the dispatched logical model.
func applyDispatchCompletionFloor(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) bool {
	if dispatch == nil || ctx == nil || ctx.VSRSelectedDecision == nil {
		return false
	}
	params := ctx.VSRSelectedDecision.GetRequestParamsConfig()
	if params == nil {
		return false
	}
	decisionKey := config.RoutingDecisionKey(ctx.Routing.RecipeName(), ctx.VSRSelectedDecision.Name)
	return applyCompletionTokenFloor(params, request, dispatch.logicalModel, decisionKey)
}

// applyCompletionTokenFloor raises the completion budget to the floor the
// selected model declares.
//
// A reasoning model spends part of its budget on thinking, so a client budget
// sized for a plain answer truncates the answer instead of shortening it. The
// floor is keyed by logical model because one decision routes to a menu of
// arms, and only the thinking arms need it.
//
// The floor runs after the request_params cap, so a floor above
// max_tokens_limit wins. Both are operator settings on the same plugin; a
// config that sets a floor above its own cap is contradictory, and the floor
// is the one that protects the answer.
//
// The raise applies to the output allowance and to nothing else. A reasoning
// bound is derived from what the client allowed, so the client's own number is
// kept on the request before it is overwritten: measured on the dev cell
// 2026-09-04, a bound derived from the 65,536 floor let a client asking for
// 512 output tokens spend 65,536 on reasoning, which bounded nothing.
func applyCompletionTokenFloor(
	params *config.RequestParamsPluginConfig,
	request *llmprotocol.Request,
	logicalModel string,
	decisionKey string,
) bool {
	if params == nil || request == nil || len(params.MinCompletionTokensByModel) == 0 {
		return false
	}
	floor, configured := params.MinCompletionTokensByModel[logicalModel]
	if !configured || floor <= 0 {
		return false
	}
	if request.Sampling.MaxOutputTokens != nil && *request.Sampling.MaxOutputTokens >= int64(floor) {
		return false
	}
	request.ClientMaxOutputTokens = request.Sampling.MaxOutputTokens
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(int64(floor))
	metrics.RecordCompletionFloorApplied(decisionKey, logicalModel)
	logging.Infof("Raised completion budget for model %q to its configured floor %d", logicalModel, floor)
	return true
}
