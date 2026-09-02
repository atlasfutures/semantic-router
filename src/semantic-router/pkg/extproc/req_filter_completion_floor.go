package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

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
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(int64(floor))
	metrics.RecordCompletionFloorApplied(decisionKey, logicalModel)
	logging.Infof("Raised completion budget for model %q to its configured floor %d", logicalModel, floor)
	return true
}
