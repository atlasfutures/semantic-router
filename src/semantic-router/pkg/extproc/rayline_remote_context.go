package extproc

import (
	"bytes"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylineremote"
)

func buildRaylineRemoteSelectionContext(
	algorithm *config.AlgorithmConfig,
	reqCtx *RequestContext,
) *selection.RaylineRemoteSelectionContext {
	if algorithm == nil ||
		algorithm.Type != config.RaylineRemoteAlgorithmType ||
		algorithm.RaylineRemote == nil {
		return nil
	}
	result := &selection.RaylineRemoteSelectionContext{}
	if reqCtx == nil {
		result.PreparationFailure = "missing_request"
		return result
	}
	if reqCtx.ClientProtocol != "" ||
		(reqCtx.ResponseAPICtx != nil &&
			reqCtx.ResponseAPICtx.IsResponseAPIRequest) {
		result.PreparationFailure = "unsupported_protocol"
		return result
	}
	rawEpisodeID := bytes.TrimSpace([]byte(
		reqCtx.Headers[algorithm.RaylineRemote.EpisodeIDHeader],
	))
	if len(rawEpisodeID) == 0 {
		result.PreparationFailure = "missing_episode_id"
		return result
	}
	decisionID, err := raylineremote.MintDecisionID()
	if err != nil {
		result.PreparationFailure = "decision_id"
		return result
	}
	result.DecisionID = decisionID
	result.RawEpisodeID = string(rawEpisodeID)
	result.RequestBody = bytes.Clone(reqCtx.OriginalRequestBody)
	if len(result.RequestBody) == 0 {
		result.PreparationFailure = "missing_request_body"
	}
	return result
}
