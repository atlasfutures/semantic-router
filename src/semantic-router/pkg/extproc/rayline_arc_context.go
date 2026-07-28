/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func (r *OpenAIRouter) buildRaylineARCSelectionContext(
	algorithm *config.AlgorithmConfig,
	reqCtx *RequestContext,
	workerCount int,
) *selection.RaylineARCSelectionContext {
	if algorithm == nil ||
		algorithm.Type != config.RaylineARCAlgorithmType ||
		algorithm.RaylineARC == nil {
		return nil
	}
	result := &selection.RaylineARCSelectionContext{}
	if reqCtx == nil {
		result.PreparationFailure = "missing_request"
		return result
	}
	rawEpisodeID := strings.TrimSpace(
		reqCtx.Headers[algorithm.RaylineARC.Episode.IDHeader],
	)
	if rawEpisodeID == "" {
		result.PreparationFailure = "missing_episode_id"
		return result
	}
	result.EpisodeIDHash = raylinearc.HashEpisodeID(rawEpisodeID)
	state, failure := r.prepareRaylineARCTransaction(
		algorithm.RaylineARC,
		reqCtx,
		result.EpisodeIDHash,
		workerCount,
	)
	if failure != "" {
		result.PreparationFailure = failure
		return result
	}
	result.State = state
	turns, err := normalizeRaylineARCTurns(reqCtx)
	if err != nil {
		code := raylinearc.TurnNormalizationErrorCode(err)
		if code == "" {
			code = "invalid_turns"
		}
		result.PreparationFailure = "turns_" + code
		r.finalizeRaylineARCAbort(reqCtx, result.PreparationFailure)
		return result
	}
	result.Turns = turns
	return result
}

func (r *OpenAIRouter) prepareRaylineARCTransaction(
	arcConfig *config.RaylineARCAlgorithmConfig,
	reqCtx *RequestContext,
	episodeIDHash string,
	workerCount int,
) (*raylinearc.EpisodeState, string) {
	if r == nil || r.RaylineARCEpisodeStore == nil {
		return nil, "episode_store"
	}
	prepareContext := reqCtx.TraceContext
	if prepareContext == nil {
		prepareContext = context.Background()
	}
	prepareContext, cancel := context.WithTimeout(
		prepareContext,
		time.Duration(
			arcConfig.Episode.AcquireTimeoutSeconds,
		)*time.Second,
	)
	defer cancel()
	lease, state, err := r.RaylineARCEpisodeStore.Prepare(
		prepareContext,
		episodeIDHash,
		workerCount,
	)
	if err != nil {
		return nil, boundedARCPrepareFailure(err)
	}
	reqCtx.RaylineARCTransaction = newRaylineARCEpisodeTransaction(
		r.RaylineARCEpisodeStore,
		lease,
		state,
		episodeIDHash,
		time.Duration(
			arcConfig.Episode.LeaseTTLSeconds,
		)*time.Second,
	)
	return state, ""
}

func normalizeRaylineARCTurns(
	reqCtx *RequestContext,
) ([]raylinearc.Turn, error) {
	protocol := raylinearc.ProtocolOpenAIChat
	switch {
	case reqCtx.ResponseAPICtx != nil &&
		reqCtx.ResponseAPICtx.IsResponseAPIRequest:
		protocol = raylinearc.ProtocolOpenAIResponses
	case reqCtx.ClientProtocol == config.ClientProtocolAnthropic:
		protocol = raylinearc.ProtocolAnthropicMessages
	}
	return raylinearc.NormalizeTurns(
		protocol,
		reqCtx.OriginalRequestBody,
	)
}

func boundedARCPrepareFailure(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "episode_canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "episode_timeout"
	case errors.Is(err, raylinearc.ErrEpisodeCapacity):
		return "episode_capacity"
	default:
		return "episode_store"
	}
}
