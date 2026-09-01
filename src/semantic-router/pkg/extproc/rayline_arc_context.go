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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
	if failure := parseRaylineARCCloseRequest(
		algorithm.RaylineARC.Episode.CloseHeader,
		reqCtx,
	); failure != "" {
		result.PreparationFailure = failure
		return result
	}
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
	turns, err := r.projectRaylineARCTurns(
		reqCtx,
		raylinearc.TurnOptions{
			IncludeSystemText: algorithm.RaylineARC.IncludeSystemText,
			DropMidConversationSystemText: algorithm.RaylineARC.
				DropMidConversationSystemText,
		},
	)
	if err != nil {
		code := raylinearc.TurnNormalizationErrorCode(err)
		if code == "" {
			code = "invalid_turns"
		}
		logRaylineARCTurnRejection(err, code)
		result.PreparationFailure = "turns_" + code
		r.finalizeRaylineARCAbort(reqCtx, result.PreparationFailure)
		return result
	}
	result.Turns = turns
	return result
}

// logRaylineARCTurnRejection records why an episode could not be prepared.
// The failure class alone is a bounded metric label and cannot name the
// offending construct, so the discriminator and its request path are emitted
// here instead. Neither carries request content.
func logRaylineARCTurnRejection(err error, code string) {
	fields := map[string]interface{}{
		"outcome":       "turns_rejected",
		"failure_class": code,
	}
	if path := raylinearc.TurnNormalizationErrorPath(err); path != "" {
		fields["request_path"] = path
	}
	if detail := raylinearc.TurnNormalizationErrorDetail(err); detail != "" {
		fields["detail"] = detail
	}
	logging.ComponentErrorEvent(
		"extproc",
		"rayline_arc_turn_normalize",
		fields,
	)
}

func parseRaylineARCCloseRequest(
	closeHeader string,
	reqCtx *RequestContext,
) string {
	if closeHeader == "" {
		return ""
	}
	switch strings.TrimSpace(reqCtx.Headers[closeHeader]) {
	case "", "false":
		reqCtx.RaylineARCCloseRequested = false
		return ""
	case "true":
		reqCtx.RaylineARCCloseRequested = true
		return ""
	default:
		return "invalid_close_signal"
	}
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
	// The owning stream already holds this router open (processWithContext),
	// so the episode store cannot be closed underneath this lease.
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
		nil,
	)
	reqCtx.RaylineARCTransaction.closeRequested = reqCtx.RaylineARCCloseRequested
	reqCtx.RaylineARCTransaction.sessionCloser = r.raylineARCSessionClose
	reqCtx.RaylineARCTransaction.sessionCloseWait = time.Duration(
		arcConfig.Encoder.TotalTimeoutSeconds,
	) * time.Second
	bindRaylineARCSelectionTransaction(reqCtx)
	return state, ""
}

// projectRaylineARCTurns renders the decoded request into ARC turns.
//
// The router decodes every public wire format exactly once, so the selector
// reads the neutral messages rather than the wire body. ARC deliberately reads
// neither of the two flattened text fields the selection context also offers,
// the single query string and the prior-turn string list: the encoder was
// trained on role-tagged turns with tool calls flattened, and both of those
// fields carry a different shape.
func (r *OpenAIRouter) projectRaylineARCTurns(
	reqCtx *RequestContext,
	options raylinearc.TurnOptions,
) ([]raylinearc.Turn, error) {
	request, err := r.raylineARCConversation(reqCtx)
	if err != nil {
		return nil, err
	}
	return raylinearc.ProjectTurns(request, options)
}

// raylineARCConversation returns the whole conversation the selector must
// read, which is not always the request body.
//
// A Responses request that carries previous_response_id names its earlier
// turns instead of repeating them. The router retains those turns and replays
// them into the provider request, but only after selection has run. The
// selector would otherwise see a long session as a single question and route
// it as one, so the retained turns are prepended here.
//
// The retained turns are decoded through the same public codec as the body, so
// the projection reads one shape and not two.
func (r *OpenAIRouter) raylineARCConversation(
	reqCtx *RequestContext,
) (*llmprotocol.Request, error) {
	request := reqCtx.SemanticRequest
	state := reqCtx.ResponseObjectState
	if request == nil || state == nil ||
		len(state.ConversationHistory) == 0 ||
		// Materialization is idempotent by this flag. It is set on the
		// dispatch path, which runs after selection, so this guard only
		// matters if that order ever changes.
		state.ProviderContextApplied {
		return request, nil
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, storedHistoryFailure(err)
	}
	history, err := materializeStoredResponseHistory(
		engine,
		state.ConversationHistory,
	)
	if err != nil {
		return nil, storedHistoryFailure(err)
	}
	if len(history) == 0 {
		return request, nil
	}
	// A copy: the request the router dispatches is owned by the dispatch path,
	// and the selector must not widen it.
	merged := *request
	merged.Messages = make(
		[]llmprotocol.Message,
		0,
		len(history)+len(request.Messages),
	)
	merged.Messages = append(merged.Messages, history...)
	merged.Messages = append(merged.Messages, request.Messages...)
	return &merged, nil
}

// storedHistoryFailure fails the episode closed with a bounded class. Routing
// on the body alone would show the selector a truncated conversation, which is
// the failure this projection exists to prevent.
func storedHistoryFailure(err error) error {
	return &raylinearc.TurnNormalizationError{
		Code: "stored_history",
		Err:  err,
	}
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
