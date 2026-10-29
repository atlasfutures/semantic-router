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
	"encoding/json"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// raylineARCForkProbe is how long a request waits on a stream's lease before
// it reads the fork registry again. A request that resolved as a stream's
// continuation at the same moment as another can find, once the other is
// admitted, that it is a fork of it instead: it then leaves for its sibling
// rather than waiting out the other's turn.
const raylineARCForkProbe = 250 * time.Millisecond

// raylineARCForkBusyMark bounds how long an admitted turn marks its stream
// busy when its finalize never clears the mark (the replica died). A turn's
// lease is renewed for as long as it streams, which can be minutes, so the
// mark is not the lease TTL; a stale mark only makes a non-continuation a
// fork instead of a rewrite until it lapses.
const raylineARCForkBusyMark = 15 * time.Minute

// raylineARCForkMaxMoves bounds how often one request releases a lease it
// no longer belongs on and moves; each move follows another request's
// admission, so the bound is reached only under a storm of them.
const raylineARCForkMaxMoves = 3

// raylineARCForkCheck is one request's fork detection: its prefix, as the
// registry compares it, and the registry it reads.
type raylineARCForkCheck struct {
	registry    raylinearc.EpisodeForkRegistry
	rootRawID   string
	rootHash    string
	request     raylinearc.EpisodeForkRequest
	idleTTL     time.Duration
	readTimeout time.Duration
	now         func() time.Time
}

// newRaylineARCForkCheck is the request's fork check, or nil when fork
// detection does not apply: it is off, the cell is not a policy-service
// cell, the request is a side call (which takes no lease and commits
// nothing), its format is one whose transcript the body does not carry
// whole (Responses), or the store keeps no registry. Every nil answer is
// today's behaviour: the request is served on the episode its header names.
func (r *OpenAIRouter) newRaylineARCForkCheck(
	arcConfig *config.RaylineARCAlgorithmConfig,
	reqCtx *RequestContext,
	rawEpisodeID string,
	rootHash string,
	format string,
	signalHeaders map[string]string,
	callKind raylinearc.PolicyCallKind,
) *raylineARCForkCheck {
	if arcConfig == nil || arcConfig.PolicyService == nil || !arcConfig.Episode.ForkDetectionEnabled() ||
		rawEpisodeID == "" || callKind == raylinearc.PolicyCallSide ||
		(format != policyFormatAnthropic && format != policyFormatOpenAI) {
		return nil
	}
	registry, ok := r.raylineARCEpisodeStoreFor(reqCtx).(raylinearc.EpisodeForkRegistry)
	if !ok {
		return nil
	}
	var body policyClientRequest
	if err := json.Unmarshal(reqCtx.RaylineARCRawBody, &body); err != nil || len(body.Messages) == 0 {
		return nil
	}
	messages, roles, err := policyMessages(body.Messages)
	if err != nil {
		return nil
	}
	digests := raylinearc.NewPrefixDigests(messages)
	clientRequest := raylinearc.PolicyClientRequest{System: body.System, Tools: body.Tools, Messages: body.Messages}
	signals := raylineARCPolicyTurnSignals(signalHeaders, format, clientRequest, messages, rootHash)
	idle := time.Duration(arcConfig.Episode.IdleTTLSeconds) * time.Second
	if idle <= 0 {
		idle = time.Hour
	}
	return &raylineARCForkCheck{
		registry:  registry,
		rootRawID: rawEpisodeID,
		rootHash:  rootHash,
		request: raylinearc.EpisodeForkRequest{
			Digests:    digests,
			Roles:      roles,
			Root:       raylinearc.ForkRootDigest(digests, body.System, body.Tools),
			Compaction: signals.Compaction != nil,
		},
		idleTTL:     idle,
		readTimeout: time.Duration(max(arcConfig.Episode.AcquireTimeoutSeconds, 1)) * time.Second,
		now:         time.Now,
	}
}

// raylineARCForkResult is a resolved placement: the stream, the episode it
// is served on, and the registry view it was resolved from.
type raylineARCForkResult struct {
	resolution raylinearc.EpisodeForkResolution
	view       raylinearc.EpisodeForkView
	rawID      string
	hash       string
	root       string
}

// resolve reads the registry and places the request. A registry that cannot
// be read places it on the root episode, as before fork detection.
func (check *raylineARCForkCheck) resolve(ctx context.Context) (raylineARCForkResult, bool) {
	readContext, cancel := context.WithTimeout(ctx, check.readTimeout)
	defer cancel()
	view, err := check.registry.ForkView(readContext, check.rootHash)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "rayline_arc_episode_fork_unavailable", map[string]interface{}{
			"episode_id_hash": check.rootHash,
			"failure_class":   boundedARCPrepareFailure(err),
		})
		return raylineARCForkResult{}, false
	}
	resolution := raylinearc.ResolveEpisodeFork(view, check.request, check.now(), check.idleTTL)
	result := raylineARCForkResult{
		resolution: resolution, view: view, rawID: check.rootRawID, hash: check.rootHash, root: check.rootHash,
	}
	if resolution.Fork() {
		result.rawID = raylinearc.SiblingEpisodeID(check.rootRawID, resolution.Stream)
		result.hash = raylinearc.HashEpisodeID(result.rawID)
	}
	return result, true
}

func (result raylineARCForkResult) streamRecord() *raylinearc.EpisodeForkStream {
	if result.resolution.Stream == raylinearc.EpisodeForkMainStream {
		return result.view.Main
	}
	return result.view.Siblings[result.resolution.Stream]
}

// admit records the request on its stream once it holds that stream's
// lease, and marks the stream busy until the turn finalizes; a rewrite also
// records the branch it displaced. Registry writes are best effort.
func (check *raylineARCForkCheck) admit(reqCtx *RequestContext, result raylineARCForkResult) {
	now := check.now()
	token := uuid.NewString()
	record := result.streamRecord().Admitted(check.request, now, now.Add(raylineARCForkBusyMark), token)
	stream := result.resolution.Stream
	evict := result.resolution.Evict
	ctx, cancel := context.WithTimeout(context.Background(), episodeFinalizeTimeout)
	defer cancel()
	if result.resolution.Displaced != "" && result.resolution.DisplacedRecord != nil {
		check.write(ctx, result.resolution.Displaced, result.resolution.DisplacedRecord, evict)
		evict = ""
	}
	check.write(ctx, stream, record, evict)
	if transaction := reqCtx.RaylineARCTransaction; transaction != nil {
		transaction.onFinalize = chainFinalize(transaction.onFinalize, func() {
			ctx, cancel := context.WithTimeout(context.Background(), episodeFinalizeTimeout)
			defer cancel()
			// Only this admission's mark: a later turn admitted on the
			// stream once the lease was released keeps its own.
			if err := check.registry.ClearForkBusy(ctx, check.rootHash, stream, token); err != nil {
				logging.ComponentWarnEvent("extproc", "rayline_arc_episode_fork_record_failed", map[string]interface{}{
					"episode_id_hash": check.rootHash,
					"failure_class":   boundedARCPrepareFailure(err),
				})
			}
		})
	}
}

func (check *raylineARCForkCheck) write(ctx context.Context, stream string, record *raylinearc.EpisodeForkStream, evict string) {
	if err := check.registry.RecordForkStream(ctx, check.rootHash, stream, record, evict); err != nil {
		logging.ComponentWarnEvent("extproc", "rayline_arc_episode_fork_record_failed", map[string]interface{}{
			"episode_id_hash": check.rootHash,
			"failure_class":   boundedARCPrepareFailure(err),
		})
	}
}

func (result raylineARCForkResult) placement() *selection.RaylineARCForkPlacement {
	placement := &selection.RaylineARCForkPlacement{
		Fork:         result.resolution.Fork(),
		Reason:       result.resolution.Reason,
		SiblingCount: result.resolution.SiblingCount,
		New:          result.resolution.New,
	}
	if placement.Fork {
		placement.ForkOf = result.root
	}
	return placement
}

// prepareRaylineARCForkedTurn places the request among its root episode's
// streams and prepares its turn there. A continuation waits on its stream's
// lease in probes, reading the registry between them: when another request
// was admitted on the stream meanwhile and this one no longer continues it,
// it moves to the stream it now resolves to (a sibling), and so never waits
// out a parallel request's turn. Once it holds a lease it reads the registry
// once more; if it no longer belongs there it releases the lease and moves.
func (r *OpenAIRouter) prepareRaylineARCForkedTurn(
	arcConfig *config.RaylineARCAlgorithmConfig,
	reqCtx *RequestContext,
	check *raylineARCForkCheck,
	result *selection.RaylineARCSelectionContext,
	workerCount int,
) (*raylinearc.EpisodeState, *selection.SelectionResult, string) {
	parent := reqCtx.TraceContext
	if parent == nil {
		parent = context.Background()
	}
	acquire := time.Duration(arcConfig.Episode.AcquireTimeoutSeconds) * time.Second
	deadline := check.now().Add(acquire)
	defer func() { reqCtx.RaylineARCAcquireTimeout = 0 }()
	moves := 0
	for {
		placed, ok := check.resolve(parent)
		if !ok {
			reqCtx.RaylineARCAcquireTimeout = 0
			return r.prepareOrJoinRaylineARCTurn(arcConfig, reqCtx, check.rootHash, workerCount, false)
		}
		remaining := deadline.Sub(check.now())
		probe := raylineARCForkProbe
		if remaining < probe {
			probe = remaining
		}
		lastAttempt := remaining <= raylineARCForkProbe || arcConfig.Episode.RelaxedConsistency()
		if probe <= 0 {
			probe = time.Millisecond
		}
		reqCtx.RaylineARCAcquireTimeout = probe
		state, coalesced, failure := r.prepareOrJoinRaylineARCTurn(arcConfig, reqCtx, placed.hash, workerCount, false)
		if failure == "episode_timeout" && !lastAttempt {
			continue
		}
		if failure != "" || coalesced != nil {
			r.applyRaylineARCForkPlacement(result, placed)
			return state, coalesced, failure
		}
		// The registry is read again under the lease, so the admission is
		// written over what the stream holds now.
		if again, ok := check.resolve(parent); ok {
			if again.hash != placed.hash && moves < raylineARCForkMaxMoves {
				// Admitted elsewhere while this one waited: the episode this
				// lease holds is no longer its stream.
				moves++
				r.finalizeRaylineARCAbort(reqCtx, "episode_fork_moved")
				reqCtx.RaylineARCTransaction, reqCtx.SelectionTransaction, reqCtx.RaylineARCInflight = nil, nil, nil
				continue
			}
			if again.hash == placed.hash {
				placed = again
			} else {
				// Out of moves: it stays where its lease is, admitted over
				// that stream's current record rather than the stale one.
				placed.view = again.view
			}
		}
		check.admit(reqCtx, placed)
		r.applyRaylineARCForkPlacement(result, placed)
		return state, nil, ""
	}
}

func (r *OpenAIRouter) applyRaylineARCForkPlacement(
	result *selection.RaylineARCSelectionContext,
	placed raylineARCForkResult,
) {
	result.EpisodeIDHash = placed.hash
	placement := placed.placement()
	result.Fork = placement
	logging.ComponentEvent("extproc", "rayline_arc_episode_fork", map[string]interface{}{
		"request_id":      result.RequestID,
		"episode_id_hash": placed.hash,
		"fork":            placement.Fork,
		"fork_of":         placement.ForkOf,
		"fork_reason":     placement.Reason,
		"fork_new":        placement.New,
		"sibling_count":   placement.SiblingCount,
		"fork_parent":     placed.resolution.Parent != raylinearc.EpisodeForkMainStream,
	})
}
