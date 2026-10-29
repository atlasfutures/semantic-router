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

// raylineARCForkMaxMoves bounds how often one request releases a lease it
// no longer belongs on and moves; each move follows another request's
// admission, so the bound is reached only under a storm of them.
const raylineARCForkMaxMoves = 3

// raylineARCForkCheck is one request's fork detection: its prefix, as the
// registry compares it, and the registry it reads.
type raylineARCForkCheck struct {
	registry    raylinearc.EpisodeForkRegistry
	snapshots   raylinearc.EpisodeSnapshotStore
	workerCount int
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
	workerCount int,
) *raylineARCForkCheck {
	if arcConfig == nil || arcConfig.PolicyService == nil || !arcConfig.Episode.ForkDetectionEnabled() ||
		rawEpisodeID == "" || callKind == raylinearc.PolicyCallSide ||
		(format != policyFormatAnthropic && format != policyFormatOpenAI) {
		return nil
	}
	store := r.raylineARCEpisodeStoreFor(reqCtx)
	registry, ok := store.(raylinearc.EpisodeForkRegistry)
	if !ok {
		return nil
	}
	snapshots, _ := store.(raylinearc.EpisodeSnapshotStore)
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
		registry:    registry,
		snapshots:   snapshots,
		workerCount: max(workerCount, 1),
		rootRawID:   rawEpisodeID,
		rootHash:    rootHash,
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
//
// held is the episode whose lease this request already holds, or "": its
// own lease does not make its stream busy to it.
func (check *raylineARCForkCheck) resolve(ctx context.Context, held string) (raylineARCForkResult, bool) {
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
	if held != "" {
		for _, stream := range append([]*raylinearc.EpisodeForkStream{view.Main}, mapValues(view.Siblings)...) {
			if stream != nil && stream.EpisodeHash == held {
				stream.Leased = false
			}
		}
	}
	resolution := raylinearc.ResolveEpisodeFork(view, check.request, check.now(), check.idleTTL)
	if resolution.Reason == raylinearc.ForkReasonRetryEdit && check.latestCommitted(ctx, view, resolution.Stream) {
		// The turn it would retry committed: the request is a branch off it.
		request := check.request
		request.Committed = map[string]bool{resolution.Stream: true}
		resolution = raylinearc.ResolveEpisodeFork(view, request, check.now(), check.idleTTL)
	}
	result := raylineARCForkResult{
		resolution: resolution, view: view, rawID: check.rootRawID, hash: check.rootHash, root: check.rootHash,
	}
	if resolution.Fork() {
		result.rawID = raylinearc.SiblingEpisodeID(check.rootRawID, resolution.Stream)
		result.hash = raylinearc.HashEpisodeID(result.rawID)
	}
	return result, true
}

// latestCommitted reports whether stream's newest admitted turn committed:
// its episode's recorded prefix is that turn's. A state that cannot be read
// counts as not committed, which keeps the request on the stream.
func (check *raylineARCForkCheck) latestCommitted(ctx context.Context, view raylinearc.EpisodeForkView, stream string) bool {
	record := view.Main
	if stream != raylinearc.EpisodeForkMainStream {
		record = view.Siblings[stream]
	}
	latest, ok := record.Latest()
	if !ok || check.snapshots == nil {
		return false
	}
	readContext, cancel := context.WithTimeout(ctx, check.readTimeout)
	defer cancel()
	state, _, err := check.snapshots.Snapshot(readContext, record.EpisodeHash, check.workerCount)
	if err != nil || state == nil || state.Policy == nil {
		return false
	}
	return state.Policy.PrefixLen == latest.Len && state.Policy.PrefixDigest == latest.Digest
}

func (result raylineARCForkResult) streamRecord() *raylinearc.EpisodeForkStream {
	if result.resolution.Stream == raylinearc.EpisodeForkMainStream {
		return result.view.Main
	}
	return result.view.Siblings[result.resolution.Stream]
}

// admit records the request on its stream once it holds that stream's
// lease; a rewrite also records the branch it displaced. A stream reads as
// busy from its lease alone, so admission marks nothing that must be
// cleared. Registry writes are best effort.
func (check *raylineARCForkCheck) admit(result raylineARCForkResult) {
	now := check.now()
	record := result.streamRecord().Admitted(check.request, now, result.hash)
	evict := result.resolution.Evict
	ctx, cancel := context.WithTimeout(context.Background(), episodeFinalizeTimeout)
	defer cancel()
	if displaced := result.resolution.Displaced; displaced != "" && result.resolution.DisplacedRecord != nil {
		dormant := result.resolution.DisplacedRecord.Clone()
		dormant.EpisodeHash = raylinearc.HashEpisodeID(raylinearc.SiblingEpisodeID(check.rootRawID, displaced))
		check.write(ctx, displaced, dormant, evict)
		evict = ""
	}
	check.write(ctx, result.resolution.Stream, record, evict)
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
		placed, ok := check.resolve(parent, "")
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
		if again, ok := check.resolve(parent, placed.hash); ok {
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
		check.admit(placed)
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

func mapValues(streams map[string]*raylinearc.EpisodeForkStream) []*raylinearc.EpisodeForkStream {
	values := make([]*raylinearc.EpisodeForkStream, 0, len(streams))
	for _, stream := range streams {
		values = append(values, stream)
	}
	return values
}
