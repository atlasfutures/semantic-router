package extproc

import (
	"context"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A two-stage policy package refuses a mid-conversation turn it would have
// to decide cold (selection_refused, stage_one_held_unknown): one whose
// history holds an assistant message the episode never attributed, with no
// episode memory -- a conversation resumed after its episode cooled or was
// evicted, or a subagent handed its parent's history. The contract's answer
// is a caller hold: an offer whose every action belongs to one model, the
// way the schedule holds one (pathfinder arc_serving_contract.md, "Stage
// one"). On that refusal, and only then, the turn is asked once more with
// the offer narrowed to a derived model. Single-stage packages never refuse
// this way, so their decisions are untouched.
//
// The model is derived from, in order:
//   - the worker that last served this episode, which the store keeps past
//     the episode's idle TTL (raylinearc.ServedWorkerStore);
//   - the worker that last served its parent session, named by the trusted
//     x-rayline-parent-session header;
//   - the package's declared fallback action's model.
//
// The router writes every record and the gateway owns the episode keys, so
// a client cannot choose the held model. Nothing in the request body is read.

// servedWorkerReadTimeout bounds the store reads a turn without episode
// memory makes; a slow store leaves the derivation to the package fallback.
const servedWorkerReadTimeout = 500 * time.Millisecond

// raylineARCServedWorkers reads the remembered serving workers of an episode
// with no previous arm: its own record, then its parent session's when the
// gateway's turn-signal headers are trusted.
func (r *OpenAIRouter) raylineARCServedWorkers(
	reqCtx *RequestContext,
	episodeIDHash string,
	signalHeaders map[string]string,
) []selection.RaylineARCServedWorker {
	store, ok := r.raylineARCEpisodeStoreFor(reqCtx).(raylinearc.ServedWorkerStore)
	if !ok {
		return nil
	}
	parent := reqCtx.TraceContext
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithTimeout(parent, servedWorkerReadTimeout)
	defer cancel()
	keys := []selection.RaylineARCServedWorker{{Source: derivedHoldSessionRecord, Worker: episodeIDHash}}
	if session := strings.TrimSpace(signalHeaders[raylineARCParentSessionHeader]); session != "" {
		keys = append(keys, selection.RaylineARCServedWorker{
			Source: derivedHoldParentSession, Worker: raylinearc.HashEpisodeID(session),
		})
	}
	var served []selection.RaylineARCServedWorker
	for _, key := range keys {
		worker, err := store.LastServedWorker(ctx, key.Worker)
		if err != nil {
			logging.ComponentWarnEvent("extproc", "rayline_arc_served_worker_read_failed", map[string]interface{}{
				"source": key.Source,
			})
			continue
		}
		if worker != "" {
			served = append(served, selection.RaylineARCServedWorker{Source: key.Source, Worker: worker})
		}
	}
	return served
}

// derivedHold is the model a stage_one_held_unknown refusal is answered
// with, and the offer narrowed to its actions.
type derivedHold struct {
	source string
	model  string
	// offer is every offered action of model: all its levels, so stage two
	// still chooses one. Empty when this turn excludes the model.
	offer []string
}

// derivedHoldOffer derives the held model from the first source that names
// one the cell binds: a record naming a worker this cell does not serve says
// nothing and is passed over. The hold is that model whatever the turn
// excludes: a held model this turn excludes (vision, capability, an
// operator's disable, a fallback exclusion) is not swapped for another, so
// its offer is empty and the turn fails rather than retrying
// (pathfinder arc_serving_contract.md, exclusions). ok is false when no
// source names a model.
func derivedHoldOffer(
	scorer *policyServiceScorer,
	arcContext *selection.RaylineARCSelectionContext,
	available []string,
) (hold derivedHold, ok bool) {
	for _, served := range arcContext.ServedWorkers {
		for arm, worker := range scorer.workerIDs {
			if worker == served.Worker && scorer.armModel(arm) != "" {
				hold, ok = derivedHold{source: served.Source, model: scorer.armModel(arm)}, true
				break
			}
		}
		if ok {
			break
		}
	}
	if !ok && scorer.fallbackActionID != "" {
		if model := scorer.actionModel(scorer.fallbackActionID); model != "" {
			hold, ok = derivedHold{source: derivedHoldPackageFallback, model: model}, true
		}
	}
	if !ok {
		return derivedHold{}, false
	}
	for _, actionID := range available {
		if scorer.actionModel(actionID) == hold.model {
			hold.offer = append(hold.offer, actionID)
		}
	}
	return hold, true
}

// The sources a derived hold names, as rayline_arc_derived_hold logs them, so
// a readout can tell a derived hold from a schedule's.
const (
	derivedHoldSessionRecord   = "session_record"
	derivedHoldParentSession   = "parent_session"
	derivedHoldPackageFallback = "package_fallback"
)

func logRaylineARCDerivedHold(arcContext *selection.RaylineARCSelectionContext, hold derivedHold) {
	logging.ComponentEvent("extproc", "rayline_arc_derived_hold", map[string]interface{}{
		"episode_id_hash": arcContext.EpisodeIDHash,
		"source":          hold.source,
		"model":           hold.model,
		"offered":         len(hold.offer),
	})
}

// markServedWorker stages the worker a policy-service main turn dispatches
// to, recorded once the turn commits.
func (transaction *raylineARCEpisodeTransaction) markServedWorker(worker string) {
	if transaction != nil {
		transaction.servedWorker = worker
	}
}

// recordServedWorker remembers the committed turn's worker past the episode
// (raylinearc.ServedWorkerStore). It is best effort: the turn is served
// either way, and a missing record only leaves a later refusal to the
// package fallback.
func (transaction *raylineARCEpisodeTransaction) recordServedWorker(ctx context.Context) {
	if transaction == nil || transaction.servedWorker == "" || transaction.episodeIDHash == "" ||
		transaction.servedStamp.IsZero() {
		return
	}
	var store raylinearc.ServedWorkerStore
	if transaction.store != nil {
		store, _ = transaction.store.(raylinearc.ServedWorkerStore)
	}
	if store == nil && transaction.snapshots != nil {
		store, _ = transaction.snapshots.(raylinearc.ServedWorkerStore)
	}
	if store == nil {
		return
	}
	if err := store.RecordServedWorker(ctx, transaction.episodeIDHash, transaction.servedWorker, transaction.servedStamp); err != nil {
		logging.ComponentWarnEvent("extproc", "rayline_arc_served_worker_record_failed", map[string]interface{}{})
	}
}
