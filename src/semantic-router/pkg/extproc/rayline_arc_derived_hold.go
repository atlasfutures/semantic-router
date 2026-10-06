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
//   - the model that last served this episode, which the store keeps past
//     the episode's idle TTL (raylinearc.ServedModelStore);
//   - the model that last served its parent session, named by the trusted
//     x-rayline-parent-session header;
//   - the package's declared fallback action's model; for a package served
//     without a manifest, the fallback_model the service's package listing
//     names (when the bindings dispatch it), else the cell's configured
//     derived_hold_model.
//
// The router writes every record and the gateway owns the episode keys, so
// a client cannot choose the held model. Nothing in the request body is read.

// servedModelReadTimeout bounds the store reads a turn without episode
// memory makes; a slow store leaves the derivation to the package fallback.
const servedModelReadTimeout = 500 * time.Millisecond

// raylineARCServedModels reads the remembered serving models of an episode
// with no previous arm: its own record, then its parent session's when the
// gateway's turn-signal headers are trusted.
func (r *OpenAIRouter) raylineARCServedModels(
	reqCtx *RequestContext,
	episodeIDHash string,
	signalHeaders map[string]string,
) []selection.RaylineARCServedModel {
	store, ok := r.raylineARCEpisodeStoreFor(reqCtx).(raylinearc.ServedModelStore)
	if !ok {
		return nil
	}
	parent := reqCtx.TraceContext
	if parent == nil {
		parent = context.Background()
	}
	keys := []servedModelKey{{source: derivedHoldSessionRecord, episodeIDHash: episodeIDHash}}
	if session := strings.TrimSpace(signalHeaders[raylineARCParentSessionHeader]); session != "" {
		keys = append(keys, servedModelKey{source: derivedHoldParentSession, episodeIDHash: raylinearc.HashEpisodeID(session)})
	}
	return readServedModels(parent, store, keys)
}

// readServedModels reads each key's record, within servedModelReadTimeout
// in all: a slow or stalled store costs the turn at most that, and leaves the
// derivation to the package fallback.
// servedModelKey is one episode whose record a turn reads, and the source it
// stands for.
type servedModelKey struct {
	source        string
	episodeIDHash string
}

func readServedModels(
	parent context.Context,
	store raylinearc.ServedModelStore,
	keys []servedModelKey,
) []selection.RaylineARCServedModel {
	ctx, cancel := context.WithTimeout(parent, servedModelReadTimeout)
	defer cancel()
	var served []selection.RaylineARCServedModel
	for _, key := range keys {
		model, err := store.LastServedModel(ctx, key.episodeIDHash)
		if err != nil {
			logging.ComponentWarnEvent("extproc", "rayline_arc_served_model_read_failed", map[string]interface{}{
				"source": key.source,
			})
			continue
		}
		if model != "" {
			served = append(served, selection.RaylineARCServedModel{Source: key.source, Model: model})
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
// one the cell binds: a record naming a model this cell does not serve says
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
	for _, served := range arcContext.ServedModels {
		if model := scorer.servedModelOf(served.Model); model != "" {
			hold, ok = derivedHold{source: served.Source, model: model}, true
			break
		}
	}
	if model, source := scorer.holdFallback(); !ok && model != "" {
		hold, ok = derivedHold{source: source, model: model}, true
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
	// derivedHoldListedFallback is the fallback_model the service's package
	// listing names, for a package without a manifest (pathfinder#3677).
	derivedHoldListedFallback = "listed_fallback"
	// derivedHoldConfigured is the configured derived_hold_model, for a
	// package without a manifest.
	derivedHoldConfigured = "configured_fallback"
)

func logRaylineARCDerivedHold(arcContext *selection.RaylineARCSelectionContext, hold derivedHold) {
	logging.ComponentEvent("extproc", "rayline_arc_derived_hold", map[string]interface{}{
		"episode_id_hash": arcContext.EpisodeIDHash,
		"source":          hold.source,
		"model":           hold.model,
		"offered":         len(hold.offer),
	})
}

// markServedModel stages the model a policy-service main turn dispatches --
// the selected action's model, not its worker, which may serve several;
// the turn's episode commit records it.
func (transaction *raylineARCEpisodeTransaction) markServedModel(model string) {
	if transaction != nil {
		transaction.servedModel = model
	}
}

// servedModelOf is the model a record names, as the cell binds it: the
// model itself when an action dispatches it, or -- a record written before
// records named models -- a worker's model, mapped as armModel maps it. ""
// when the cell binds neither, so the record says nothing here.
func (scorer *policyServiceScorer) servedModelOf(recorded string) string {
	for _, actionID := range scorer.actionOrder {
		if scorer.actionModel(actionID) == recorded {
			return recorded
		}
	}
	for arm, worker := range scorer.workerIDs {
		if worker == recorded {
			return scorer.armModel(arm)
		}
	}
	return ""
}
