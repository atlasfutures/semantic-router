package extproc

import (
	"context"
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A conversation can reach a two-stage policy package with no memory of the
// model serving it: its history holds assistant messages the episode never
// attributed, and the episode itself has cooled or been evicted -- a
// conversation resumed after a pause, or a subagent handed its parent's
// history. VSR keeps one memory that outlives the episode: the model this
// conversation's own decisions last served, which the store holds past the
// episode's idle TTL (raylinearc.ServedModelStore). It continues a decision
// the policy already made for this conversation.
//
// VSR reads that record before it asks, and when the package's manifest
// says it is two-stage the first ask carries it: attested
// (selection.attested_held_action_id, memex-desktop #7417) with every action
// still offered, so a stage one that reads the field holds that model or
// decides a typed user turn against it; or, for a stage one that is not
// attestable, narrowed to the model's actions, the caller hold the contract
// describes (pathfinder arc_serving_contract.md, "Stage one"). With no
// record the first ask goes out plain and the policy decides the turn at a
// start (pathfinder ADR 0149).
//
// The narrowed re-ask remains for the two packages the first ask cannot
// serve: one that predates the attested field and refuses the attested ask
// (stage_one_held_unknown), and one VSR does not know to be two-stage, which
// keeps the record for that refusal. A refusal that stands fails
// (rayline_arc_undecidable).
//
// Nothing else names a model. A parent session's model does not (ADR 0117:
// the parent link is metadata and never holds a subagent's actions), and
// neither does a package, listed or configured fallback model, which the
// policy never chose.
//
// The router writes every record and the gateway owns the episode keys, so
// a client cannot choose the held model. Nothing in the request body is read.

// servedModelReadTimeout bounds the store reads a turn without episode
// memory makes; a slow store leaves the turn without a record, so it fails as
// undecidable.
const servedModelReadTimeout = 500 * time.Millisecond

// raylineARCServedModels reads the remembered serving model of an episode
// with no previous arm: its own record only.
func (r *OpenAIRouter) raylineARCServedModels(
	reqCtx *RequestContext,
	episodeIDHash string,
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
	return readServedModels(parent, store, keys)
}

// readServedModels reads each key's record, within servedModelReadTimeout
// in all: a slow or stalled store costs the turn at most that, and leaves the
// turn without a record.
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
// record names a model: the turn is then undecidable.
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

// derivedHoldSessionRecord is the one source a derived hold names, as
// rayline_arc_derived_hold logs it, so a readout can tell a derived hold from
// a schedule's: the model this episode's own policy decisions served.
const derivedHoldSessionRecord = "session_record"

// logRaylineARCDerivedHold records the record a turn is held on and which
// ask carries it: "first" (attested or narrowed before the service is
// asked), "re-ask" (narrowed after an attested first ask was refused), or
// "none" (the recorded model is excluded this turn, so no ask carries it).
func logRaylineARCDerivedHold(arcContext *selection.RaylineARCSelectionContext, hold derivedHold, ask string) {
	logging.ComponentEvent("extproc", "rayline_arc_derived_hold", map[string]interface{}{
		"episode_id_hash": arcContext.EpisodeIDHash,
		"source":          hold.source,
		"model":           hold.model,
		"offered":         len(hold.offer),
		"ask":             ask,
	})
}

// logRaylineARCAttestedHold records a turn the service decided or held with the
// served model attested (memex-desktop #7417): every action stayed offered,
// so a typed user turn after a lost replica is decided, not held by VSR.
// served is the action the service then chose: another model's when a typed
// user turn switched, the attested model's when the turn was held.
func logRaylineARCAttestedHold(arcContext *selection.RaylineARCSelectionContext, hold derivedHold, attested, served string) {
	logging.ComponentEvent("extproc", "rayline_arc_attested_hold", map[string]interface{}{
		"episode_id_hash": arcContext.EpisodeIDHash,
		"source":          hold.source,
		"model":           hold.model,
		"attested_action": attested,
		"served_action":   served,
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
			// A snapshot's package may bind no action on a worker the cell
			// keeps for another snapshot: its record says nothing here.
			if scorer.snapshot != "" && !scorer.armBound(arm) {
				return ""
			}
			return scorer.armModel(arm)
		}
	}
	return ""
}

// policyServiceAnswered is whether err is an answer the policy service gave
// (a refusal, a rejection, a backend failure it reported), as against a call
// it never answered: the client's timeout and transport classes, and a
// request it could not build. Only an answered attested ask is asked again
// narrowed; a lost one is the turn's failure as it stands.
func policyServiceAnswered(err error) bool {
	var failure *raylinearc.PolicyServiceError
	if !errors.As(err, &failure) {
		return false
	}
	switch failure.Class {
	case "timeout", "transport", "request":
		return false
	}
	return true
}

// arcFailurePolicyUndecidable is the selection failure of a turn the policy
// would not decide and no record of this conversation's own decisions could
// hold: the router serves nothing in its place. It answers 422 with the
// public class "undecidable" (rayline_arc_selection_response.go).
const arcFailurePolicyUndecidable = "policy_undecidable"

// logRaylineARCUndecidable records why a turn failed undecided: the policy's
// refusal reason, the record a hold was retried on (empty when none named a
// model), and whether a model was already held when the policy refused.
func logRaylineARCUndecidable(
	arcContext *selection.RaylineARCSelectionContext,
	reason string,
	derivedSource string,
	held bool,
	sideCall bool,
) {
	logging.ComponentErrorEvent("extproc", "rayline_arc_undecidable", map[string]interface{}{
		"request_id":      arcContext.RequestID,
		"episode_id_hash": arcContext.EpisodeIDHash,
		"reason":          reason,
		"derived_source":  derivedSource,
		"held":            held,
		"side_call":       sideCall,
	})
}
