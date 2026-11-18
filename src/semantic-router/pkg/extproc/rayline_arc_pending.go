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
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A tracked route lookup on a policy-service cell does not commit a turn: the
// caller runs the model out of the cell's sight, may ask again for the same
// turn, or may not run it at all. Its decision is stored as the episode's
// pending decision instead, and the conversation's next request resolves it
// before anything reads the episode: the next turn records it as a turn that
// ran (inferred), and any other request drops it (raylinearc.ResolvePending).

// resolveRaylineARCPendingDecision resolves the pending decision the
// prepared episode holds against this request, so the selection, the holds
// read before it, and the turn's own commit all see the resolved episode.
// It also marks a route lookup's transaction as one that stores its decision
// pending. Only a policy-service cell calls it. A side call or a coalesced
// resend resolves nothing: neither commits, and the next main turn resolves
// it. It returns the pending decision the request dropped unexecuted, nil
// when it dropped none.
func resolveRaylineARCPendingDecision(reqCtx *RequestContext, arcContext *selection.RaylineARCSelectionContext) *raylinearc.PendingDecision {
	transaction := reqCtx.RaylineARCTransaction
	if transaction == nil || transaction.sideCall || transaction.leaseless ||
		arcContext.PolicyCallKind == raylinearc.PolicyCallSide {
		return nil
	}
	transaction.decidesPending = reqCtx.RaylineRouteSurface == string(routerruntime.RouteDecisionSurfaceRoutes)
	if transaction.borrowed || arcContext.Coalesced != nil {
		// A resend that joined an identical lookup in flight shares its
		// decision and the episode it was decided on (already resolved), and
		// reports the same pending turn; it writes nothing.
		return nil
	}
	state := arcContext.State
	if !transaction.decidesPending && (state == nil || state.Pending == nil) {
		return nil
	}
	_, messages, roles, err := policyClientRequestOf(arcContext)
	if err != nil {
		// The selection refuses the same body; there is nothing to resolve
		// against.
		return nil
	}
	digests := raylinearc.NewPrefixDigests(messages)
	transaction.requestDigests = digests
	if state == nil || state.Pending == nil {
		return nil
	}
	resolved, resolution := raylinearc.ResolvePending(state, digests, roles)
	arcContext.State, transaction.state = resolved, resolved
	transaction.resolvedPending, transaction.readTurnIndex = true, state.TurnIndex
	logging.ComponentEvent("extproc", "rayline_arc_pending_decision", map[string]interface{}{
		"episode_id_hash": arcContext.EpisodeIDHash,
		"request_id":      reqCtx.RequestID,
		"route_id":        state.Pending.RouteID,
		"resolution":      resolution,
		"turn_index":      arcTurnIndexForTelemetry(state.TurnIndex),
	})
	if resolution == raylinearc.PendingInferred {
		return nil
	}
	return state.Pending
}

// raylineARCServedModelsFor is the served-model record a turn without episode
// memory is held to (rayline_arc_derived_hold.go). A tracked lookup writes its
// decided model there, so after a request dropped that decision unexecuted
// the record names a model that did not run: the record as the dropped
// decision read it is used instead. A lookup keeps what it used for the
// decision it stores, so repeated asks keep the record from before them.
func (r *OpenAIRouter) raylineARCServedModelsFor(
	reqCtx *RequestContext,
	arcContext *selection.RaylineARCSelectionContext,
	dropped *raylinearc.PendingDecision,
) []selection.RaylineARCServedModel {
	var served []selection.RaylineARCServedModel
	switch {
	case dropped == nil:
		served = r.raylineARCServedModels(reqCtx, arcContext.EpisodeIDHash)
	case dropped.PriorServedModel != "":
		served = []selection.RaylineARCServedModel{{Source: derivedHoldSessionRecord, Model: dropped.PriorServedModel}}
	}
	if transaction := reqCtx.RaylineARCTransaction; transaction != nil && len(served) > 0 {
		transaction.priorServedModel = served[0].Model
	}
	return served
}

// commitPending stores the lookup's decision as the episode's pending
// decision and releases the lease. The trajectory -- turn count, previous
// arm, ledger, warmth, boundary -- is written as the request resolved it;
// only what later requests are routed through moves with the decision: the
// encoder owner, the policy session and the served-model record.
func (transaction *raylineARCEpisodeTransaction) commitPending(
	ctx context.Context,
	requestContext *RequestContext,
) error {
	pending := func() (*raylinearc.EpisodeState, error) {
		return transaction.pendingState(requestContext.RequestID)
	}
	transaction.finalizeOnce.Do(func() {
		defer transaction.releaseHold()
		if transaction.borrowed {
			metrics.RecordRaylineARCEpisodeTransaction("coalesced", "")
			return
		}
		// The selection can still read the lookup as a side call, which
		// writes nothing.
		if transaction.sideCall {
			transaction.commitSideCall(ctx)
			return
		}
		if transaction.relaxed {
			transaction.commitRelaxed(ctx, requestContext, pending, "relaxed_pending")
			return
		}
		transaction.stopRenewal()
		// The service staged this lookup's turn for an append that has no
		// reply to append; the next turn is sent in full, as after an abort.
		transaction.dropPolicyStagedTurn(ctx)
		if !transaction.selectionReady || transaction.leaseLost.Load() {
			transaction.finalizeErr = ErrRaylineARCEpisodeLeaseLost
			transaction.abortStore(ctx)
			return
		}
		next, err := pending()
		if err != nil {
			transaction.finalizeErr = err
			transaction.abortStore(ctx)
			return
		}
		transaction.finalizeErr = transaction.store.Commit(ctx, transaction.lease, transaction.lease.Version(), next)
		if transaction.finalizeErr != nil {
			transaction.abortStore(ctx)
			return
		}
		transaction.state = next
		transaction.endReplacedPolicySession(ctx)
		metrics.RecordRaylineARCEpisodeTransaction("commit", "pending")
		recordCommittedARCEpisodeTelemetry(requestContext, transaction)
	})
	return transaction.finalizeErr
}

// pendingState is the episode with this lookup's decision as its pending
// decision: the prepared (resolved) episode, the routing fields the
// selection moved, and no boundary decision, so a repeat ask decides again
// rather than replaying this one.
func (transaction *raylineARCEpisodeTransaction) pendingState(routeID string) (*raylinearc.EpisodeState, error) {
	if transaction.state == nil || transaction.requestDigests == nil {
		return nil, errors.New("a pending decision needs its request's episode and messages")
	}
	arm := transaction.selectedArm
	if arm < 0 || arm >= len(transaction.state.Warmth) {
		return nil, errors.New("selected arm is out of range")
	}
	next := cloneARCState(transaction.state)
	transaction.applyRoutingState(next)
	next.PolicyBoundary = nil
	next.Pending = raylinearc.NewPendingDecision(
		routeID, arm, transaction.servedModel, next.TurnIndex, transaction.serializedTokens,
		transaction.requestDigests, transaction.policyNext, time.Now(),
	)
	next.Pending.PriorServedModel = transaction.priorServedModel
	// The served-model record names the decided model, as a committed
	// lookup's did: a conversation that comes back after its episode expired,
	// with nothing reported, is held to the last decision, assumed to have
	// run. A repeat ask overwrites it with the newest decision. A decision
	// that names no model leaves the record as it was.
	next.ServedModel = transaction.servedModel
	return next, nil
}
