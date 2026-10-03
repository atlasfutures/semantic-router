package extproc

import (
	"context"
	"errors"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// selectionOutcomeRefusal is the outcome of a turn the provider refused.
const selectionOutcomeRefusal = "refusal"

// responseRefused reports a model refusal: a content_filter stop, or refusal
// content whatever stop its source format gave it.
func responseRefused(response *llmprotocol.Response) bool {
	if response == nil {
		return false
	}
	if response.StopReason == llmprotocol.StopContentFilter {
		return true
	}
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentRefusal {
				return true
			}
		}
	}
	return false
}

// declineRefusedTurn ends a refused policy-service turn without recording it.
//
// Collection never continued past a refusal: a refused query ended with no
// delivered turn (pathfinder end_task_query_refused, operator rulings
// 2026-09-23/24). Serving does the same, so a refused turn writes no ledger
// entry, completes no turn and leaves the held arm as it was; the outcome
// class makes the process-terminal finalizer abort it as a refusal. A
// retained boundary decision that chose the refused arm is cleared, so the
// client's retry decides again instead of reusing the arm that refused
// (pathfinder docs/arc_fallback_design.md, section 5.1).
func declineRefusedTurn(ctx *RequestContext) {
	if ctx == nil {
		return
	}
	ctx.SelectionSettlement.OutcomeClass = selectionOutcomeRefusal
	if ctx.RaylineARCTransaction == nil || ctx.VSRRaylineARC == nil {
		return
	}
	refusal := refusedTurn{arm: ctx.VSRRaylineARC.SelectedArm}
	// With ADR 0120's fallback on, the refusing model is excluded for the
	// rest of this context, so the next turn decides among the others.
	if decision := ctx.VSRSelectedDecision; decision != nil && decision.Algorithm != nil &&
		decision.Algorithm.RaylineARC != nil && decision.Algorithm.RaylineARC.PolicyService.FallbackEnabled() {
		refusal.exclude = refusedModel(ctx)
		refusal.context = ctx.VSRRaylineARC.PolicyTurnState
		refusal.decidedFrom = ctx.RaylineARCTransaction.storedPolicy()
		refusal.committed = ctx.VSRRaylineARC.PolicyNextState
	}
	clearContext, cancel := context.WithTimeout(context.Background(), episodeFinalizeTimeout)
	defer cancel()
	ctx.RaylineARCTransaction.stageRefusal(clearContext, refusal)
}

// refusedModel is the model that refused: the action's trained model, or
// its worker's.
func refusedModel(ctx *RequestContext) string {
	if model := ctx.VSRRaylineARC.PolicyActionModel; model != "" {
		return model
	}
	return ctx.VSRSelectedModel
}

// refusedTurn is what a refusal changes in its episode: the retained
// boundary decision on the refused arm is cleared, and with the fallback on
// the refusing model is excluded.
type refusedTurn struct {
	arm     int
	exclude string
	// boundary is the exact decision a coalesced resend was dispatched under;
	// nil for the request that holds the lease, which clears on the arm.
	boundary *raylinearc.PolicyBoundaryDecision
	// context is the policy state the refused turn was decided in, and
	// decidedFrom the stored one it was derived from. The exclusion is
	// recorded in context while the store still holds decidedFrom: a
	// refused compaction turn then excludes in the compacted context its
	// retry continues, not in the one it left.
	context     *raylinearc.PolicyEpisodeState
	decidedFrom *raylinearc.PolicyEpisodeState
	// committed is the state this turn would commit had it been answered: a
	// coalesced copy of it that was answered commits exactly this prefix.
	committed *raylinearc.PolicyEpisodeState
}

// storedPolicy is the policy state the transaction read.
func (transaction *raylineARCEpisodeTransaction) storedPolicy() *raylinearc.PolicyEpisodeState {
	if transaction == nil || transaction.state == nil {
		return nil
	}
	return transaction.state.Policy
}

// samePolicyContext reports whether two stored policy states are the same
// context at the same prefix.
func samePolicyContext(a, b *raylinearc.PolicyEpisodeState) bool {
	if a == nil || b == nil {
		return a == b
	}
	// The ledger tells a committed turn from the state it read even when the
	// turn's request did not grow the prefix (it is bounded, so compared whole).
	return a.Epoch == b.Epoch && a.CompactionCount == b.CompactionCount &&
		a.PrefixLen == b.PrefixLen && a.PrefixDigest == b.PrefixDigest && slices.Equal(a.Ledger, b.Ledger)
}

// apply is state with the refusal's changes, and whether it changed.
func (refusal refusedTurn) apply(state *raylinearc.EpisodeState) (*raylinearc.EpisodeState, bool) {
	if state == nil {
		return nil, false
	}
	next := cloneARCState(state)
	changed := false
	if boundary := next.PolicyBoundary; boundary != nil && boundary.Arm == refusal.arm &&
		(refusal.boundary == nil || *boundary == *refusal.boundary) {
		next.PolicyBoundary = nil
		changed = true
	}
	if refusal.exclude != "" && !next.Policy.Excludes(refusal.exclude) {
		base, ok := refusal.exclusionBase(next.Policy)
		if !ok {
			return next, changed
		}
		if policy := base.WithExclusion(refusal.exclude, turnFailureRefusal); policy.Excludes(refusal.exclude) {
			next.Policy, changed = policy, true
		}
	}
	return next, changed
}

// exclusionBase is the policy state a refusal's exclusion is added to, given
// what the store holds now, and false when the refusal no longer belongs:
//   - the store still holds what the turn read: the turn's own context,
//     keeping what the store excluded since, so a late refusal never lifts a
//     newer exclusion;
//   - the store holds this very turn, committed by a coalesced copy that was
//     answered (the same epoch, compaction and conversation, by digest): what
//     it holds now;
//   - anything else: nothing. The store may hold another context (a
//     compaction or a prefix break since), or a sibling one a concurrent
//     relaxed request opened with the same epoch number, and a context starts
//     with the full offer. It may also hold later turns of this context, after
//     an answered copy of the refused turn committed and the client went on;
//     the model answered that very request, so its duplicate's refusal is
//     dropped rather than matched to a context it cannot be proven part of.
func (refusal refusedTurn) exclusionBase(stored *raylinearc.PolicyEpisodeState) (*raylinearc.PolicyEpisodeState, bool) {
	if refusal.context == nil {
		return stored, true
	}
	if samePolicyContext(stored, refusal.decidedFrom) {
		base := refusal.context
		// Only a turn that stayed in the stored context keeps what it
		// excluded; a turn that opened a new one starts it with the full offer.
		if stored != nil && stored.Epoch == refusal.context.Epoch {
			for _, exclusion := range stored.Exclusions {
				base = base.WithExclusion(exclusion.Model, exclusion.Class)
			}
		}
		return base, true
	}
	if committed := refusal.committed; stored != nil && committed != nil && samePolicyContext(stored, committed) &&
		stored.EpochStartTurn == committed.EpochStartTurn && stored.CompactionSummary == committed.CompactionSummary {
		return stored, true
	}
	return nil, false
}

// stageRefusal stores the refusal's changes, leaving the rest of the
// episode as stored. Best effort, like retaining a boundary: a failure leaves
// the retry to reuse the decision and the model offered, as before.
func (transaction *raylineARCEpisodeTransaction) stageRefusal(ctx context.Context, refusal refusedTurn) {
	if transaction == nil || transaction.state == nil || transaction.sideCall || transaction.stateless {
		return
	}
	// A resend that borrowed the lease, or a turn whose lease is already
	// given back, stages through a short lease of its own.
	if transaction.borrowed || transaction.leaseReleased.Load() {
		if transaction.state.PolicyBoundary != nil {
			boundary := *transaction.state.PolicyBoundary
			refusal.boundary = &boundary
		}
		transaction.stageBorrowedRefusal(ctx, refusal)
		return
	}
	next, changed := refusal.apply(transaction.state)
	if !changed {
		return
	}
	var err error
	if transaction.relaxed {
		// A relaxed turn never waits on episode state: the same short bound
		// as staging the boundary.
		relaxedContext, cancel := context.WithTimeout(ctx, relaxedBoundaryStageTimeout)
		defer cancel()
		err = transaction.snapshots.CommitIfUnchanged(relaxedContext, transaction.episodeIDHash, transaction.read, next)
	} else if stager, ok := transaction.store.(raylinearc.EpisodeStateStager); ok {
		err = stager.Stage(ctx, transaction.lease, next)
	} else {
		return
	}
	if err != nil {
		if !errors.Is(err, raylinearc.ErrEpisodeLeaseLost) && !errors.Is(err, raylinearc.ErrEpisodeConflict) {
			logging.ComponentWarnEvent("extproc", "rayline_arc_refused_boundary_clear_failed", map[string]interface{}{
				"failure_class": boundedARCEpisodeFailure(err),
			})
		}
		return
	}
	transaction.state = next
	metrics.RecordRaylineARCEpisodeTransaction("boundary_cleared", selectionOutcomeRefusal)
}

// stageBorrowedRefusal stores a refusal for a coalesced resend. The resend
// holds no lease: the request that decided owns it, and if that request
// failed it aborted and kept the decision, so the only refusal would
// otherwise leave the retry on the refusing arm and model. The resend takes
// the lease briefly itself and applies the refusal to the episode as it now
// stands: the boundary is cleared only if it is still the exact decision the
// resend was dispatched under, and the exclusion is added. A lease still held
// means the deciding request is running: the refusal is handed to it.
func (transaction *raylineARCEpisodeTransaction) stageBorrowedRefusal(parent context.Context, refusal refusedTurn) {
	stager, ok := transaction.store.(raylinearc.EpisodeStateStager)
	if !ok || transaction.store == nil {
		return
	}
	if refusal.boundary == nil {
		// No decision was retained for this resend: none is cleared.
		refusal.arm = -1
	}
	ctx, cancel := context.WithTimeout(parent, relaxedBoundaryStageTimeout)
	defer cancel()
	// stageContext is the context the lease was taken under; the refusal is
	// staged under the same one.
	stageContext := ctx
	lease, current, err := transaction.store.Prepare(ctx, transaction.episodeIDHash, len(transaction.state.Warmth))
	if err != nil {
		// The deciding request still holds the lease: hand the refusal to
		// it, so it applies it as it aborts. If it has already taken its last
		// look, the hand-over is refused and the refusal is retried here,
		// once the lease is released.
		if transaction.inflight == nil || transaction.inflight.noteRefusal(refusal) {
			return
		}
		// The leader has taken its last look and is finishing, which can
		// outlast a short wait (an encoder close, a slow commit): wait for it
		// to release the lease, within the refusal's own bound.
		if transaction.inflight.awaitFinished(parent) != nil {
			return
		}
		// The lease may now be held by a newer turn rather than the leader:
		// wait for it within the refusal's whole bound, not a short one.
		retryContext, retryCancel := context.WithCancel(parent)
		defer retryCancel()
		if lease, current, err = transaction.store.Prepare(retryContext, transaction.episodeIDHash, len(transaction.state.Warmth)); err != nil {
			return
		}
		stageContext = retryContext
	}
	defer func() {
		// Released under its own short bound: a stalled store must not hold
		// the refusal back.
		releaseContext, release := context.WithTimeout(context.Background(), relaxedBoundaryStageTimeout)
		defer release()
		_ = transaction.store.Abort(releaseContext, lease)
	}()
	next, changed := refusal.apply(current)
	if changed && stager.Stage(stageContext, lease, next) == nil {
		metrics.RecordRaylineARCEpisodeTransaction("boundary_cleared", selectionOutcomeRefusal)
	}
}

// takeHandedOverRefusal returns the refusal a coalesced resend handed to
// this request, once: one a failed commit already took, or else the entry's,
// sealing it.
func (transaction *raylineARCEpisodeTransaction) takeHandedOverRefusal() *refusedTurn {
	if refused := transaction.handedOver; refused != nil {
		transaction.handedOver = nil
		return refused
	}
	return transaction.inflight.takeRefusal()
}

// applyHandedOverRefusal applies a refusal a coalesced resend handed to this
// request while it held the lease. It runs as this request aborts, under its
// own lease.
func (transaction *raylineARCEpisodeTransaction) applyHandedOverRefusal(ctx context.Context) {
	refused := transaction.takeHandedOverRefusal()
	if refused == nil || transaction.state == nil {
		return
	}
	stager, ok := transaction.store.(raylinearc.EpisodeStateStager)
	if !ok {
		return
	}
	next, changed := refused.apply(transaction.state)
	if changed && stager.Stage(ctx, transaction.lease, next) == nil {
		transaction.state = next
		metrics.RecordRaylineARCEpisodeTransaction("boundary_cleared", selectionOutcomeRefusal)
	}
}
