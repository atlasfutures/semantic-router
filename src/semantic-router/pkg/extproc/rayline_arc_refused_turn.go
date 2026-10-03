package extproc

import (
	"context"
	"errors"

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
	clearContext, cancel := context.WithTimeout(context.Background(), episodeFinalizeTimeout)
	defer cancel()
	ctx.RaylineARCTransaction.clearRefusedBoundary(clearContext, ctx.VSRRaylineARC.SelectedArm)
}

// clearRefusedBoundary removes a retained boundary decision that chose arm,
// leaving the rest of the episode as stored. Best effort, like retaining it:
// a failure leaves the retry to reuse the decision, as before.
func (transaction *raylineARCEpisodeTransaction) clearRefusedBoundary(ctx context.Context, arm int) {
	if transaction == nil || transaction.state == nil || transaction.state.PolicyBoundary == nil ||
		transaction.state.PolicyBoundary.Arm != arm || transaction.sideCall || transaction.stateless {
		return
	}
	if transaction.borrowed {
		transaction.clearBorrowedRefusedBoundary(ctx, arm)
		return
	}
	cleared := cloneARCState(transaction.state)
	cleared.PolicyBoundary = nil
	var err error
	if transaction.relaxed {
		// A relaxed turn never waits on episode state: the same short bound
		// as staging the boundary.
		relaxedContext, cancel := context.WithTimeout(ctx, relaxedBoundaryStageTimeout)
		defer cancel()
		err = transaction.snapshots.CommitIfUnchanged(relaxedContext, transaction.episodeIDHash, transaction.read, cleared)
	} else if stager, ok := transaction.store.(raylinearc.EpisodeStateStager); ok {
		err = stager.Stage(ctx, transaction.lease, cleared)
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
	transaction.state = cleared
	metrics.RecordRaylineARCEpisodeTransaction("boundary_cleared", selectionOutcomeRefusal)
}

// clearBorrowedRefusedBoundary clears the boundary decision for a coalesced
// resend that was refused. The resend holds no lease: the request that
// decided owns it, and if that request failed it aborted and kept the
// decision, so the only refusal would otherwise leave the retry on the
// refusing arm. The resend takes the lease briefly itself, and clears the
// decision only if it still chose the refused arm. A lease still held means
// the deciding request is running, and its own outcome decides.
func (transaction *raylineARCEpisodeTransaction) clearBorrowedRefusedBoundary(parent context.Context, arm int) {
	stager, ok := transaction.store.(raylinearc.EpisodeStateStager)
	if !ok || transaction.store == nil {
		return
	}
	ctx, cancel := context.WithTimeout(parent, relaxedBoundaryStageTimeout)
	defer cancel()
	lease, current, err := transaction.store.Prepare(ctx, transaction.episodeIDHash, len(transaction.state.Warmth))
	if err != nil {
		return
	}
	defer func() { _ = transaction.store.Abort(context.Background(), lease) }()
	if current == nil || current.PolicyBoundary == nil || current.PolicyBoundary.Arm != arm {
		return
	}
	cleared := cloneARCState(current)
	cleared.PolicyBoundary = nil
	if stager.Stage(ctx, lease, cleared) == nil {
		metrics.RecordRaylineARCEpisodeTransaction("boundary_cleared", selectionOutcomeRefusal)
	}
}
