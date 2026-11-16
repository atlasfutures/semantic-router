package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The start deadline (policy_service.start_deadline_seconds, 8 s by default).
//
// A turn that starts its episode on the service may land on a container that
// is still starting: a burst of new conversations scales the policy Server
// up, and Modal routes new sessions to the new container, whose encoder takes
// about a minute to load. The start then waits 25 s or more, or fails on the
// session start's own bound or the decide's total timeout, and the turn is
// lost with it. Nothing about the episode is known yet, so nothing is held.
//
// Such a turn is bounded by the start deadline instead. A start still
// undecided by then, or whose Modal session start failed, is served the
// policy package's fallback action, as an over-limit lookup is
// (rayline_routes_encode_bypass.go): the policy scores nothing and the
// episode's policy state is left as it was, so its next turn starts the
// episode again, by then most likely on a warm container. A session this
// turn started is ended, since no commit records it.
//
// Only routed turns fall back. A /v1/routes lookup keeps its own deadline and
// fails loudly (rayline_routes_lookup_scope.go): its caller asked for the
// policy's route, and a fallback answer would read as one. Nor does a turn
// carrying routing controls (an execution, a preset or priced models), whose
// result the fallback action cannot honour. A package that names no fallback
// action, or one this turn does not offer, fails as before.

// policyStartFallbackOfferKind is the offer kind of a start served the
// package's fallback action.
const policyStartFallbackOfferKind = "start_deadline_fallback"

// Why a start fell back.
const (
	policyStartLapseDeadline     = "deadline"
	policyStartLapseSessionStart = "session_start"
)

// startFallbackEligible reports whether this turn is bounded by the start
// deadline: a routed turn that starts its episode, with no routing controls,
// on a cell whose package names a fallback action the turn offers.
func (scorer *policyServiceScorer) startFallbackEligible(
	transition string,
	arcContext *selection.RaylineARCSelectionContext,
	routeControls *selection.RaylineARCRouteControls,
	execution *selection.RaylineARCRouteExecution,
	executionV2 *selection.RaylineARCRouteExecutionV2,
	available []string,
) bool {
	return scorer.startDeadline > 0 && transition == raylinearc.PolicyTransitionStart &&
		!arcContext.RoutesLookup && routeControls == nil && execution == nil && executionV2 == nil &&
		scorer.fallbackActionID != "" && slices.Contains(available, scorer.fallbackActionID)
}

// policyStartLapsed reports whether a start failed because it ran out of the
// start deadline (and not the turn's own time), or because its Modal session
// could not be started.
func policyStartLapsed(turnCtx, decideCtx context.Context, err error) bool {
	return policyStartLapseReason(decideCtx, err) != "" && turnCtx.Err() == nil
}

// policyStartLapseReason names why a start lapsed, empty when it did not.
func policyStartLapseReason(decideCtx context.Context, err error) string {
	if errors.Is(decideCtx.Err(), context.DeadlineExceeded) {
		return policyStartLapseDeadline
	}
	var failure *raylinearc.PolicyServiceError
	if errors.As(err, &failure) && failure.Class == raylinearc.PolicySessionStartClass {
		return policyStartLapseSessionStart
	}
	return ""
}

// policyStartFallback is the turn served on the package's fallback action
// after its start lapsed; nil when that action is not a valid decision here,
// and the turn then fails as before.
func (selector *raylineARCSelector) policyStartFallback(
	armed *raylineARCArmedComponents,
	selCtx *selection.SelectionContext,
	arcContext *selection.RaylineARCSelectionContext,
	state *raylinearc.EpisodeState,
	scorer *policyServiceScorer,
	turn *raylinearc.PolicyEpisodeState,
	messages []json.RawMessage,
	sideCall bool,
	atBoundary bool,
	retained bool,
	workerIDs []string,
	reason string,
	latency time.Duration,
) *selection.SelectionResult {
	actionID := scorer.fallbackActionID
	// No ledger: the episode's policy state is left as it was, so the next
	// turn starts the episode again.
	result := selector.policyUndecidedResult(
		armed, selCtx, arcContext, state, scorer, &raylinearc.EncoderResult{}, actionID, "", false,
		turn, messages, sideCall, atBoundary, retained, workerIDs,
	)
	if result == nil {
		return nil
	}
	// Nothing was measured: the counts are unknown, not zero.
	result.RaylineARC.TokenCountsUnknown = true
	result.RaylineARC.OfferKind = policyStartFallbackOfferKind
	result.RaylineARC.PolicyLatency = latency
	result.Reasoning = "policy-service ARC decision (" + policyStartFallbackOfferKind + ")"
	logging.ComponentWarnEvent("extproc", "rayline_arc_start_deadline_fallback", map[string]interface{}{
		"request_id": arcContext.RequestID, "episode_id_hash": arcContext.EpisodeIDHash,
		"reason": reason, "action_id": actionID, "side_call": sideCall,
		"deadline_ms": scorer.startDeadline.Milliseconds(), "latency_ms": latency.Milliseconds(),
	})
	return result
}
