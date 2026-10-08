package extproc

import (
	"context"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Cell exclusions (ADR 0120 Phase 1b) take a provider route out of the offer
// for a short time after it failed for capacity or availability: a 429, a 5xx,
// or a timeout. Unlike a refusal, which is about a model and a conversation,
// these are about the route, so they hold for every episode the process
// serves, and they expire on their own. They are kept per process: a replica
// learns from the failures it sees.
//
// no_endpoint is not among them: OpenRouter answers it as often for what one
// request asks (a price cap, an image input the provider does not take) as
// for a route that is gone, and excluding the route for everyone on the
// strength of one request would reroute requests it could serve.

// cellExclusionClasses are the failure classes that exclude a route.
var cellExclusionClasses = map[string]bool{
	turnFailureRateLimited: true,
	turnFailureUpstream5xx: true,
	turnFailureTimeout:     true,
}

type cellExclusion struct {
	class string
	until time.Time
}

// raylineARCCellExclusions maps a route to its exclusion. A route is a
// worker (its model ref), the backend endpoint it resolves to and the
// provider model there (policyWorkerRoute), so a config reload that moves the
// worker to another endpoint or model starts clean.
type raylineARCCellExclusions struct {
	mu     sync.Mutex
	worker map[string]cellExclusion
}

// raylineARCWorkerExclusions is the process's table. A route is the same in
// every decision that binds it, so one table serves them all; only
// fallback-enabled decisions read or write it.
var raylineARCWorkerExclusions = &raylineARCCellExclusions{worker: map[string]cellExclusion{}}

// exclude takes worker out of the offer until until; a later failure extends
// it, an earlier one never shortens it. Each insertion sweeps out what has
// expired, so a route a reload removed, which no scorer looks up again, does
// not stay in the table.
func (exclusions *raylineARCCellExclusions) exclude(worker, class string, until time.Time) {
	exclusions.mu.Lock()
	defer exclusions.mu.Unlock()
	exclusions.sweep(time.Now())
	if current, ok := exclusions.worker[worker]; ok && current.until.After(until) {
		return
	}
	exclusions.worker[worker] = cellExclusion{class: class, until: until}
}

// sweep drops every exclusion expired at now. The caller holds mu.
func (exclusions *raylineARCCellExclusions) sweep(now time.Time) {
	for worker, exclusion := range exclusions.worker {
		if !now.Before(exclusion.until) {
			delete(exclusions.worker, worker)
		}
	}
}

// active returns the class worker is excluded for at now, if it is.
func (exclusions *raylineARCCellExclusions) active(worker string, now time.Time) (string, bool) {
	exclusions.mu.Lock()
	defer exclusions.mu.Unlock()
	current, ok := exclusions.worker[worker]
	if !ok {
		return "", false
	}
	if !now.Before(current.until) {
		delete(exclusions.worker, worker)
		return "", false
	}
	return current.class, true
}

// noteCellExclusion excludes the route a failed turn was dispatched to, when
// its decision serves with the fallback on and the class is about the route.
func noteCellExclusion(ctx *RequestContext, class string) {
	if !cellExclusionClasses[class] || ctx.VSRRaylineARC == nil {
		return
	}
	decision := ctx.VSRSelectedDecision
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.RaylineARC == nil {
		return
	}
	policy := decision.Algorithm.RaylineARC.PolicyService
	arm := ctx.VSRRaylineARC.SelectedArm
	if !policy.FallbackEnabled() || arm < 0 || arm >= len(decision.ModelRefs) {
		return
	}
	worker := decision.ModelRefs[arm].Model
	providerModel := ctx.VSRRaylineARC.WorkerProviderModel
	route := ctx.VSRRaylineARC.WorkerRoute
	if route == "" {
		return
	}
	raylineARCWorkerExclusions.exclude(route, class, time.Now().Add(policy.CellExclusionTTL()))
	logging.ComponentEvent("extproc", "rayline_arc_cell_exclusion", map[string]interface{}{
		"request_id": ctx.RequestID, "worker": worker, "provider_model": providerModel,
		"failure_class": class, "ttl_seconds": policy.CellExclusionTTL().Seconds(),
	})
	clearFailedRouteBoundary(ctx, arm, class)
}

// clearFailedRouteBoundary clears a retained boundary decision that chose
// the arm whose route just failed, so the client's retry decides again
// rather than reusing it (pathfinder docs/arc_fallback_design.md, section
// 5.1: a transient failure leaves the retained boundary arm unchanged unless
// excluded, then cleared). A retry within the exclusion would break the hold
// anyway. One after it, from a client that waits longer than the exclusion
// lasts, would otherwise be pinned to the failed arm without the policy
// deciding again, and fail again for as long as the route is limited
// (semantic-router#241). The turn itself commits nothing, as any failed turn.
func clearFailedRouteBoundary(ctx *RequestContext, arm int, class string) {
	if ctx.RaylineARCTransaction == nil {
		return
	}
	clearContext, cancel := context.WithTimeout(context.Background(), episodeFinalizeTimeout)
	defer cancel()
	ctx.RaylineARCTransaction.stageRefusal(clearContext, refusedTurn{arm: arm, outcome: class})
}
