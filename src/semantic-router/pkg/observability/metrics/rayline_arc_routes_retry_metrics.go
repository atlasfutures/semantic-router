package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// RaylineARCRoutesCallerBudget counts route lookups by caller-budget
	// outcome: admitted, rate_limited and inflight_limited (answered 429
	// caller_budget), fail_open (the shared store failed, or the process
	// table was full, so the lookup was counted per process or not at all)
	// and anonymous (the lookup named no caller). backend is redis
	// (cell-wide) or local (per process).
	RaylineARCRoutesCallerBudget = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_routes_caller_budget_total",
			Help: "Rayline ARC route lookups by per-caller budget outcome and backend.",
		},
		[]string{"outcome", "backend"},
	)
	// RaylineARCRoutesRefusalCache counts the refusal cache: hit (a repeat
	// answered from memory, no backend call), stored (a deterministic
	// decision refusal remembered) and evicted (dropped at the size bound).
	RaylineARCRoutesRefusalCache = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_routes_refusal_cache_total",
			Help: "Rayline ARC route lookup refusal cache events by outcome.",
		},
		[]string{"outcome"},
	)
	// RaylineARCRetryHints counts the retry headers the cell stamped on its
	// own non-2xx answers: surface is routes (/v1/routes) or turn (a live
	// ARC turn), status the HTTP status, should_retry the x-should-retry
	// value ("" when none is sent) and retry_after whether a Retry-After
	// was present on the answer.
	RaylineARCRetryHints = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_retry_hints_total",
			Help: "Rayline ARC cell non-2xx answers by surface, status and the retry headers they carry.",
		},
		[]string{"surface", "status", "should_retry", "retry_after"},
	)
)

// RecordRaylineARCRoutesCallerBudget records one caller-budget outcome.
func RecordRaylineARCRoutesCallerBudget(outcome, backend string) {
	RaylineARCRoutesCallerBudget.WithLabelValues(outcome, backend).Inc()
}

// RecordRaylineARCRoutesRefusalCache records one refusal-cache event.
func RecordRaylineARCRoutesRefusalCache(outcome string) {
	RaylineARCRoutesRefusalCache.WithLabelValues(outcome).Inc()
}

// RecordRaylineARCRetryHint records the retry headers on one cell answer.
func RecordRaylineARCRetryHint(surface, status, shouldRetry string, retryAfter bool) {
	present := "false"
	if retryAfter {
		present = "true"
	}
	RaylineARCRetryHints.WithLabelValues(surface, status, shouldRetry, present).Inc()
}
