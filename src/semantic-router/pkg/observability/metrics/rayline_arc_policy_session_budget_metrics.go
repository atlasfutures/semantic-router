package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// RaylineARCPolicySessionBudget counts the turns that would open a new policy
// session and were checked against their caller's new-session budget, by
// outcome: admitted (the turn opens its session), over_budget_sessionless
// (decided one-shot on the shared overflow session, no session opened) and
// redis_error (the count was unknown, so the turn was admitted: fail open).
var RaylineARCPolicySessionBudget = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "llm_rayline_arc_policy_session_budget_total",
		Help: "Rayline ARC turns checked against their caller's new policy-session budget, by outcome.",
	},
	[]string{"outcome"},
)

// RecordRaylineARCPolicySessionBudget records one budget check's outcome.
func RecordRaylineARCPolicySessionBudget(outcome string) {
	RaylineARCPolicySessionBudget.WithLabelValues(outcome).Inc()
}
