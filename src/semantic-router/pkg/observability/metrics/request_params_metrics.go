package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	RequestParamsBlocked = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_request_params_blocked_total",
			Help: "Total number of blocked request parameters",
		},
		[]string{"decision", "param"},
	)

	// Capped counters intentionally omit original/capped as label values to avoid unbounded
	// Prometheus cardinality from arbitrary client-supplied numbers.
	RequestParamsMaxTokensCapped = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_request_params_max_tokens_capped_total",
			Help: "Total number of times max_tokens was capped",
		},
		[]string{"decision"},
	)

	RequestParamsMaxNCapped = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_request_params_max_n_capped_total",
			Help: "Total number of times n was capped",
		},
		[]string{"decision"},
	)

	// The floor is per model, and model names come from config, so the label
	// set stays bounded.
	RequestParamsCompletionFloorApplied = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_request_params_completion_floor_applied_total",
			Help: "Total number of times max_tokens was raised to a per-model floor",
		},
		[]string{"decision", "model"},
	)

	// One label only. The value is an episode hash, which is unbounded, so it
	// is never a label; the decision names the config that asked for it.
	RequestParamsUpstreamSessionIDApplied = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_request_params_upstream_session_id_applied_total",
			Help: "Total number of requests that carried an upstream session_id for provider affinity",
		},
		[]string{"decision"},
	)

	RequestParamsUnknownFieldStripped = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_request_params_unknown_field_stripped_total",
			Help: "Total number of unknown fields stripped from requests",
		},
		[]string{"decision", "field"},
	)
)

func RecordBlockedParam(decision, param string) {
	RequestParamsBlocked.WithLabelValues(decision, param).Inc()
}

func RecordMaxTokensCapped(decision string) {
	RequestParamsMaxTokensCapped.WithLabelValues(decision).Inc()
}

func RecordMaxNCapped(decision string) {
	RequestParamsMaxNCapped.WithLabelValues(decision).Inc()
}

func RecordCompletionFloorApplied(decision, model string) {
	RequestParamsCompletionFloorApplied.WithLabelValues(decision, model).Inc()
}

func RecordUnknownFieldStripped(decision, field string) {
	RequestParamsUnknownFieldStripped.WithLabelValues(decision, field).Inc()
}

func RecordUpstreamSessionIDApplied(decision string) {
	RequestParamsUpstreamSessionIDApplied.WithLabelValues(decision).Inc()
}
