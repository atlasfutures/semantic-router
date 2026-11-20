package metrics

import (
	"sync/atomic"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// RaylineARCRoutesAdmissions counts route lookups by admission outcome:
	// admitted (a slot at once), queued (a slot after waiting), shed (no slot
	// within the queue wait: answered 429), cancelled (the caller or the
	// lookup deadline gave up while waiting) and fail_open (the shared store
	// failed, so the lookup was admitted on the per-process cap alone).
	// backend is redis (cell-wide) or local (per process).
	RaylineARCRoutesAdmissions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_routes_admission_total",
			Help: "Rayline ARC route lookups by admission outcome and gate backend.",
		},
		[]string{"outcome", "backend"},
	)
	// RaylineARCRoutesAdmissionWaitSeconds is how long a lookup waited at the
	// admission gate, whatever the outcome. Lookups serialized behind one
	// another show up as waits near the holder's encode time.
	RaylineARCRoutesAdmissionWaitSeconds = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_rayline_arc_routes_admission_wait_seconds",
			Help:    "Seconds a Rayline ARC route lookup waited for an admission slot, by outcome.",
			Buckets: []float64{0.005, 0.025, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30},
		},
		[]string{"outcome"},
	)
)

// RecordRaylineARCRoutesAdmission records one admission outcome and its wait.
func RecordRaylineARCRoutesAdmission(outcome, backend string, waitSeconds float64) {
	RaylineARCRoutesAdmissions.WithLabelValues(outcome, backend).Inc()
	RaylineARCRoutesAdmissionWaitSeconds.WithLabelValues(outcome).Observe(waitSeconds)
}

// raylineARCRoutesLookupSessionsSource reads the retained encoder sessions
// tracked lookups hold in this process; nil until the router sets it.
var raylineARCRoutesLookupSessionsSource atomic.Pointer[func() int]

// RaylineARCRoutesLookupSessions is, per process, the retained encoder
// sessions tracked route lookups hold: the distinct episodes whose last
// lookup left a session on the encoder within the episode idle TTL. A dev
// check sums it against the encoder container's session budget. Read at
// scrape time.
var RaylineARCRoutesLookupSessions = promauto.NewGaugeFunc(
	prometheus.GaugeOpts{
		Name: "llm_rayline_arc_routes_lookup_sessions",
		Help: "Retained encoder sessions held by this process's tracked Rayline ARC route lookups (estimate, within the episode idle TTL).",
	},
	func() float64 {
		if source := raylineARCRoutesLookupSessionsSource.Load(); source != nil {
			return float64((*source)())
		}
		return 0
	},
)

// SetRaylineARCRoutesLookupSessionsSource sets what the lookup-sessions gauge reads.
func SetRaylineARCRoutesLookupSessionsSource(source func() int) {
	raylineARCRoutesLookupSessionsSource.Store(&source)
}

// RaylineARCRoutesEncodeBypasses counts route lookups over
// routes_api.max_encode_tokens, by outcome: held (answered with the
// conversation's held model), fallback (with the package's fallback action)
// and encoded (neither to serve, or routing controls: encoded as before).
var RaylineARCRoutesEncodeBypasses = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "llm_rayline_arc_routes_encode_bypass_total",
		Help: "Rayline ARC route lookups over the encode limit, by outcome.",
	},
	[]string{"outcome"},
)

// RecordRaylineARCRoutesEncodeBypass records one lookup over the encode limit.
func RecordRaylineARCRoutesEncodeBypass(outcome string) {
	RaylineARCRoutesEncodeBypasses.WithLabelValues(outcome).Inc()
}

// RaylineARCRoutesLookups counts route lookups by the routing.execution
// envelope they carry: none, v1 or v2. Read beside each other, the labels
// show what share of lookups send a caller-held execution context, and in
// which version. A lookup is counted once its routing block is read,
// whatever it is then answered, including a refusal of its messages. One
// refused before that (a body that is not JSON, a routing block that is
// itself invalid) is not, as its envelope is not known.
var RaylineARCRoutesLookups = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "llm_rayline_arc_routes_lookups_total",
		Help: "Rayline ARC route lookups by the routing.execution envelope they carry (none, v1, v2).",
	},
	[]string{"execution"},
)

// RecordRaylineARCRoutesLookup records one route lookup by execution version.
func RecordRaylineARCRoutesLookup(execution string) {
	RaylineARCRoutesLookups.WithLabelValues(execution).Inc()
}

// RaylineARCRouteReports counts /v1/report calls that were read -- answered
// 200, or failed against the episode store -- by who reported (gateway,
// client), what happened (recorded, ignored, error) and why (the answer's
// reason, or the error code). Refusals of a malformed request are not
// counted.
var RaylineARCRouteReports = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "llm_rayline_arc_route_reports_total",
		Help: "Rayline ARC reports of what ran after a route lookup, by source, status and reason.",
	},
	[]string{"source", "status", "reason"},
)

// RecordRaylineARCRouteReport records one report's answer.
func RecordRaylineARCRouteReport(source, status, reason string) {
	RaylineARCRouteReports.WithLabelValues(source, status, reason).Inc()
}
