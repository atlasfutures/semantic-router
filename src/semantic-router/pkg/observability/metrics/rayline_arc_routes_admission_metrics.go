package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// RaylineARCRoutesAdmissions counts route lookups by admission outcome:
	// admitted (a slot at once), queued (a slot after waiting), shed (no slot
	// within the queue wait: answered 429), cancelled (the caller or the
	// lookup deadline gave up while waiting) and fail_open (the shared store
	// failed, so the lookup was admitted on the per-process cap alone),
	// and exempt (a tracked lookup predicted to append, which skipped the
	// gate). backend is redis (cell-wide), local (per process) or none
	// (exempt).
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

// RaylineARCRoutesAdmissionPredictions counts tracked lookups whose admission
// was decided on an encode prediction (routes_api.admission.append_exempt), by
// admission (gated or exempt), the prediction (extends, or why the encode was
// predicted full) and the session_action the decision actually reported
// (created, rebuilt, appended, reused; none when the decision failed or the
// service reported no session). An exempt lookup whose session_action is
// created or rebuilt was a full encode the prediction missed.
var RaylineARCRoutesAdmissionPredictions = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "llm_rayline_arc_routes_admission_prediction_total",
		Help: "Rayline ARC tracked route lookups by admission, encode prediction and actual session action.",
	},
	[]string{"admission", "prediction", "session_action"},
)

// RecordRaylineARCRoutesAdmissionPrediction records one predicted lookup.
func RecordRaylineARCRoutesAdmissionPrediction(admission, prediction, sessionAction string) {
	RaylineARCRoutesAdmissionPredictions.WithLabelValues(admission, prediction, sessionAction).Inc()
}

// RecordRaylineARCRoutesAdmission records one admission outcome and its wait.
func RecordRaylineARCRoutesAdmission(outcome, backend string, waitSeconds float64) {
	RaylineARCRoutesAdmissions.WithLabelValues(outcome, backend).Inc()
	RaylineARCRoutesAdmissionWaitSeconds.WithLabelValues(outcome).Observe(waitSeconds)
}
