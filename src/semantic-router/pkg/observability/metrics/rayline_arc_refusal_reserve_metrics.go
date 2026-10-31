package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// RaylineARCRefusalReserves counts what became of streamed ARC turns a
	// model refused before any text or tool call (pathfinder#6066 P1c):
	// served by a re-serve, or passed through and why. family is the
	// refusing model's vendor family, or "other".
	RaylineARCRefusalReserves = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_refusal_reserves_total",
			Help: "Rayline ARC refused streamed turns, by re-serve outcome and the refusing model's vendor family.",
		},
		[]string{"outcome", "family"},
	)
	// RaylineARCRefusalReserveAddedSeconds is how long a re-served turn's
	// client waited past the refusal for the served model's first text or
	// tool call.
	RaylineARCRefusalReserveAddedSeconds = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_rayline_arc_refusal_reserve_added_seconds",
			Help:    "Seconds from a refusal to the re-served model's first visible content.",
			Buckets: []float64{0.5, 1, 2, 4, 8, 15, 30, 45, 60, 90},
		},
	)
	// RaylineARCPreludeHoldSeconds is how long a held turn's frames were
	// withheld before its first text or tool call released them: the cost of
	// the hold on turns that were not refused.
	RaylineARCPreludeHoldSeconds = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_rayline_arc_prelude_hold_seconds",
			Help:    "Seconds a held Rayline ARC turn's frames were withheld before its first text or tool call.",
			Buckets: []float64{0.25, 0.5, 1, 2, 4, 8, 15, 30, 60},
		},
		[]string{"family"},
	)
)

// RecordRaylineARCRefusalReserve counts a refused turn's re-serve outcome.
func RecordRaylineARCRefusalReserve(outcome, family string) {
	if family == "" {
		family = "other"
	}
	RaylineARCRefusalReserves.WithLabelValues(outcome, family).Inc()
}

// RecordRaylineARCPreludeHold observes a held turn's release.
func RecordRaylineARCPreludeHold(family string, seconds float64) {
	if family == "" {
		family = "other"
	}
	RaylineARCPreludeHoldSeconds.WithLabelValues(family).Observe(seconds)
}

// RecordRaylineARCRefusalReserveAdded observes a re-served turn's added wait.
func RecordRaylineARCRefusalReserveAdded(seconds float64) {
	RaylineARCRefusalReserveAddedSeconds.Observe(seconds)
}
