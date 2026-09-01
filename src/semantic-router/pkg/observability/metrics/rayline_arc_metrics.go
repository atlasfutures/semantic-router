/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	RaylineARCSelectionFailures = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_selection_failures_total",
			Help: "Rayline ARC fail-closed selections by bounded failure class.",
		},
		[]string{"class"},
	)
	// RaylineARCEncoderAdmissions counts admission-gate outcomes. A shed is a
	// deliberate capacity decision, so it is never folded into the encoder
	// failure counters: admitted + shed is the offered decision rate, which is
	// the only production measurement of arrival rate the router can take.
	RaylineARCEncoderAdmissions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_encoder_admissions_total",
			Help: "Rayline ARC encoder admission outcomes (admitted or shed).",
		},
		[]string{"outcome"},
	)
	RaylineARCEncoderInflight = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "llm_rayline_arc_encoder_inflight",
			Help: "Rayline ARC encoder calls currently holding an admission slot.",
		},
	)
	// RaylineARCEncoderInflightHighWater only rises. Compared against the
	// configured cap it shows whether the cap has ever bound.
	RaylineARCEncoderInflightHighWater = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "llm_rayline_arc_encoder_inflight_high_water",
			Help: "Largest concurrent Rayline ARC encoder call count seen since start.",
		},
	)
	RaylineARCEncoderLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_rayline_arc_encoder_latency_seconds",
			Help:    "End-to-end latency of the dedicated Rayline ARC vLLM encoder request.",
			Buckets: prometheus.ExponentialBuckets(0.01, 2, 16),
		},
	)
	RaylineARCTokens = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_rayline_arc_tokens",
			Help:    "Rayline ARC serialized, full, truncated, and cached token counts.",
			Buckets: []float64{16, 64, 256, 1024, 4096, 16384, 65536, 131072, 262144},
		},
		[]string{"kind"},
	)
	RaylineARCSwitchCost = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_rayline_arc_switch_cost_usd",
			Help:    "Estimated cold-switch cost for the selected Rayline ARC arm.",
			Buckets: prometheus.ExponentialBuckets(0.000001, 4, 12),
		},
	)
	RaylineARCCacheMissTokens = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_rayline_arc_cache_miss_tokens",
			Help:    "Estimated cache-miss tokens for the selected Rayline ARC arm.",
			Buckets: []float64{0, 64, 256, 1024, 4096, 16384, 65536, 131072, 262144},
		},
	)
	RaylineARCComponentReady = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_rayline_arc_component_ready",
			Help: "Readiness of the artifact/head and pinned vLLM encoder component.",
		},
		[]string{"component"},
	)
	RaylineARCEpisodeTransactions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_episode_transactions_total",
			Help: "Rayline ARC episode transaction outcomes and bounded failure classes.",
		},
		[]string{"outcome", "failure_class"},
	)
	RaylineARCSessionActions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_session_actions_total",
			Help: "Rayline ARC retained-session results by bounded lifecycle action.",
		},
		[]string{"action"},
	)
	RaylineARCEncoderReplicaRoutes = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_encoder_replica_routes_total",
			Help: "Rayline ARC retained-encoder routes by bounded direct or failover outcome.",
		},
		[]string{"outcome"},
	)
	RaylineARCEncoderReplicaAttempts = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_rayline_arc_encoder_replica_attempts",
			Help:    "Number of retained-encoder replica attempts per Rayline ARC selection.",
			Buckets: []float64{1, 2},
		},
	)
	RaylineARCEncoderSessionCloses = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_encoder_session_closes_total",
			Help: "Rayline ARC retained-session close fanout by bounded outcome.",
		},
		[]string{"outcome"},
	)
	RaylineARCProviderLogicalRequests = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_provider_logical_requests_total",
			Help: "Logical Rayline ARC OpenRouter requests by bounded final outcome.",
		},
		[]string{"outcome"},
	)
	RaylineARCProviderAttempts = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_provider_attempts_total",
			Help: "OpenRouter wire attempts beneath logical Rayline ARC requests by final outcome.",
		},
		[]string{"outcome"},
	)
	RaylineARCProviderRetries = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_provider_retries_total",
			Help: "Envoy-owned OpenRouter retries beneath logical Rayline ARC requests by final outcome.",
		},
		[]string{"outcome"},
	)
	RaylineARCProviderRetryExhaustions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_provider_retry_exhaustions_total",
			Help: "Rayline ARC OpenRouter retry budgets exhausted by bounded final status.",
		},
		[]string{"status"},
	)
)

func RecordRaylineARCFailure(class string) {
	RaylineARCSelectionFailures.WithLabelValues(class).Inc()
}

// RecordRaylineARCEncoderAdmission records one admission-gate outcome together
// with the gate's own occupancy. Callers pass the counts they read from the
// gate so that this package stays free of any dependency on the encoder.
func RecordRaylineARCEncoderAdmission(
	admitted bool,
	inflight int,
	highWater int,
) {
	outcome := "shed"
	if admitted {
		outcome = "admitted"
	}
	RaylineARCEncoderAdmissions.WithLabelValues(outcome).Inc()
	SetRaylineARCEncoderInflight(inflight)
	RaylineARCEncoderInflightHighWater.Set(float64(highWater))
}

// SetRaylineARCEncoderInflight publishes current admission occupancy. Callers
// also call it when a slot is released, so the gauge falls back to zero once a
// burst ends instead of holding the last admitted value until the next call.
func SetRaylineARCEncoderInflight(inflight int) {
	RaylineARCEncoderInflight.Set(float64(inflight))
}

func RecordRaylineARCSelection(
	latency time.Duration,
	serialized int,
	full int,
	truncated int,
	cached int,
	retained int,
	appended int,
	sessionAction string,
	switchCost float64,
	cacheMissTokens int,
) {
	RaylineARCEncoderLatency.Observe(latency.Seconds())
	for kind, count := range map[string]int{
		"serialized": serialized,
		"full":       full,
		"truncated":  truncated,
		"cached":     cached,
		"retained":   retained,
		"appended":   appended,
	} {
		RaylineARCTokens.WithLabelValues(kind).Observe(float64(count))
	}
	RaylineARCSwitchCost.Observe(switchCost)
	RaylineARCCacheMissTokens.Observe(float64(cacheMissTokens))
	if sessionAction != "" {
		RaylineARCSessionActions.WithLabelValues(sessionAction).Inc()
	}
}

func RecordRaylineARCEncoderReplicaRoute(attempts int, failover bool) {
	if attempts <= 0 {
		return
	}
	outcome := "direct"
	if failover {
		outcome = "failover"
	}
	RaylineARCEncoderReplicaRoutes.WithLabelValues(outcome).Inc()
	RaylineARCEncoderReplicaAttempts.Observe(float64(attempts))
}

func RecordRaylineARCEncoderSessionClose(
	closed int,
	unavailable int,
	failed int,
) {
	for outcome, count := range map[string]int{
		"closed":      closed,
		"unavailable": unavailable,
		"failed":      failed,
	} {
		if count > 0 {
			RaylineARCEncoderSessionCloses.WithLabelValues(outcome).Add(
				float64(count),
			)
		}
	}
}

func SetRaylineARCComponentReady(ready bool) {
	SetRaylineARCNamedComponentReady("artifact_head_encoder", ready)
}

func SetRaylineARCNamedComponentReady(
	component string,
	ready bool,
) {
	value := 0.0
	if ready {
		value = 1
	}
	RaylineARCComponentReady.WithLabelValues(component).Set(value)
}

func RecordRaylineARCEpisodeTransaction(
	outcome string,
	failureClass string,
) {
	RaylineARCEpisodeTransactions.WithLabelValues(
		outcome,
		failureClass,
	).Inc()
}

func RecordRaylineARCProviderRequest(
	outcome string,
	statusCode int,
	attempts uint64,
	exhausted bool,
) {
	if attempts < 1 {
		attempts = 1
	}
	RaylineARCProviderLogicalRequests.WithLabelValues(outcome).Inc()
	RaylineARCProviderAttempts.WithLabelValues(outcome).Add(float64(attempts))
	if attempts > 1 {
		RaylineARCProviderRetries.WithLabelValues(outcome).Add(
			float64(attempts - 1),
		)
	}
	if exhausted {
		RaylineARCProviderRetryExhaustions.WithLabelValues(
			strconv.Itoa(statusCode),
		).Inc()
	}
}
