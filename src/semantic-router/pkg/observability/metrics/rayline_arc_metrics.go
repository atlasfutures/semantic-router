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
)

func RecordRaylineARCFailure(class string) {
	RaylineARCSelectionFailures.WithLabelValues(class).Inc()
}

func RecordRaylineARCSelection(
	latency time.Duration,
	serialized int,
	full int,
	truncated int,
	cached int,
	switchCost float64,
	cacheMissTokens int,
) {
	RaylineARCEncoderLatency.Observe(latency.Seconds())
	for kind, count := range map[string]int{
		"serialized": serialized,
		"full":       full,
		"truncated":  truncated,
		"cached":     cached,
	} {
		RaylineARCTokens.WithLabelValues(kind).Observe(float64(count))
	}
	RaylineARCSwitchCost.Observe(switchCost)
	RaylineARCCacheMissTokens.Observe(float64(cacheMissTokens))
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
