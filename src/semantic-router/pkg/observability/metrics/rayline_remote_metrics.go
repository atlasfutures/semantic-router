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
	raylineRemoteReady = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "llm_rayline_remote_ready",
			Help: "Readiness of the pinned Pathfinder selection contract.",
		},
	)
	raylineRemoteSelections = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_remote_selections_total",
			Help: "Remote Rayline selections by candidate ordinal.",
		},
		[]string{"candidate_index"},
	)
	raylineRemoteSelectionLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_rayline_remote_selection_latency_seconds",
			Help:    "Latency of Pathfinder prepare calls.",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 16),
		},
	)
	raylineRemoteFailures = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_remote_failures_total",
			Help: "Remote Rayline failures by bounded lifecycle stage and class.",
		},
		[]string{"stage", "class"},
	)
	raylineRemoteTransactions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_remote_transactions_total",
			Help: "Remote Rayline transaction terminal operations and outcomes.",
		},
		[]string{"operation", "outcome"},
	)
)

func SetRaylineRemoteReady(ready bool) {
	value := 0.0
	if ready {
		value = 1
	}
	raylineRemoteReady.Set(value)
}

func RecordRaylineRemoteSelection(
	candidateIndex int,
	latency time.Duration,
) {
	raylineRemoteSelections.WithLabelValues(
		strconv.Itoa(candidateIndex),
	).Inc()
	raylineRemoteSelectionLatency.Observe(latency.Seconds())
}

func RecordRaylineRemoteFailure(stage string, class string) {
	raylineRemoteFailures.WithLabelValues(stage, class).Inc()
}

func RecordRaylineRemoteTransaction(
	operation string,
	outcome string,
) {
	raylineRemoteTransactions.WithLabelValues(
		operation,
		outcome,
	).Inc()
}
