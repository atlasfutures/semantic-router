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

// ARC live routing prices. Every label is a closed set: the fetch outcome
// (ok, transport, timeout, token, unauthorized, forbidden, not_found,
// unavailable, status, decode, request), the cache status a
// request read (fresh, stale, expired, absent), and the mode the policy
// service echoed (live, package, none).
var (
	RaylineARCRoutingPriceFetches = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_routing_price_fetches_total",
			Help: "Rayline ARC routing price feed fetches by bounded outcome.",
		},
		[]string{"outcome"},
	)
	// RaylineARCRoutingPriceLookups counts decision requests by the cache
	// status they read; expired and absent are the requests that fell back
	// to the package's prices because the feed was unusable.
	RaylineARCRoutingPriceLookups = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_routing_price_lookups_total",
			Help: "Rayline ARC decision requests by routing price cache status.",
		},
		[]string{"status"},
	)
	RaylineARCRoutingPriceAge = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "llm_rayline_arc_routing_price_age_seconds",
			Help: "Age of the cached Rayline ARC routing prices when a decision request last read them.",
		},
	)
	RaylineARCRoutingPriceDecisions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_routing_price_decisions_total",
			Help: "Rayline ARC policy decisions by the price mode the service echoed.",
		},
		[]string{"mode"},
	)
	RaylineARCRoutingPriceFallbackArms = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "llm_rayline_arc_routing_price_fallback_arms_total",
			Help: "Arms the policy service priced at the package rate instead of the live one.",
		},
	)
)

func RecordRaylineARCRoutingPriceFetch(outcome string) {
	RaylineARCRoutingPriceFetches.WithLabelValues(outcome).Inc()
}

func RecordRaylineARCRoutingPriceLookup(status string, age time.Duration) {
	RaylineARCRoutingPriceLookups.WithLabelValues(status).Inc()
	RaylineARCRoutingPriceAge.Set(age.Seconds())
}

func RecordRaylineARCRoutingPriceDecision(mode string, fallbackArms int) {
	RaylineARCRoutingPriceDecisions.WithLabelValues(mode).Inc()
	RaylineARCRoutingPriceFallbackArms.Add(float64(fallbackArms))
}
