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

package extproc

// ARC live routing prices in the policy-service mode: the cache that holds
// the feed, the routing_prices each decision request carries, and the check
// that the service's routing_prices_used echo is the prices this request
// sent, since the echo becomes the decision's price audit.

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const (
	arcFailurePolicyPricesUnsent   = "policy_prices_unsent"
	arcFailurePolicyPricesMismatch = "policy_prices_mismatch"
	arcFailurePolicyPricesUnechoed = "policy_prices_unechoed"

	// maxLoggedFallbackArms bounds the fallback arms one selection log
	// lists, the same as the ledger audit keeps.
	maxLoggedFallbackArms = 8
	// maxLoggedFilledModels bounds the models whose filled legs one
	// selection log lists, the same as the ledger audit keeps.
	maxLoggedFilledModels = 8
)

// newRoutingPriceCache resolves the configured feed; nil when routing_prices
// is not configured. A configured static credential that is unset leaves
// live prices off, said once in the log: every decision then uses package
// prices. google_id_token mints from the metadata server on each fetch that
// needs a token, so a missing identity shows as fetch failures instead.
func newRoutingPriceCache(cfg *config.RaylineARCRoutingPricesConfig) *raylinearc.RoutingPriceCache {
	return newRoutingPriceCacheWithMetadata(cfg, raylinearc.DefaultMetadataIdentityURL)
}

func newRoutingPriceCacheWithMetadata(cfg *config.RaylineARCRoutingPricesConfig, metadataURL string) *raylinearc.RoutingPriceCache {
	if cfg == nil {
		return nil
	}
	var credential string
	var tokens raylinearc.RoutingPriceTokenSource
	if cfg.Auth == config.RaylineARCRoutingPricesAuthGoogleIDToken {
		tokens = raylinearc.NewMetadataIDTokenSource(cfg.Audience, metadataURL, nil)
	} else {
		var err error
		if credential, err = raylineARCOptionalSecret(cfg.CredentialEnv); err != nil {
			logging.ComponentErrorEvent("extproc", "rayline_arc_routing_prices_unavailable", map[string]interface{}{
				"class": "credential",
			})
			return nil
		}
	}
	return raylinearc.NewRoutingPriceCache(raylinearc.RoutingPriceCacheConfig{
		URL:          cfg.URL,
		Credential:   credential,
		TokenSource:  tokens,
		TTL:          time.Duration(cfg.TTLSeconds) * time.Second,
		RefreshAhead: time.Duration(cfg.RefreshAheadSeconds) * time.Second,
		MaxAge:       time.Duration(cfg.MaxAgeSeconds) * time.Second,
		FetchTimeout: time.Duration(cfg.FetchTimeoutMillis) * time.Millisecond,
		OnFetch:      observeRoutingPriceFetch,
	})
}

func observeRoutingPriceFetch(class string, status int) {
	metrics.RecordRaylineARCRoutingPriceFetch(class)
	if class == "ok" {
		return
	}
	logging.ComponentWarnEvent("extproc", "rayline_arc_routing_prices_fetch_failed", map[string]interface{}{
		"class": class, "status": status,
	})
}

// attachRoutingPrices puts the cached rates on the request and returns the
// cache status it read; empty when live prices are not configured. Past max
// age, or before the first fetch, no rates are attached and the service
// scores with the package's prices.
func attachRoutingPrices(cache *raylinearc.RoutingPriceCache, request *raylinearc.PolicyDecisionRequest) string {
	if cache == nil {
		return ""
	}
	prices, status, age := cache.Lookup()
	metrics.RecordRaylineARCRoutingPriceLookup(string(status), age)
	request.RoutingPrices = prices
	return string(status)
}

// routingPriceEchoMismatch names what makes the echo untrue of this
// request: no echo when the request sent routing_prices, an echo when it
// sent none, or another feed version or as-of time than it sent. The
// service echoes exactly when the request carried prices, naming their
// store_version and as_of verbatim, in either mode (package when the
// package is ineligible or every arm fell back). A decision priced from
// sent rates must leave an audit, so a missing echo fails the turn. Empty
// when the echo, or its absence, can be recorded as it is.
func routingPriceEchoMismatch(sent *raylinearc.PolicyRoutingPrices, used *raylinearc.PolicyRoutingPricesUsed) string {
	if used == nil {
		if sent != nil {
			return arcFailurePolicyPricesUnechoed
		}
		return ""
	}
	if sent == nil {
		return arcFailurePolicyPricesUnsent
	}
	if *used.StoreVersion != sent.StoreVersion || !sameInstant(*used.AsOf, sent.AsOf) {
		return arcFailurePolicyPricesMismatch
	}
	// filled_legs is keyed by the request's rate keys; a model this request
	// did not send cannot have had a leg filled.
	for model := range used.FilledLegs {
		if _, ok := sent.Rates[model]; !ok {
			return arcFailurePolicyPricesMismatch
		}
	}
	return ""
}

// sameInstant compares two RFC 3339 timestamps as instants, so the service
// may write as_of in another offset form than the feed did.
func sameInstant(left string, right string) bool {
	if left == right {
		return true
	}
	leftTime, leftErr := time.Parse(time.RFC3339Nano, left)
	rightTime, rightErr := time.Parse(time.RFC3339Nano, right)
	return leftErr == nil && rightErr == nil && leftTime.Equal(rightTime)
}

func recordRoutingPriceDecision(used *raylinearc.PolicyRoutingPricesUsed) {
	if used == nil {
		metrics.RecordRaylineARCRoutingPriceDecision("none", 0)
		return
	}
	metrics.RecordRaylineARCRoutingPriceDecision(used.Mode, len(used.FallbackArms))
}

// addPolicyPriceAuditFields writes a policy decision's price audit into its
// selection log: the package, the chosen arm, the predicted cost, the cache
// status the request read, and the service's echo in full (the ledger keeps
// a bounded copy).
func addPolicyPriceAuditFields(fields map[string]interface{}, trace *selection.RaylineARCTrace) {
	fields["policy_arm_id"] = trace.PolicyArmID
	fields["policy_package_sha256"] = trace.PolicyPackageSHA256
	if trace.PolicyPredictedCostUSD != nil {
		fields["policy_predicted_cost_usd"] = *trace.PolicyPredictedCostUSD
	}
	if trace.RoutingPricesStatus != "" {
		fields["routing_prices_status"] = trace.RoutingPricesStatus
	}
	used := trace.RoutingPricesUsed
	if used == nil {
		return
	}
	fields["routing_prices_mode"] = used.Mode
	// The echo lists every arm that fell back, over a thousand on a large
	// package; the log keeps the first few, as the ledger does, and the count.
	arms := used.FallbackArms
	if len(arms) > maxLoggedFallbackArms {
		arms = arms[:maxLoggedFallbackArms]
	}
	fields["routing_prices_fallback_arms"] = arms
	fields["routing_prices_fallback_arm_count"] = len(used.FallbackArms)
	filled, filledCount := raylinearc.BoundedFilledLegs(used.FilledLegs, maxLoggedFilledModels, raylinearc.MaxRoutingPriceRateKeyBytes)
	if filled == nil {
		filled = map[string][]string{}
	}
	fields["routing_prices_filled_legs"] = filled
	fields["routing_prices_filled_model_count"] = filledCount
	for name, value := range map[string]*string{
		"routing_prices_store_version": used.StoreVersion,
		"routing_prices_as_of":         used.AsOf,
		"routing_prices_digest":        used.Digest,
	} {
		if value != nil {
			fields[name] = *value
		}
	}
}

// policySelectedPredictedCost is the chosen action's predicted cost, nil when
// the service did not predict one.
func policySelectedPredictedCost(response *raylinearc.PolicyDecisionResponse) *float64 {
	for _, action := range response.Actions {
		if action.ActionID == response.Decision.SelectedActionID && action.PredictedCostUSD != nil {
			cost := *action.PredictedCostUSD
			return &cost
		}
	}
	return nil
}
