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

package raylinearc

// Wire types for ARC live routing prices: the price feed this router reads,
// the routing_prices it attaches to a policy decision request, and the
// routing_prices_used echo the service answers with. The feed's per-model
// rates are forwarded as they are, keyed by provider model id; this router
// never maps them to trained arms, which is the service's job.

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

const (
	// RoutingPricesSchema names both the feed response and the request's
	// routing_prices object.
	RoutingPricesSchema   = "arc-routing-prices.v1"
	routingPricesCurrency = "USD"
	routingPricesUnit     = "usd_per_million_tokens"

	RoutingPricesModeLive    = "live"
	RoutingPricesModePackage = "package"

	maxRoutingPriceStoreVersionBytes = 256
	maxRoutingPriceTimestampBytes    = 64
	maxRoutingPriceModelIDBytes      = 256
	maxRoutingPriceModels            = 4096
	// maxRoutingPriceFallbackArms is only a safety bound: the service lists
	// every arm that fell back, unbounded, and a package can price over a
	// thousand arms, most of which fall back when the feed covers few of
	// their models. That must degrade to package prices, not fail the
	// decode. The ledger and the log keep only the first few and the count.
	maxRoutingPriceFallbackArms = 65_536
)

// RoutingPriceFeedModel is one model's effective rates at the feed's as_of,
// in USD per million tokens. A leg the source does not know is null.
type RoutingPriceFeedModel struct {
	Model      string   `json:"model"`
	Input      *float64 `json:"input"`
	CacheRead  *float64 `json:"cache_read"`
	CacheWrite *float64 `json:"cache_write"`
	Output     *float64 `json:"output"`
	ValidFrom  *string  `json:"valid_from"`
	ValidUntil *string  `json:"valid_until"`
	Source     *string  `json:"source"`
}

// RoutingPriceFeed is the price feed's response.
type RoutingPriceFeed struct {
	Schema       string                  `json:"schema"`
	StoreVersion string                  `json:"store_version"`
	AsOf         string                  `json:"as_of"`
	Currency     string                  `json:"currency"`
	Unit         string                  `json:"unit"`
	Models       []RoutingPriceFeedModel `json:"models"`
}

// PolicyRoutingRate is one model's rates as the decision request carries
// them, the feed's entry less its model id. Every member is written, null
// when unknown.
type PolicyRoutingRate struct {
	Input      *float64 `json:"input"`
	CacheRead  *float64 `json:"cache_read"`
	CacheWrite *float64 `json:"cache_write"`
	Output     *float64 `json:"output"`
	ValidFrom  *string  `json:"valid_from"`
	ValidUntil *string  `json:"valid_until"`
	Source     *string  `json:"source"`
}

// PolicyRoutingPrices is the decision request's routing_prices.
type PolicyRoutingPrices struct {
	Schema       string                       `json:"schema"`
	StoreVersion string                       `json:"store_version"`
	AsOf         string                       `json:"as_of"`
	Rates        map[string]PolicyRoutingRate `json:"rates"`
}

// PolicyRoutingPricesUsed is the decision response's routing_prices_used:
// whether live rates scored the decision, the feed identity it used, the
// digest of the effective per-arm rates, and the arms that fell back to the
// package's rate.
type PolicyRoutingPricesUsed struct {
	Mode         string   `json:"mode"`
	StoreVersion *string  `json:"store_version"`
	AsOf         *string  `json:"as_of"`
	Digest       *string  `json:"digest"`
	FallbackArms []string `json:"fallback_arms"`
}

// DecodeRoutingPriceFeed reads a feed response. Unknown fields are allowed,
// since the feed may grow and this router forwards only the rates; the
// schema, currency, unit and identity are not.
func DecodeRoutingPriceFeed(body []byte) (*RoutingPriceFeed, error) {
	var feed RoutingPriceFeed
	if err := json.Unmarshal(body, &feed); err != nil {
		return nil, err
	}
	if feed.Schema != RoutingPricesSchema {
		return nil, fmt.Errorf("routing price feed schema %q", feed.Schema)
	}
	if feed.Currency != routingPricesCurrency || feed.Unit != routingPricesUnit {
		return nil, fmt.Errorf("routing price feed is %s %s", feed.Currency, feed.Unit)
	}
	if feed.StoreVersion == "" || len(feed.StoreVersion) > maxRoutingPriceStoreVersionBytes {
		return nil, errors.New("routing price feed store_version is empty or too long")
	}
	if err := validRoutingPriceTimestamp(feed.AsOf); err != nil {
		return nil, fmt.Errorf("routing price feed as_of: %w", err)
	}
	if len(feed.Models) > maxRoutingPriceModels {
		return nil, fmt.Errorf("routing price feed lists %d models", len(feed.Models))
	}
	seen := make(map[string]bool, len(feed.Models))
	for _, model := range feed.Models {
		if model.Model == "" || len(model.Model) > maxRoutingPriceModelIDBytes {
			return nil, errors.New("routing price feed model id is empty or too long")
		}
		if seen[model.Model] {
			return nil, fmt.Errorf("routing price feed lists model %q twice", model.Model)
		}
		seen[model.Model] = true
		if model.ValidUntil != nil {
			if err := validRoutingPriceTimestamp(*model.ValidUntil); err != nil {
				return nil, fmt.Errorf("routing price feed valid_until: %w", err)
			}
		}
	}
	return &feed, nil
}

// RequestPrices is the feed as the decision request's routing_prices: every
// model's rates, keyed by its id, values as served.
func (feed *RoutingPriceFeed) RequestPrices() *PolicyRoutingPrices {
	rates := make(map[string]PolicyRoutingRate, len(feed.Models))
	for _, model := range feed.Models {
		rates[model.Model] = PolicyRoutingRate{
			Input: model.Input, CacheRead: model.CacheRead, CacheWrite: model.CacheWrite,
			Output: model.Output, ValidFrom: model.ValidFrom, ValidUntil: model.ValidUntil, Source: model.Source,
		}
	}
	return &PolicyRoutingPrices{
		Schema: RoutingPricesSchema, StoreVersion: feed.StoreVersion, AsOf: feed.AsOf, Rates: rates,
	}
}

func validRoutingPriceTimestamp(value string) error {
	if len(value) > maxRoutingPriceTimestampBytes {
		return errors.New("timestamp too long")
	}
	_, err := time.Parse(time.RFC3339Nano, value)
	return err
}

// validate refuses an echo this router could not record truthfully: an
// unknown mode, an echo without the feed identity, or values past the
// ledger's bounds. The service echoes exactly when the request carried
// routing_prices and always names that store_version and as_of; a nil echo
// answers a request without them. digest is null only when the service holds
// no per-arm price table for the package.
func (used *PolicyRoutingPricesUsed) validate() error {
	if used == nil {
		return nil
	}
	if used.Mode != RoutingPricesModeLive && used.Mode != RoutingPricesModePackage {
		return fmt.Errorf("routing_prices_used mode %q", used.Mode)
	}
	if used.StoreVersion == nil || *used.StoreVersion == "" || used.AsOf == nil || *used.AsOf == "" {
		return errors.New("routing_prices_used names no store_version or as_of")
	}
	if used.StoreVersion != nil && len(*used.StoreVersion) > maxRoutingPriceStoreVersionBytes {
		return errors.New("routing_prices_used store_version too long")
	}
	if used.AsOf != nil && len(*used.AsOf) > maxRoutingPriceTimestampBytes {
		return errors.New("routing_prices_used as_of too long")
	}
	if used.Digest != nil && !isLowerHex64(*used.Digest) {
		return errors.New("routing_prices_used digest is not a sha256")
	}
	if len(used.FallbackArms) > maxRoutingPriceFallbackArms {
		return errors.New("routing_prices_used lists too many fallback arms")
	}
	for _, arm := range used.FallbackArms {
		if arm == "" || len(arm) > maxPolicyArmIDBytes {
			return errors.New("routing_prices_used fallback arm id is empty or too long")
		}
	}
	return nil
}
