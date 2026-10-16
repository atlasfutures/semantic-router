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
	"regexp"
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
	// MaxRoutingPriceModelIDBytes is the longest model id the feed and the
	// echo may name.
	MaxRoutingPriceModelIDBytes = maxRoutingPriceModelIDBytes
	maxRoutingPriceModels       = 4096
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
	// legsPresent is whether the entry wrote all four legs, null or not.
	// The service fills a null leg from the package but takes an omitted
	// one as a malformed entry, so the request must not turn one into the
	// other.
	legsPresent bool
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
// digest of the effective per-arm rates, the arms that fell back to the
// package's rate, and per request rate key the legs the feed sent null that
// took the package's value (per-leg fill), in the contract's leg order.
type PolicyRoutingPricesUsed struct {
	Mode         string              `json:"mode"`
	StoreVersion *string             `json:"store_version"`
	AsOf         *string             `json:"as_of"`
	Digest       *string             `json:"digest"`
	FallbackArms []string            `json:"fallback_arms"`
	FilledLegs   map[string][]string `json:"filled_legs"`
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
	var legs struct {
		Models []map[string]json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(body, &legs); err != nil {
		return nil, err
	}
	seen := make(map[string]bool, len(feed.Models))
	for index := range feed.Models {
		model := &feed.Models[index]
		model.legsPresent = true
		for _, leg := range RoutingPriceLegs {
			if _, ok := legs.Models[index][leg]; !ok {
				model.legsPresent = false
			}
		}
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
// model's rates, keyed by its id, values as served. A model whose entry
// omits a leg is left out, so its arms keep their package rates, as the
// service treats an entry with an omitted leg; written as null, the leg
// would instead be filled from the package beside the entry's live legs.
func (feed *RoutingPriceFeed) RequestPrices() *PolicyRoutingPrices {
	rates := make(map[string]PolicyRoutingRate, len(feed.Models))
	for _, model := range feed.Models {
		if !model.legsPresent {
			continue
		}
		rates[model.Model] = PolicyRoutingRate{
			Input: model.Input, CacheRead: model.CacheRead, CacheWrite: model.CacheWrite,
			Output: model.Output, ValidFrom: model.ValidFrom, ValidUntil: model.ValidUntil, Source: model.Source,
		}
	}
	return &PolicyRoutingPrices{
		Schema: RoutingPricesSchema, StoreVersion: feed.StoreVersion, AsOf: feed.AsOf, Rates: rates,
	}
}

// routingPriceTimestampPattern is the policy service's RFC 3339 form
// (service.py _RFC3339): time.Parse alone also accepts a comma before the
// fraction and more than nine fraction digits, which the service refuses as
// an invalid request. The service's datetime also refuses year 0000 and an
// offset of 24 hours or 60 minutes, which time.Parse accepts.
var routingPriceTimestampPattern = regexp.MustCompile(
	`^(\d{4})-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?(Z|[+-](\d{2}):(\d{2}))$`)

func validRoutingPriceTimestamp(value string) error {
	if len(value) > maxRoutingPriceTimestampBytes {
		return errors.New("timestamp too long")
	}
	match := routingPriceTimestampPattern.FindStringSubmatch(value)
	if match == nil || match[1] == "0000" || (match[4] != "" && (match[4] >= "24" || match[5] >= "60")) {
		return fmt.Errorf("timestamp %q is not RFC 3339 with an offset", value)
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
	// A service from before per-leg fill sends no filled_legs: nothing was
	// filled.
	if used.FilledLegs == nil {
		used.FilledLegs = map[string][]string{}
	}
	if len(used.FilledLegs) > maxRoutingPriceModels {
		return errors.New("routing_prices_used fills too many models")
	}
	for model, legs := range used.FilledLegs {
		if model == "" || len(model) > maxRoutingPriceModelIDBytes {
			return errors.New("routing_prices_used filled model id is empty or too long")
		}
		if err := validRoutingPriceLegs(legs); err != nil {
			return fmt.Errorf("routing_prices_used filled_legs: %w", err)
		}
	}
	return nil
}

// RoutingPriceLegs are a rate's legs in the contract's order, the order
// filled_legs lists them in.
var RoutingPriceLegs = [...]string{"input", "cache_read", "cache_write", "output"}

// validRoutingPriceLegs refuses an unknown leg, a repeated one, or legs out
// of the contract's order.
func validRoutingPriceLegs(legs []string) error {
	next := 0
	for _, leg := range legs {
		position := -1
		for index, known := range RoutingPriceLegs {
			if leg == known {
				position = index
				break
			}
		}
		if position < 0 {
			return fmt.Errorf("unknown leg %q", leg)
		}
		if position < next {
			return fmt.Errorf("leg %q repeated or out of order", leg)
		}
		next = position + 1
	}
	return nil
}
