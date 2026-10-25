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
// rates are forwarded as they are, keyed by rate key (RoutingPriceRateKey);
// this router never maps them to trained arms, which is the service's job.

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
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
	// RoutingPricesModeCaller is a decision scored in a Decisions caller's
	// own price world (routing_prices.basis: caller, pathfinder#5668).
	RoutingPricesModeCaller = "caller"
	// RoutingPricesBasisCaller marks a request's routing_prices as the
	// caller's own: exactly four legs per model, applied as sent, never
	// filled from the package.
	RoutingPricesBasisCaller = "caller"

	maxRoutingPriceStoreVersionBytes = 256
	maxRoutingPriceTimestampBytes    = 64
	maxRoutingPriceModelIDBytes      = 256
	maxRoutingPriceEndpointTagBytes  = 128
	// MaxRoutingPriceRateKeyBytes is the longest rate key the request and
	// the echo may name: a model id, or a model id and endpoint tag joined
	// by RoutingPriceEndpointSeparator.
	MaxRoutingPriceRateKeyBytes = maxRoutingPriceModelIDBytes + len(RoutingPriceEndpointSeparator) +
		maxRoutingPriceEndpointTagBytes
	maxRoutingPriceModels = 4096
	// RoutingPriceEndpointSeparator joins a model id and an endpoint tag in
	// a tagged rate key. No OpenRouter id contains it, and the feed may not
	// name one that does, so a tagged key never equals a model id.
	RoutingPriceEndpointSeparator = "@"
	// maxRoutingPriceFallbackArms is only a safety bound: the service lists
	// every arm that fell back, unbounded, and a package can price over a
	// thousand arms, most of which fall back when the feed covers few of
	// their models. That must degrade to package prices, not fail the
	// decode. The ledger and the log keep only the first few and the count.
	maxRoutingPriceFallbackArms = 65_536
)

// RoutingPriceFeedModel is one model endpoint's effective rates at the
// feed's as_of, in USD per million tokens. A leg the source does not know is
// null. An entry is one (model, endpoint_tag): the OpenRouter id and the
// OpenRouter endpoint the rates are read at, null from a feed that predates
// the tag (memex-desktop#7457).
type RoutingPriceFeedModel struct {
	Model       string   `json:"model"`
	EndpointTag *string  `json:"endpoint_tag"`
	Input       *float64 `json:"input"`
	CacheRead   *float64 `json:"cache_read"`
	CacheWrite  *float64 `json:"cache_write"`
	Output      *float64 `json:"output"`
	ValidFrom   *string  `json:"valid_from"`
	ValidUntil  *string  `json:"valid_until"`
	Source      *string  `json:"source"`
	// HigherTier is the long-prompt tier, null for a flat-rate model. It is
	// read only to settle a call (routing_prices_settlement.go), never
	// forwarded, and kept raw so a malformed tier refuses only its own row
	// at settlement rather than the whole feed.
	HigherTier json.RawMessage `json:"higher_tier"`
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
	// Basis is "caller" for a Decisions caller's own prices (pathfinder
	// #5668), "" for the feed's. A caller's rates are sent as exactly their
	// four legs, and Exclusive with them.
	Basis     string `json:"-"`
	Exclusive bool   `json:"-"`
}

type policyRoutingPricesFeedWire struct {
	Schema       string                       `json:"schema"`
	StoreVersion string                       `json:"store_version"`
	AsOf         string                       `json:"as_of"`
	Rates        map[string]PolicyRoutingRate `json:"rates"`
}

type policyCallerRateWire struct {
	Input      *float64 `json:"input"`
	CacheRead  *float64 `json:"cache_read"`
	CacheWrite *float64 `json:"cache_write"`
	Output     *float64 `json:"output"`
}

// UnmarshalJSON reads either form.
func (prices *PolicyRoutingPrices) UnmarshalJSON(body []byte) error {
	var wire struct {
		policyRoutingPricesFeedWire
		Basis     string `json:"basis"`
		Exclusive *bool  `json:"exclusive"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		return err
	}
	*prices = PolicyRoutingPrices{
		Schema: wire.Schema, StoreVersion: wire.StoreVersion, AsOf: wire.AsOf, Rates: wire.Rates, Basis: wire.Basis,
	}
	if wire.Exclusive != nil {
		prices.Exclusive = *wire.Exclusive
	}
	return nil
}

// MarshalJSON writes the feed's rates unchanged, and a caller's as exactly
// four legs beside basis and exclusive: the service refuses any other
// member of a caller's rate.
func (prices PolicyRoutingPrices) MarshalJSON() ([]byte, error) {
	if prices.Basis == "" {
		return json.Marshal(policyRoutingPricesFeedWire{
			Schema: prices.Schema, StoreVersion: prices.StoreVersion, AsOf: prices.AsOf, Rates: prices.Rates,
		})
	}
	rates := make(map[string]policyCallerRateWire, len(prices.Rates))
	for model, rate := range prices.Rates {
		rates[model] = policyCallerRateWire{Input: rate.Input, CacheRead: rate.CacheRead, CacheWrite: rate.CacheWrite, Output: rate.Output}
	}
	return json.Marshal(struct {
		Schema       string                          `json:"schema"`
		StoreVersion string                          `json:"store_version"`
		AsOf         string                          `json:"as_of"`
		Rates        map[string]policyCallerRateWire `json:"rates"`
		Basis        string                          `json:"basis"`
		Exclusive    bool                            `json:"exclusive"`
	}{prices.Schema, prices.StoreVersion, prices.AsOf, rates, prices.Basis, prices.Exclusive})
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
	// PricesSHA256, Exclusive and ExcludedArms are present exactly in
	// caller mode (pathfinder#5668): the caller world's public price_sha256,
	// its exclusivity, and the arms it left out.
	PricesSHA256 *string  `json:"prices_sha256,omitempty"`
	Exclusive    *bool    `json:"exclusive,omitempty"`
	ExcludedArms []string `json:"excluded_arms,omitempty"`
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
		if strings.Contains(model.Model, RoutingPriceEndpointSeparator) {
			return nil, fmt.Errorf("routing price feed model id %q contains %q", model.Model, RoutingPriceEndpointSeparator)
		}
		if model.EndpointTag != nil && (*model.EndpointTag == "" || len(*model.EndpointTag) > maxRoutingPriceEndpointTagBytes) {
			return nil, errors.New("routing price feed endpoint_tag is empty or too long")
		}
		endpoint := model.Model + RoutingPriceEndpointSeparator
		if model.EndpointTag != nil {
			endpoint += *model.EndpointTag
		}
		if seen[endpoint] {
			return nil, fmt.Errorf("routing price feed lists model %q at one endpoint twice", model.Model)
		}
		seen[endpoint] = true
		if model.ValidUntil != nil {
			if err := validRoutingPriceTimestamp(*model.ValidUntil); err != nil {
				return nil, fmt.Errorf("routing price feed valid_until: %w", err)
			}
		}
	}
	// A model the feed lists at several endpoints is keyed by endpoint, so
	// every one of its entries must name its endpoint.
	endpoints := feed.endpointCounts()
	for _, model := range feed.Models {
		if endpoints[model.Model] > 1 && model.EndpointTag == nil {
			return nil, fmt.Errorf("routing price feed lists model %q at several endpoints, one untagged", model.Model)
		}
	}
	return &feed, nil
}

func (feed *RoutingPriceFeed) endpointCounts() map[string]int {
	counts := make(map[string]int, len(feed.Models))
	for _, model := range feed.Models {
		counts[model.Model]++
	}
	return counts
}

// RoutingPriceRateKey is the rate key of a model at an endpoint: the model
// id for a model the feed lists at one endpoint, which is every model a
// feed from before endpoint tags lists, and "<model>@<endpoint_tag>" for
// each endpoint of a model it lists at several (a standard endpoint and its
// flex twin). The policy service looks an arm up by its worker's endpoint
// key first and its model id second (docs/arc_serving_contract.md).
func RoutingPriceRateKey(model string, endpointTag string) string {
	return model + RoutingPriceEndpointSeparator + endpointTag
}

// RequestPrices is the feed as the decision request's routing_prices: every
// entry's rates, keyed by its rate key (RoutingPriceRateKey), values as
// served. An entry that omits a leg is left out, so its arms keep their
// package rates, as the service treats an entry with an omitted leg;
// written as null, the leg would instead be filled from the package beside
// the entry's live legs.
func (feed *RoutingPriceFeed) RequestPrices() *PolicyRoutingPrices {
	rates := make(map[string]PolicyRoutingRate, len(feed.Models))
	endpoints := feed.endpointCounts()
	for _, model := range feed.Models {
		if !model.legsPresent {
			continue
		}
		key := model.Model
		if endpoints[model.Model] > 1 {
			if model.EndpointTag == nil {
				continue // DecodeRoutingPriceFeed refuses such a feed
			}
			key = RoutingPriceRateKey(model.Model, *model.EndpointTag)
		}
		rates[key] = PolicyRoutingRate{
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
	if used.Mode != RoutingPricesModeLive && used.Mode != RoutingPricesModePackage && used.Mode != RoutingPricesModeCaller {
		return fmt.Errorf("routing_prices_used mode %q", used.Mode)
	}
	caller := used.PricesSHA256 != nil || used.Exclusive != nil || used.ExcludedArms != nil
	if (used.Mode == RoutingPricesModeCaller) != caller ||
		used.Mode == RoutingPricesModeCaller && (used.PricesSHA256 == nil || !isLowerHex64(*used.PricesSHA256) || used.Exclusive == nil) {
		return errors.New("routing_prices_used caller fields come with, and only with, caller mode")
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
		if model == "" || len(model) > MaxRoutingPriceRateKeyBytes {
			return errors.New("routing_prices_used filled rate key is empty or too long")
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
