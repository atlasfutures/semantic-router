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

// Settlement at the live routing price: which feed row prices one served
// call, and at which of its tiers. The decide request forwards the feed's
// base rates by rate key; settling a call needs more, since the call was
// served at one endpoint and one prompt size. A call is priced from the feed
// only when its row is certain; otherwise the caller prices it from the
// cell's configured rates and records the reason named here.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"strings"
	"time"
)

// Why the feed did not price a call. A closed set: it is written on every
// usage record the feed could have priced and did not.
const (
	SettlementFeedAbsent        = "feed_absent"
	SettlementFeedExpired       = "feed_expired"
	SettlementModelUnknown      = "model_unknown"
	SettlementModelNotInFeed    = "model_not_in_feed"
	SettlementEndpointUnpinned  = "endpoint_unpinned"
	SettlementEndpointNotInFeed = "endpoint_not_in_feed"
	SettlementEndpointAmbiguous = "endpoint_ambiguous"
	SettlementRowInvalid        = "row_invalid"
	SettlementRowNotInEffect    = "row_not_in_effect"
)

// The tier a call was billed at.
const (
	PriceTierStandard = "standard"
	PriceTierHigher   = "higher"
)

// SettlementLegs are one tier's rates in USD per million tokens.
type SettlementLegs struct {
	Input      float64
	CacheRead  float64
	CacheWrite float64
	Output     float64
}

// SettlementPin is the OpenRouter provider object the dispatched body
// carried: what decides which endpoints may have served the call.
type SettlementPin struct {
	Order          []string
	Only           []string
	Ignore         []string
	AllowFallbacks *bool
}

// SettlementCall is what a served call says about where it was served.
type SettlementCall struct {
	// Model is the OpenRouter id the dispatched body named.
	Model string
	Pin   *SettlementPin
	// UpstreamProvider is the provider the response named, if any.
	UpstreamProvider string
	// PromptTokens is the whole prompt: uncached, cache read and cache
	// write together.
	PromptTokens int
	Now          time.Time
}

// SettlementPrice is the feed's price for one call and the identity of the
// feed and row it came from.
type SettlementPrice struct {
	Legs         SettlementLegs
	EndpointTag  string
	Tier         string
	StoreVersion string
	AsOf         string
	FeedSHA256   string
}

// RoutingPriceSettlement is one fetched feed, indexed for settlement.
type RoutingPriceSettlement struct {
	StoreVersion string
	AsOf         string
	// FeedSHA256 is the digest of the feed body as served.
	FeedSHA256 string
	asOf       time.Time
	rows       map[string][]settlementRow
}

type settlementRow struct {
	endpointTag string
	legs        SettlementLegs
	higher      *settlementTier
	validFrom   *time.Time
	validUntil  *time.Time
	// invalid marks a row that cannot price any call: a null or negative
	// leg, or a long-context tier that is not whole.
	invalid bool
}

type settlementTier struct {
	minPromptTokens int
	legs            SettlementLegs
}

// newRoutingPriceSettlement indexes a decoded feed by model. body is the
// feed as served, for its digest.
func newRoutingPriceSettlement(feed *RoutingPriceFeed, body []byte) *RoutingPriceSettlement {
	digest := sha256.Sum256(body)
	settlement := &RoutingPriceSettlement{
		StoreVersion: feed.StoreVersion,
		AsOf:         feed.AsOf,
		FeedSHA256:   hex.EncodeToString(digest[:]),
		rows:         make(map[string][]settlementRow, len(feed.Models)),
	}
	// DecodeRoutingPriceFeed has validated as_of.
	settlement.asOf, _ = time.Parse(time.RFC3339Nano, feed.AsOf)
	for _, model := range feed.Models {
		settlement.rows[model.Model] = append(settlement.rows[model.Model], newSettlementRow(model))
	}
	return settlement
}

func newSettlementRow(model RoutingPriceFeedModel) settlementRow {
	row := settlementRow{}
	if model.EndpointTag != nil {
		row.endpointTag = strings.ToLower(*model.EndpointTag)
	}
	legs, ok := settlementLegs(model.Input, model.CacheRead, model.CacheWrite, model.Output)
	row.legs = legs
	row.invalid = !ok
	if model.ValidFrom != nil {
		from, err := time.Parse(time.RFC3339Nano, *model.ValidFrom)
		if err != nil {
			row.invalid = true
		}
		row.validFrom = &from
	}
	if model.ValidUntil != nil {
		// DecodeRoutingPriceFeed has validated valid_until.
		until, _ := time.Parse(time.RFC3339Nano, *model.ValidUntil)
		row.validUntil = &until
	}
	higher, ok := decodeSettlementTier(model.HigherTier)
	row.higher = higher
	if !ok {
		row.invalid = true
	}
	return row
}

// decodeSettlementTier reads a row's higher_tier: absent or null is a
// flat-rate row; anything else must be a whole tier, a positive integer
// min_prompt_tokens and four non-null, non-negative legs.
func decodeSettlementTier(raw json.RawMessage) (*settlementTier, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, true
	}
	var wire struct {
		MinPromptTokens *float64 `json:"min_prompt_tokens"`
		Input           *float64 `json:"input"`
		CacheRead       *float64 `json:"cache_read"`
		CacheWrite      *float64 `json:"cache_write"`
		Output          *float64 `json:"output"`
	}
	if json.Unmarshal(raw, &wire) != nil || wire.MinPromptTokens == nil {
		return nil, false
	}
	threshold := *wire.MinPromptTokens
	if threshold < 1 || threshold != math.Trunc(threshold) || threshold > float64(math.MaxInt32) {
		return nil, false
	}
	legs, ok := settlementLegs(wire.Input, wire.CacheRead, wire.CacheWrite, wire.Output)
	if !ok {
		return nil, false
	}
	return &settlementTier{minPromptTokens: int(threshold), legs: legs}, true
}

func settlementLegs(input, cacheRead, cacheWrite, output *float64) (SettlementLegs, bool) {
	for _, leg := range []*float64{input, cacheRead, cacheWrite, output} {
		if leg == nil || *leg < 0 {
			return SettlementLegs{}, false
		}
	}
	return SettlementLegs{Input: *input, CacheRead: *cacheRead, CacheWrite: *cacheWrite, Output: *output}, true
}

// Price returns the feed's price for the call, or the reason it has none.
func (settlement *RoutingPriceSettlement) Price(call SettlementCall) (SettlementPrice, string) {
	if call.Model == "" {
		return SettlementPrice{}, SettlementModelUnknown
	}
	rows := settlement.rows[call.Model]
	if len(rows) == 0 {
		return SettlementPrice{}, SettlementModelNotInFeed
	}
	row, reason := resolveSettlementRow(rows, call)
	if row == nil {
		return SettlementPrice{}, reason
	}
	if row.invalid {
		return SettlementPrice{}, SettlementRowInvalid
	}
	if (row.validFrom != nil && call.Now.Before(*row.validFrom)) ||
		(row.validUntil != nil && !call.Now.Before(*row.validUntil)) {
		return SettlementPrice{}, SettlementRowNotInEffect
	}
	price := SettlementPrice{
		Legs: row.legs, EndpointTag: row.endpointTag, Tier: PriceTierStandard,
		StoreVersion: settlement.StoreVersion, AsOf: settlement.AsOf, FeedSHA256: settlement.FeedSHA256,
	}
	// The tier bills prompts of min_prompt_tokens or more, whole: once a
	// call reaches it, every leg is the tier's.
	if row.higher != nil && call.PromptTokens >= row.higher.minPromptTokens {
		price.Legs, price.Tier = row.higher.legs, PriceTierHigher
	}
	return price, ""
}

// resolveSettlementRow picks the row of the endpoint that served the call.
// The pin must bound the endpoints (only, or order without fallbacks); a
// pin entry admits its own tag and the tag's variants ("openai" admits
// "openai/flex"), as OpenRouter matches provider slugs. One entry admitting
// one row is that row; otherwise the response's provider must name exactly
// one admitted row.
func resolveSettlementRow(rows []settlementRow, call SettlementCall) (*settlementRow, string) {
	entries, ignored := call.Pin.admittedEntries()
	if len(entries) == 0 {
		return nil, SettlementEndpointUnpinned
	}
	var candidates []*settlementRow
	for index := range rows {
		tag := rows[index].endpointTag
		if tag != "" && pinAdmits(entries, tag) && !pinAdmits(ignored, tag) {
			candidates = append(candidates, &rows[index])
		}
	}
	switch {
	case len(candidates) == 0:
		return nil, SettlementEndpointNotInFeed
	case len(candidates) == 1 && len(entries) == 1:
		return candidates[0], ""
	}
	provider := providerSlugKey(call.UpstreamProvider)
	var named *settlementRow
	for _, candidate := range candidates {
		base, _, _ := strings.Cut(candidate.endpointTag, "/")
		if provider != "" && providerSlugKey(base) == provider {
			if named != nil {
				return nil, SettlementEndpointAmbiguous
			}
			named = candidate
		}
	}
	if named == nil {
		return nil, SettlementEndpointAmbiguous
	}
	return named, ""
}

// admittedEntries is the pin's bounded endpoint list and its ignore list,
// lower-cased; no entries when the pin lets OpenRouter route anywhere.
func (pin *SettlementPin) admittedEntries() ([]string, []string) {
	if pin == nil {
		return nil, nil
	}
	var entries []string
	switch {
	case len(pin.Only) > 0:
		entries = pin.Only
	case pin.AllowFallbacks != nil && !*pin.AllowFallbacks && len(pin.Order) > 0:
		entries = pin.Order
	default:
		return nil, nil
	}
	return uniqueLowerSlugs(entries), uniqueLowerSlugs(pin.Ignore)
}

func uniqueLowerSlugs(slugs []string) []string {
	seen := make(map[string]bool, len(slugs))
	unique := make([]string, 0, len(slugs))
	for _, slug := range slugs {
		slug = strings.ToLower(strings.TrimSpace(slug))
		if slug != "" && !seen[slug] {
			seen[slug] = true
			unique = append(unique, slug)
		}
	}
	return unique
}

func pinAdmits(entries []string, tag string) bool {
	for _, entry := range entries {
		if tag == entry || strings.HasPrefix(tag, entry+"/") {
			return true
		}
	}
	return false
}

// providerSlugKey folds a provider's display name ("Google AI Studio") and
// its slug ("google-ai-studio") to one key: lower case, letters and digits.
func providerSlugKey(name string) string {
	var key strings.Builder
	for _, r := range strings.ToLower(name) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			key.WriteRune(r)
		}
	}
	return key.String()
}
