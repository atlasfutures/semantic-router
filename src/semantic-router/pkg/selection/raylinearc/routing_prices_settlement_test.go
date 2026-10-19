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

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
	"time"
)

// settlementFeedBody is a feed in today's shape: a tiered model at one
// endpoint, a model at a standard endpoint and its flex twin, a model at two
// providers, and rows a call cannot be settled at.
const settlementFeedBody = `{"schema": "arc-routing-prices.v1", "store_version": "119",
	"as_of": "2026-10-19T12:00:00.000Z", "currency": "USD", "unit": "usd_per_million_tokens", "models": [
	{"model": "openai/gpt-sol", "endpoint_tag": "openai", "input": 5, "cache_read": 0.5, "cache_write": 0,
	 "output": 30, "valid_from": "2026-10-18T00:00:00.000Z", "valid_until": null, "source": "openrouter:openai",
	 "billing_model": "gpt-sol",
	 "higher_tier": {"min_prompt_tokens": 272000, "input": 10, "cache_read": 1, "cache_write": 0, "output": 45}},
	{"model": "openai/gpt-luna", "endpoint_tag": "openai", "input": 3, "cache_read": 0.3, "cache_write": 0,
	 "output": 15, "valid_from": null, "valid_until": null, "source": "openrouter:openai", "higher_tier": null},
	{"model": "openai/gpt-luna", "endpoint_tag": "openai/flex", "input": 1.5, "cache_read": 0.15, "cache_write": 0,
	 "output": 7.5, "valid_from": null, "valid_until": null, "source": "openrouter:openai/flex", "higher_tier": null},
	{"model": "moonshotai/kimi", "endpoint_tag": "moonshotai/int4", "input": 0.6, "cache_read": 0.1,
	 "cache_write": 0, "output": 2.5, "valid_from": null, "valid_until": null, "higher_tier": null},
	{"model": "moonshotai/kimi", "endpoint_tag": "google-ai-studio", "input": 0.7, "cache_read": 0.12,
	 "cache_write": 0, "output": 2.8, "valid_from": null, "valid_until": null, "higher_tier": null},
	{"model": "google/gemini-pro", "endpoint_tag": "google-vertex/global", "input": 2, "cache_read": 0.2,
	 "cache_write": 0.4, "output": 12, "valid_from": null, "valid_until": null,
	 "higher_tier": {"min_prompt_tokens": 200000, "input": 4, "cache_read": 0.4, "cache_write": null, "output": 18}},
	{"model": "z-ai/glm", "endpoint_tag": "z-ai", "input": 0.6, "cache_read": null, "cache_write": null,
	 "output": 2.2, "valid_from": null, "valid_until": null, "higher_tier": null},
	{"model": "x-ai/grok", "endpoint_tag": "xai", "input": 3, "cache_read": 0.75, "cache_write": 0, "output": 15,
	 "valid_from": null, "valid_until": "2026-10-19T13:00:00.000Z", "higher_tier": null}]}`

var settlementNow = time.Date(2026, 10, 19, 12, 30, 0, 0, time.UTC)

func newTestSettlement(t *testing.T) *RoutingPriceSettlement {
	t.Helper()
	feed, err := DecodeRoutingPriceFeed([]byte(settlementFeedBody))
	if err != nil {
		t.Fatal(err)
	}
	return newRoutingPriceSettlement(feed, []byte(settlementFeedBody))
}

func onlyPin(slugs ...string) *SettlementPin {
	return &SettlementPin{Only: slugs}
}

func TestSettlementPricesAPinnedCallAtItsRowAndNamesTheFeed(t *testing.T) {
	settlement := newTestSettlement(t)
	price, reason := settlement.Price(SettlementCall{
		Model: "openai/gpt-sol", Pin: onlyPin("openai"), PromptTokens: 1000, Now: settlementNow,
	})
	if reason != "" {
		t.Fatalf("refused: %s", reason)
	}
	digest := sha256.Sum256([]byte(settlementFeedBody))
	want := SettlementPrice{
		Legs:        SettlementLegs{Input: 5, CacheRead: 0.5, CacheWrite: 0, Output: 30},
		EndpointTag: "openai", Tier: PriceTierStandard, StoreVersion: "119",
		AsOf: "2026-10-19T12:00:00.000Z", FeedSHA256: hex.EncodeToString(digest[:]),
	}
	if price != want {
		t.Fatalf("price %+v, want %+v", price, want)
	}
}

// The tier is chosen per call against the whole prompt, inclusive of the
// threshold: a prompt of exactly min_prompt_tokens is billed at the tier,
// and a cache-heavy prompt counts its cached tokens toward it.
func TestSettlementTierBoundaryIsInclusiveOfTheWholePrompt(t *testing.T) {
	settlement := newTestSettlement(t)
	for _, test := range []struct {
		prompt int
		tier   string
		input  float64
	}{
		{prompt: 271_999, tier: PriceTierStandard, input: 5},
		{prompt: 272_000, tier: PriceTierHigher, input: 10},
		{prompt: 900_000, tier: PriceTierHigher, input: 10},
	} {
		price, reason := settlement.Price(SettlementCall{
			Model: "openai/gpt-sol", Pin: onlyPin("openai"), PromptTokens: test.prompt, Now: settlementNow,
		})
		if reason != "" || price.Tier != test.tier || price.Legs.Input != test.input {
			t.Fatalf("prompt %d: %+v %q, want tier %s input %v", test.prompt, price, reason, test.tier, test.input)
		}
		if test.tier == PriceTierHigher && price.Legs != (SettlementLegs{Input: 10, CacheRead: 1, CacheWrite: 0, Output: 45}) {
			t.Fatalf("higher tier does not bill every leg at the tier: %+v", price.Legs)
		}
	}
}

// A tier that is not whole refuses its row for every call, short prompts
// too: the row cannot say what a long one costs, so it says nothing.
func TestSettlementRefusesARowWithAPartialTier(t *testing.T) {
	settlement := newTestSettlement(t)
	for _, prompt := range []int{10, 250_000} {
		if _, reason := settlement.Price(SettlementCall{
			Model: "google/gemini-pro", Pin: onlyPin("google-vertex/global"), PromptTokens: prompt, Now: settlementNow,
		}); reason != SettlementRowInvalid {
			t.Fatalf("prompt %d: reason %q, want %s", prompt, reason, SettlementRowInvalid)
		}
	}
	if _, reason := settlement.Price(SettlementCall{
		Model: "z-ai/glm", Pin: onlyPin("z-ai"), PromptTokens: 10, Now: settlementNow,
	}); reason != SettlementRowInvalid {
		t.Fatalf("null base legs: reason %q", reason)
	}
}

func TestDecodeSettlementTierRefusesMalformedTiers(t *testing.T) {
	for _, raw := range []string{
		`{"input": 1, "cache_read": 1, "cache_write": 1, "output": 1}`,
		`{"min_prompt_tokens": 0, "input": 1, "cache_read": 1, "cache_write": 1, "output": 1}`,
		`{"min_prompt_tokens": 1000.5, "input": 1, "cache_read": 1, "cache_write": 1, "output": 1}`,
		`{"min_prompt_tokens": "272000", "input": 1, "cache_read": 1, "cache_write": 1, "output": 1}`,
		`{"min_prompt_tokens": 1000, "input": 1, "cache_read": 1, "output": 1}`,
		`{"min_prompt_tokens": 1000, "input": -1, "cache_read": 1, "cache_write": 1, "output": 1}`,
		`[]`,
	} {
		if tier, ok := decodeSettlementTier([]byte(raw)); ok || tier != nil {
			t.Fatalf("tier %s accepted as %+v", raw, tier)
		}
	}
	for _, raw := range []string{``, `null`} {
		if tier, ok := decodeSettlementTier([]byte(raw)); !ok || tier != nil {
			t.Fatalf("flat-rate %q read as %+v %v", raw, tier, ok)
		}
	}
}

// A malformed tier refuses its own row at settlement, not the whole feed
// the decide request is sent.
func TestRoutingPriceFeedWithMalformedTierStillDecodes(t *testing.T) {
	body := strings.Replace(settlementFeedBody, `"min_prompt_tokens": 272000`, `"min_prompt_tokens": "lots"`, 1)
	feed, err := DecodeRoutingPriceFeed([]byte(body))
	if err != nil {
		t.Fatalf("feed refused for one malformed tier: %v", err)
	}
	if _, reason := newRoutingPriceSettlement(feed, []byte(body)).Price(SettlementCall{
		Model: "openai/gpt-sol", Pin: onlyPin("openai"), PromptTokens: 10, Now: settlementNow,
	}); reason != SettlementRowInvalid {
		t.Fatalf("reason %q", reason)
	}
}

func TestSettlementResolvesTheEndpointFromThePin(t *testing.T) {
	settlement := newTestSettlement(t)
	noFallbacks := false
	for _, test := range []struct {
		name     string
		model    string
		pin      *SettlementPin
		provider string
		tag      string
		reason   string
	}{
		{name: "only the flex variant", model: "openai/gpt-luna", pin: onlyPin("openai/flex"), tag: "openai/flex"},
		{
			name: "a base slug pin admits its variants too", model: "openai/gpt-luna", pin: onlyPin("openai"),
			provider: "OpenAI", reason: SettlementEndpointAmbiguous,
		},
		{
			name: "a base slug pin less its ignored variant", model: "openai/gpt-luna",
			pin: &SettlementPin{Order: []string{"openai"}, Ignore: []string{"openai/flex"}, AllowFallbacks: &noFallbacks},
			tag: "openai",
		},
		{
			name: "order with fallbacks is unbounded", model: "openai/gpt-sol",
			pin: &SettlementPin{Order: []string{"openai"}}, reason: SettlementEndpointUnpinned,
		},
		{name: "no pin", model: "openai/gpt-sol", reason: SettlementEndpointUnpinned},
		{name: "a case-insensitive pin", model: "openai/gpt-sol", pin: onlyPin(" OpenAI "), tag: "openai"},
		{
			name: "a pin the feed has no row for", model: "openai/gpt-sol", pin: onlyPin("azure"),
			reason: SettlementEndpointNotInFeed,
		},
		{
			name: "a model the feed does not list", model: "openai/gpt-terra", pin: onlyPin("openai"),
			reason: SettlementModelNotInFeed,
		},
		{name: "no model", pin: onlyPin("openai"), reason: SettlementModelUnknown},
		{
			name: "several endpoints, the response names one", model: "moonshotai/kimi",
			pin: onlyPin("moonshotai/int4", "google-ai-studio"), provider: "Google AI Studio", tag: "google-ai-studio",
		},
		{
			name: "several endpoints, the response names a variant's base", model: "moonshotai/kimi",
			pin: onlyPin("moonshotai/int4", "google-ai-studio"), provider: "MoonshotAI", tag: "moonshotai/int4",
		},
		{
			name: "several endpoints, the response names none of them", model: "moonshotai/kimi",
			pin: onlyPin("moonshotai/int4", "google-ai-studio"), provider: "Together", reason: SettlementEndpointAmbiguous,
		},
		{
			name: "several endpoints, no provider named", model: "moonshotai/kimi",
			pin: onlyPin("moonshotai/int4", "google-ai-studio"), reason: SettlementEndpointAmbiguous,
		},
		// The pin also admits an endpoint the feed does not price, so the
		// one row it does is not certain without the response's word.
		{
			name: "one row, but the pin admits another endpoint", model: "openai/gpt-sol",
			pin: onlyPin("openai", "azure"), provider: "Azure", reason: SettlementEndpointAmbiguous,
		},
		{
			name: "one row, the response names it", model: "openai/gpt-sol",
			pin: onlyPin("openai", "azure"), provider: "OpenAI", tag: "openai",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			price, reason := settlement.Price(SettlementCall{
				Model: test.model, Pin: test.pin, UpstreamProvider: test.provider, PromptTokens: 10, Now: settlementNow,
			})
			if reason != test.reason || price.EndpointTag != test.tag {
				t.Fatalf("got %q %q, want %q %q", price.EndpointTag, reason, test.tag, test.reason)
			}
		})
	}
}

func TestSettlementHonoursTheRowValidityWindow(t *testing.T) {
	settlement := newTestSettlement(t)
	call := SettlementCall{Model: "x-ai/grok", Pin: onlyPin("xai"), PromptTokens: 10, Now: settlementNow}
	if _, reason := settlement.Price(call); reason != "" {
		t.Fatalf("inside the window: %s", reason)
	}
	call.Now = time.Date(2026, 10, 19, 13, 0, 0, 0, time.UTC)
	if _, reason := settlement.Price(call); reason != SettlementRowNotInEffect {
		t.Fatalf("at valid_until: %q", reason)
	}
	call = SettlementCall{
		Model: "openai/gpt-sol", Pin: onlyPin("openai"), PromptTokens: 10,
		Now: time.Date(2026, 10, 17, 0, 0, 0, 0, time.UTC),
	}
	if _, reason := settlement.Price(call); reason != SettlementRowNotInEffect {
		t.Fatalf("before valid_from: %q", reason)
	}
}

// The cache settles at the feed only while it would still send it on a
// decide request: from the first fetch until max age, by its fetch or by
// its own as_of.
func TestRoutingPriceCacheSettlementFollowsMaxAge(t *testing.T) {
	feed := newFakePriceFeed(t, []byte(settlementFeedBody))
	clock := &fakeClock{now: time.Date(2026, 10, 19, 12, 1, 0, 0, time.UTC)}
	cache := newTestPriceCache(feed, clock, &fetchLog{})
	if settlement, reason := cache.Settlement(); settlement != nil || reason != SettlementFeedAbsent {
		t.Fatalf("before any fetch: %v %q", settlement, reason)
	}
	if err := cache.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if settlement, reason := cache.Settlement(); settlement == nil || reason != "" || settlement.StoreVersion != "119" {
		t.Fatalf("fresh: %v %q", settlement, reason)
	}
	// Stale, past the TTL and within max age, still settles.
	clock.advance(5 * time.Minute)
	if settlement, _ := cache.Settlement(); settlement == nil {
		t.Fatal("stale feed not settled at")
	}
	// The feed's as_of (12:00) reaches max age (10m) before its fetch does.
	clock.advance(4 * time.Minute)
	if settlement, reason := cache.Settlement(); settlement != nil || reason != SettlementFeedExpired {
		t.Fatalf("as_of past max age: %v %q", settlement, reason)
	}
	clock.advance(time.Hour)
	if settlement, reason := cache.Settlement(); settlement != nil || reason != SettlementFeedExpired {
		t.Fatalf("fetch past max age: %v %q", settlement, reason)
	}
}
