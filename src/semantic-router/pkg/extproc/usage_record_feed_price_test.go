package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// settlementTestFeed lists a tiered model at one endpoint and a model at a
// standard endpoint and its flex twin, as of the fixture clock.
const settlementTestFeed = `{"schema": "arc-routing-prices.v1", "store_version": "119",
	"as_of": "2026-10-14T16:00:00Z", "currency": "USD", "unit": "usd_per_million_tokens", "models": [
	{"model": "openai/gpt-sol", "endpoint_tag": "openai", "input": 5, "cache_read": 0.5, "cache_write": 0,
	 "output": 30, "valid_from": "2026-10-14T00:00:00Z", "valid_until": null, "source": "openrouter:openai",
	 "higher_tier": {"min_prompt_tokens": 272000, "input": 10, "cache_read": 1, "cache_write": 0, "output": 45}},
	{"model": "openai/gpt-luna", "endpoint_tag": "openai", "input": 3, "cache_read": 0.3, "cache_write": 0,
	 "output": 15, "valid_from": null, "valid_until": null, "higher_tier": null},
	{"model": "openai/gpt-luna", "endpoint_tag": "openai/flex", "input": 1.5, "cache_read": 0.15,
	 "cache_write": 0, "output": 7.5, "valid_from": null, "valid_until": null, "higher_tier": null}]}`

// settledARCContext is the request context of a call an ARC decision on a
// live-priced cell dispatched to OpenRouter with body.
func settledARCContext(t *testing.T, trace *selection.RaylineARCTrace, body string) *RequestContext {
	t.Helper()
	ctx := &RequestContext{
		RequestID: "rt_feed-000", RequestModel: "priced-model", VSRRaylineARC: trace,
		DispatchedToOpenRouter: true,
	}
	recordDispatchedProviderPin(ctx, []byte(body))
	ctx.DispatchedOutbound = readDispatchedOutboundParams([]byte(body), nil)
	return ctx
}

func settlementUsage(prompt, cached, completion int) responseUsageMetrics {
	return responseUsageMetrics{
		promptTokens: prompt, promptTokensReported: true,
		cachedPromptTokens: cached, cachedPromptTokensReported: true,
		cacheWriteTokens: 0, cacheWriteTokensReported: true,
		completionTokens: completion, completionTokensReported: true,
		totalTokens: prompt + completion, totalTokensReported: true,
	}
}

func requireCost(t *testing.T, fields map[string]interface{}, want float64) {
	t.Helper()
	if got, _ := fields["cost"].(float64); math.Abs(got-want) > 1e-12 {
		t.Fatalf("cost = %v, want %v (record %v)", fields["cost"], want, fields)
	}
}

func requireFields(t *testing.T, fields map[string]interface{}, want map[string]interface{}) {
	t.Helper()
	for key, value := range want {
		if fields[key] != value {
			t.Fatalf("%s = %v, want %v (record %v)", key, fields[key], value, fields)
		}
	}
}

// An ARC call on a live-priced cell settles at the feed row of the endpoint
// its pin admits, at the tier its whole prompt reaches, and the record names
// the feed, the endpoint and the tier. Uncertain calls settle at the rate
// card and say why; a call with no usage stays unpriced.
func TestUsageRecordSettlesAnARCCallAtTheLiveRoutingPrice(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	digestEcho := strings.Repeat("d", 64)
	fixture.fake.echo = func(request raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
		version, asOf := request.RoutingPrices.StoreVersion, request.RoutingPrices.AsOf
		return &raylinearc.PolicyRoutingPricesUsed{
			Mode: raylinearc.RoutingPricesModeLive, StoreVersion: &version, AsOf: &asOf, Digest: &digestEcho,
			FallbackArms: []string{}, FilledLegs: map[string][]string{},
		}
	}
	clock := fixture.withRoutingPriceFeed(t, true, []byte(settlementTestFeed))
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	trace := result.RaylineARC
	if trace.SettlementPrices == nil {
		t.Fatal("the decision did not carry the routing price cache to settlement")
	}
	router := usageRecordRouter()
	digest := sha256.Sum256([]byte(settlementTestFeed))
	pinned := `{"model": "openai/gpt-sol", "provider": {"only": ["openai"], "allow_fallbacks": false}}`

	// A short prompt is billed at the standard legs, the feed's, not the
	// rate card's (3/0.3/3.75/15).
	logs := captureLogs(t)
	router.reportNonStreamingUsage(settledARCContext(t, trace, pinned), time.Second, settlementUsage(10_000, 4_000, 1_000))
	fields := findLogEvent(t, logs, "llm_usage")
	requireCost(t, fields, (6_000*5+4_000*0.5+1_000*30)/1e6)
	requireFields(t, fields, map[string]interface{}{
		"pricing": usagePricingPriced, "pricing_source": usagePricingSourceFeed, "pricing_feed_fallback": nil,
		"pricing_feed_store_version": "119", "pricing_feed_as_of": "2026-10-14T16:00:00Z",
		"pricing_feed_sha256": hex.EncodeToString(digest[:]), "pricing_endpoint_tag": "openai",
		"price_tier": raylinearc.PriceTierStandard, "currency": "USD",
		"pricing_prompt_per_1m": 5.0, "pricing_cached_input_per_1m": 0.5,
		"pricing_cache_write_per_1m": 0.0, "pricing_completion_per_1m": 30.0,
	})

	// A cache-heavy prompt reaches the tier by its cached tokens: 272k in
	// all, 262k of them read from cache, bills every leg at the tier.
	logs = captureLogs(t)
	router.reportNonStreamingUsage(settledARCContext(t, trace, pinned), time.Second, settlementUsage(272_000, 262_000, 1_000))
	fields = findLogEvent(t, logs, "llm_usage")
	requireCost(t, fields, (10_000*10+262_000*1+1_000*45)/1e6)
	requireFields(t, fields, map[string]interface{}{
		"pricing_source": usagePricingSourceFeed, "price_tier": raylinearc.PriceTierHigher,
		"pricing_prompt_per_1m": 10.0, "pricing_completion_per_1m": 45.0,
	})

	// A base-slug pin admits the standard endpoint and its flex twin: which
	// served is not certain, so the rate card prices it.
	logs = captureLogs(t)
	ambiguous := `{"model": "openai/gpt-luna", "provider": {"only": ["openai"], "allow_fallbacks": false}}`
	router.reportNonStreamingUsage(settledARCContext(t, trace, ambiguous), time.Second, settlementUsage(1_000, 0, 100))
	fields = findLogEvent(t, logs, "llm_usage")
	requireCost(t, fields, (1_000*3+100*15)/1e6)
	requireFields(t, fields, map[string]interface{}{
		"pricing_source": usagePricingSourceConfig, "pricing_feed_fallback": raylinearc.SettlementEndpointAmbiguous,
		"pricing_feed_store_version": nil, "pricing_endpoint_tag": nil, "price_tier": raylinearc.PriceTierStandard,
	})

	// The flex twin's own pin settles at the flex row.
	logs = captureLogs(t)
	flex := `{"model": "openai/gpt-luna", "provider": {"only": ["openai/flex"], "allow_fallbacks": false}}`
	router.reportNonStreamingUsage(settledARCContext(t, trace, flex), time.Second, settlementUsage(1_000, 0, 100))
	fields = findLogEvent(t, logs, "llm_usage")
	requireCost(t, fields, (1_000*1.5+100*7.5)/1e6)
	requireFields(t, fields, map[string]interface{}{"pricing_endpoint_tag": "openai/flex"})

	// A call that stated no usage stays unpriced, whatever the feed says.
	logs = captureLogs(t)
	router.reportNonStreamingUsage(settledARCContext(t, trace, pinned), time.Second, responseUsageMetrics{
		invalid: true, invalidReason: "authoritative_usage_missing",
	})
	fields = findLogEvent(t, logs, "llm_usage")
	requireFields(t, fields, map[string]interface{}{
		"pricing": usagePricingNoUsage, "cost": nil, "price_tier": nil, "pricing_source": usagePricingSourceConfig,
	})

	// Past max age the feed prices nothing.
	clock.advance(10 * time.Minute)
	logs = captureLogs(t)
	router.reportNonStreamingUsage(settledARCContext(t, trace, pinned), time.Second, settlementUsage(10_000, 4_000, 1_000))
	fields = findLogEvent(t, logs, "llm_usage")
	requireCost(t, fields, (6_000*3+4_000*0.3+1_000*15)/1e6)
	requireFields(t, fields, map[string]interface{}{
		"pricing_source": usagePricingSourceConfig, "pricing_feed_fallback": raylinearc.SettlementFeedExpired,
	})
}

// A cell without routing_prices settles exactly as before: the rate card,
// no fallback reason, no feed identity.
func TestUsageRecordWithoutTheFeedSettlesAtTheRateCard(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	if result.RaylineARC.SettlementPrices != nil {
		t.Fatal("a cell without routing_prices carried a price cache")
	}
	logs := captureLogs(t)
	pinned := `{"model": "openai/gpt-sol", "provider": {"only": ["openai"], "allow_fallbacks": false}}`
	usageRecordRouter().reportNonStreamingUsage(settledARCContext(t, result.RaylineARC, pinned), time.Second,
		settlementUsage(10_000, 4_000, 1_000))
	fields := findLogEvent(t, logs, "llm_usage")
	requireCost(t, fields, (6_000*3+4_000*0.3+1_000*15)/1e6)
	requireFields(t, fields, map[string]interface{}{
		"pricing_source": usagePricingSourceConfig, "pricing_feed_fallback": nil, "pricing_feed_as_of": nil,
		"pricing_feed_sha256": nil, "pricing_endpoint_tag": nil, "price_tier": raylinearc.PriceTierStandard,
	})
}
