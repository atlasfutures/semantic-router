package extproc

// Settlement at the live routing price: the thin hook between a usage
// record and raylinearc's settlement index. Which row and tier price a call
// is decided there; this reads what the request context knows about where
// the call was served.

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Values of llmUsageRecord.PricingSource.
const (
	usagePricingSourceFeed   = "feed"
	usagePricingSourceConfig = "config"
)

// usageFeedNotOpenRouter is the fallback reason of a call dispatched past
// OpenRouter: the feed prices OpenRouter endpoints only.
const usageFeedNotOpenRouter = "not_openrouter"

// feedSettlementPrice is the live routing price of the call, or the reason
// the feed did not price it. Both are empty when the feed is not in play:
// no ARC decision carried it, or routing_prices is not configured.
func feedSettlementPrice(ctx *RequestContext, upstreamProvider *string, usage responseUsageMetrics) (*raylinearc.SettlementPrice, string) {
	if ctx == nil || ctx.VSRRaylineARC == nil || ctx.VSRRaylineARC.SettlementPrices == nil {
		return nil, ""
	}
	cache := ctx.VSRRaylineARC.SettlementPrices
	settlement, reason := cache.Settlement()
	if settlement == nil {
		return nil, reason
	}
	if !ctx.DispatchedToOpenRouter {
		return nil, usageFeedNotOpenRouter
	}
	call := raylinearc.SettlementCall{
		PromptTokens: modelpricing.Normalize(modelPricingUsage(usage)).PromptTokens,
		Now:          cache.Now(),
	}
	if ctx.DispatchedOutbound != nil {
		call.Model, call.Pin = ctx.DispatchedOutbound.Model, ctx.DispatchedOutbound.ProviderPin
	}
	// The standard re-send of a flex call is the same model on the twin's
	// pin, which replaced the dispatched body's.
	if ctx.FlexResend.served() {
		call.Pin = settlementPinFromPreferences(ctx.FlexResend.twinPin)
	}
	if upstreamProvider != nil {
		call.UpstreamProvider = *upstreamProvider
	}
	price, reason := settlement.Price(call)
	if reason != "" {
		return nil, reason
	}
	return &price, ""
}

func feedSettlementRates(price *raylinearc.SettlementPrice) modelpricing.Rates {
	cacheWrite := price.Legs.CacheWrite
	return modelpricing.Rates{
		Currency:         "USD",
		PromptPer1M:      price.Legs.Input,
		CachedInputPer1M: price.Legs.CacheRead,
		CacheWritePer1M:  &cacheWrite,
		CompletionPer1M:  price.Legs.Output,
	}
}

// recordFeedPrice names the feed and row that priced the call.
func recordFeedPrice(record *llmUsageRecord, price *raylinearc.SettlementPrice) {
	record.PricingSource = usagePricingSourceFeed
	record.PricingFeedStoreVersion = nonEmpty(price.StoreVersion)
	record.PricingFeedAsOf = nonEmpty(price.AsOf)
	record.PricingFeedSHA256 = nonEmpty(price.FeedSHA256)
	record.PricingEndpointTag = nonEmpty(price.EndpointTag)
	record.PriceTier = nonEmpty(price.Tier)
}

// settlementPinFromPreferences is a provider object as settlement reads it,
// its lists copied.
func settlementPinFromPreferences(preferences *config.OpenRouterProviderPreferences) *raylinearc.SettlementPin {
	if preferences == nil {
		return nil
	}
	return &raylinearc.SettlementPin{
		Order:          append([]string(nil), preferences.Order...),
		Only:           append([]string(nil), preferences.Only...),
		Ignore:         append([]string(nil), preferences.Ignore...),
		AllowFallbacks: preferences.AllowFallbacks,
	}
}
