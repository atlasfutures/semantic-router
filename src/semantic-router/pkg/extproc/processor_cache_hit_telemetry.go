package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func (r *OpenAIRouter) reportCacheHitTelemetry(
	ctx *RequestContext,
	cachedBody []byte,
	lookupLatency time.Duration,
) {
	if r == nil || ctx == nil {
		return
	}
	semanticResponse, err := r.decodeClientResponse(cachedBody, ctx)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "cache_hit_response_decode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		return
	}
	usage := responseUsageFromSemanticUsage(semanticResponse.Usage)
	totalTokens := responseUsageTotal(usage)
	if totalTokens > 0 {
		recordSessionTurn(ctx, usage, sessiontelemetry.TurnPricing{})
	}
	if ctx.RequestModel != "" {
		recordModelUsageTokens(ctx.RequestModel, usage)
		metrics.RecordModelCompletionLatency(
			ctx.RequestModel,
			lookupLatency.Seconds(),
		)
	}

	replayUsage := r.buildCacheHitReplayUsage(ctx, usage)
	r.updateRouterReplayUsageCost(ctx, replayUsage)
	r.observeRouterLearningUsageTelemetry(
		ctx,
		lookupLatency,
		usage,
		replayUsage,
	)
	// A cached body can still hold the charge its original call reported.
	// No provider was called for this turn, so nothing was charged for it,
	// and replaying the original charge would count it once per hit.
	usage.providerCost = llmprotocol.ProviderCost{}
	record := r.newLLMUsageRecord(ctx, usage)
	latencyMillis := lookupLatency.Milliseconds()
	record.CompletionLatencyMS = &latencyMillis
	// No upstream was called, so the turn's upstream cost is a known zero;
	// the provider charge stays null because no provider stated one.
	zero := 0.0
	record.Pricing = usagePricingCacheHit
	record.Cost = &zero
	record.FromCache, record.CacheHit = true, true
	// A cache hit never reaches the response headers, so the request's
	// declaration is the observed truth: the router is the thing streaming,
	// and createCacheHitResponse re-encodes the cached body as SSE exactly
	// when the request asked for it.
	record.Streaming = ctx.ExpectStreamingResponse
	emitLLMUsageRecord(record)
}

func (r *OpenAIRouter) buildCacheHitReplayUsage(
	ctx *RequestContext,
	usage responseUsageMetrics,
) routerreplay.UsageCost {
	replayUsage := r.buildReplayUsageCost(ctx, usage)
	replayUsage.ActualCost = replayFloat64Ptr(0)
	if replayUsage.BaselineCost != nil {
		replayUsage.CostSavings = replayFloat64Ptr(*replayUsage.BaselineCost)
	}
	return replayUsage
}
