package extproc

import "time"

func (r *OpenAIRouter) selectionOutcomeForUsage(
	ctx *RequestContext,
	usage responseUsageMetrics,
	latency time.Duration,
	outcomeClass string,
) selectionActualOutcome {
	outcome := selectionActualOutcome{
		OutcomeClass: outcomeClass,
	}
	if ctx != nil {
		outcome.StatusCode = ctx.UpstreamStatusCode
	}
	addSelectionUsage(&outcome, usage)
	addSelectionLatency(&outcome, ctx, latency)
	r.addSelectionCost(&outcome, ctx, usage)
	return outcome
}

func addSelectionUsage(
	outcome *selectionActualOutcome,
	usage responseUsageMetrics,
) {
	if usage.promptTokensReported {
		value := usage.promptTokens
		outcome.InputTokens = &value
	}
	if usage.completionTokensReported {
		value := usage.completionTokens
		outcome.OutputTokens = &value
	}
}

func addSelectionLatency(
	outcome *selectionActualOutcome,
	ctx *RequestContext,
	latency time.Duration,
) {
	if ctx != nil && !ctx.StartTime.IsZero() && latency >= 0 {
		value := float64(latency.Microseconds()) / 1_000
		outcome.LatencyMS = &value
	}
}

func (r *OpenAIRouter) addSelectionCost(
	outcome *selectionActualOutcome,
	ctx *RequestContext,
	usage responseUsageMetrics,
) {
	if ctx != nil && r != nil && r.Config != nil &&
		(usage.promptTokensReported ||
			usage.completionTokensReported) {
		if pricing, ok := r.Config.GetFullModelPricing(
			ctx.RequestModel,
		); ok {
			value := costForResponseUsage(usage, pricing)
			outcome.CostUSD = &value
		}
	}
}
