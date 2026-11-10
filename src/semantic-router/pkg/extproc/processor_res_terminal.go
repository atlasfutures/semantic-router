package extproc

import "github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

// takeNeutralResponseUsage consumes provider-reported usage from the response
// already decoded by the neutral protocol engine. Estimated request tokens are
// never substituted for missing terminal evidence.
func (r *OpenAIRouter) takeNeutralResponseUsage(ctx *RequestContext) responseUsageMetrics {
	response := attributedResponse(ctx)
	if r == nil || ctx == nil || response == nil {
		return invalidResponseTerminalUsage("authoritative_usage_missing")
	}
	usage := responseUsageFromSemanticUsage(response.Usage)
	if ctx.ContextRecoveryProviderCost != nil {
		// The body the usage was decoded from was re-encoded for the
		// client, and no client encoding carries the charge.
		usage.providerCost = *ctx.ContextRecoveryProviderCost
	}
	return usage
}

func invalidResponseTerminalUsage(reason string) responseUsageMetrics {
	return responseUsageMetrics{invalid: true, invalidReason: reason}
}

func responseUsageFromSemanticUsage(usage llmprotocol.Usage) responseUsageMetrics {
	if usage.State != llmprotocol.UsageAvailable {
		invalid := invalidResponseTerminalUsage("authoritative_usage_invalid")
		invalid.providerCost = usage.ProviderCost
		return invalid
	}
	input, inputReported, inputOK := authoritativeTerminalTokenCount(usage.InputTotal)
	output, outputReported, outputOK := authoritativeTerminalTokenCount(usage.OutputTotal)
	total, totalReported, totalOK := settlementTotalTokenCount(usage, input, output)
	cacheRead, cacheReadReported, cacheReadOK := authoritativeTerminalTokenCount(usage.InputCacheRead)
	cacheWrite, cacheWriteReported, cacheWriteOK := authoritativeTerminalTokenCount(usage.InputCacheWrite)
	if !inputOK || !outputOK || !totalOK || !cacheReadOK || !cacheWriteOK ||
		!inputReported || !outputReported {
		invalid := invalidResponseTerminalUsage("response_terminal_invalid")
		invalid.providerCost = usage.ProviderCost
		return invalid
	}
	return normalizeResponseUsage(responseUsageMetrics{
		providerCost: usage.ProviderCost,
		promptTokens: input, promptTokensReported: inputReported,
		completionTokens: output, completionTokensReported: outputReported,
		totalTokens: total, totalTokensReported: totalReported,
		cachedPromptTokens: cacheRead, cachedPromptTokensReported: cacheReadReported,
		cacheWriteTokens: cacheWrite, cacheWriteTokensReported: cacheWriteReported,
	})
}

func authoritativeTerminalTokenCount(count llmprotocol.TokenCount) (int, bool, bool) {
	if count.Value == nil {
		return 0, false, count.Provenance == "" || count.Provenance == llmprotocol.UsageUnknown
	}
	if count.Provenance != llmprotocol.UsageAuthoritative {
		return 0, true, false
	}
	value := *count.Value
	maximum := int64(^uint(0) >> 1)
	if value < 0 || value > maximum {
		return 0, true, false
	}
	return int(value), true, true
}

func settlementTotalTokenCount(usage llmprotocol.Usage, input, output int) (int, bool, bool) {
	if usage.Total.Value == nil {
		if input > int(^uint(0)>>1)-output {
			return 0, false, false
		}
		return input + output, true, true
	}
	value := *usage.Total.Value
	maximum := int64(^uint(0) >> 1)
	if value < 0 || value > maximum {
		return 0, true, false
	}
	switch usage.Total.Provenance {
	case llmprotocol.UsageAuthoritative:
		return int(value), true, true
	case llmprotocol.UsageDerived:
		if input > int(^uint(0)>>1)-output || int(value) != input+output {
			return 0, true, false
		}
		return int(value), true, true
	default:
		return 0, true, false
	}
}

// responseUsageFromRestatedUsage reads a count a cut turn's terminal restated.
//
// responseUsageFromSemanticUsage admits only authoritative counts, because the
// row it builds is a settlement. A truncated terminal's count is not one: its
// output leg is the larger of what the upstream stated and what the Router
// estimated (reconciledTruncationUsage), and carries estimated provenance to
// say so. The row takes the values as they are and says it is an estimate.
func responseUsageFromRestatedUsage(usage llmprotocol.Usage) responseUsageMetrics {
	if usage.State != llmprotocol.UsageAvailable || usage.InputTotal.Value == nil || usage.OutputTotal.Value == nil {
		invalid := invalidResponseTerminalUsage("restated_usage_incomplete")
		invalid.providerCost = usage.ProviderCost
		return invalid
	}
	count := func(value llmprotocol.TokenCount) (int, bool) {
		if value.Value == nil {
			return 0, false
		}
		return int(*value.Value), true
	}
	prompt, _ := count(usage.InputTotal)
	completion, _ := count(usage.OutputTotal)
	cacheRead, cacheReadReported := count(usage.InputCacheRead)
	cacheWrite, cacheWriteReported := count(usage.InputCacheWrite)
	return normalizeResponseUsage(responseUsageMetrics{
		providerCost: usage.ProviderCost,
		promptTokens: prompt, promptTokensReported: true,
		completionTokens: completion, completionTokensReported: true,
		totalTokens: prompt + completion, totalTokensReported: true,
		cachedPromptTokens: cacheRead, cachedPromptTokensReported: cacheReadReported,
		cacheWriteTokens: cacheWrite, cacheWriteTokensReported: cacheWriteReported,
		estimated: true,
	})
}
