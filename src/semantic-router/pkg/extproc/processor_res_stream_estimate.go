package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// What a truncated turn cost, when the upstream never said.
//
// A stream cut before its usage frame leaves no provider counts, and CP9k left
// the turn uncounted rather than guess about money. Measured on the dev cell
// 2026-09-04, that hid 11,244 events of generation from the cell's own
// telemetry while the upstream billed for all of them, so the turn is now
// counted from what the Router itself observed and the line says the counts
// are an estimate.
//
// "Observed" is a narrow claim. It is the text the Router forwarded
// downstream, accumulated while it translated the stream -- answer, refusal,
// reasoning and tool-call arguments -- and the request text it measured before
// dispatch. Both are converted by the character-based counter the Router
// already routes with, which is four bytes to a token and not a provider
// tokenizer. Nothing here reaches for a number the Router did not see: the
// routing token floor is not used, because it folds in the client's output
// reserve and would report a prompt larger than the prompt.
//
// The estimate is therefore low on both sides. Request tokens omit tool
// schemas and images, which the Router measures separately, and generated
// tokens omit anything the upstream produced after the cut. Under-reporting is
// the direction to be wrong in here.
func estimatedTruncatedStreamUsage(ctx *RequestContext) (responseUsageMetrics, bool) {
	generated := generatedTextBytes(ctx.SemanticStreamState)
	if generated == 0 {
		return responseUsageMetrics{}, false
	}
	completion := estimatedTokensForBytes(generated)
	prompt := estimatedTokensForBytes(ctx.VSRContextTextBytes)
	return responseUsageMetrics{
		promptTokens:             prompt,
		promptTokensReported:     prompt > 0,
		completionTokens:         completion,
		completionTokensReported: true,
		totalTokens:              prompt + completion,
		totalTokensReported:      true,
		estimated:                true,
	}, true
}

// generatedTextBytes is everything the turn produced that reached the client.
func generatedTextBytes(state *semanticResponseStreamState) int {
	if state == nil {
		return 0
	}
	bytes := 0
	for _, item := range state.items {
		bytes += len(item.text) + len(item.refusal) + len(item.reasoning)
		if item.toolCall != nil {
			bytes += len(item.toolCall.Arguments)
		}
	}
	return bytes
}

func estimatedTokensForBytes(count int) int {
	if count <= 0 {
		return 0
	}
	return 1 + (count-1)/classification.CharactersPerToken
}

// responseUsageSource says where the counts on an llm_usage line came from, so
// a consumer can tell a settlement from an estimate without reading the
// absence of a field.
func responseUsageSource(usage responseUsageMetrics) string {
	if usage.estimated {
		return "stream_estimate"
	}
	return "authoritative"
}

// attachTruncatedStreamUsage tells the client what the turn it is losing cost.
//
// Until this existed the count stopped at the Router's own telemetry: the
// client received an error frame with no numbers on it, so the proxy in front
// of the cell billed nothing while the upstream billed for the whole
// generation. The count now rides the terminal the codec was already going to
// write.
//
// It is only ever called for a turn the Router itself ended. A provider
// failure is the provider's to describe, and an estimate on it would be the
// Router inventing a number about someone else's turn.
func (r *OpenAIRouter) attachTruncatedStreamUsage(ctx *RequestContext) {
	if ctx == nil || ctx.ProtocolResponseStream == nil {
		return
	}
	usage := truncatedStreamUsage(ctx, invalidResponseTerminalUsage("stream_cut"))
	if usage.invalid {
		// Nothing was seen, so there is nothing to say. The turn is named as
		// uncounted on the telemetry side instead.
		return
	}
	ctx.ProtocolResponseStream.SetTruncationUsage(
		truncatedStreamNeutralUsage(usage), llmprotocol.UsageSourceStreamEstimate,
	)
}

// truncatedStreamNeutralUsage carries the counted row back into the shape the
// codec writes from.
//
// The whole row travels as an estimate even when the upstream stated some of
// it before the cut. A turn that did not finish is not a settlement, whatever
// its numbers came from, and one flag per row is what a reconciler can sum.
// The Router's own llm_usage line keeps the finer distinction: usage_source
// there says whether the Router invented the numbers, and truncated says the
// turn was cut.
func truncatedStreamNeutralUsage(usage responseUsageMetrics) *llmprotocol.Usage {
	count := func(value int, reported bool) llmprotocol.TokenCount {
		if !reported {
			return llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown}
		}
		return llmprotocol.TokenCount{
			Value: llmprotocol.Int64(int64(value)), Provenance: llmprotocol.UsageEstimated,
		}
	}
	neutral := &llmprotocol.Usage{
		State:           llmprotocol.UsageAvailable,
		InputUncached:   llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown},
		InputCacheRead:  count(usage.cachedPromptTokens, usage.cachedPromptTokensReported),
		InputCacheWrite: count(usage.cacheWriteTokens, usage.cacheWriteTokensReported),
		OutputReasoning: llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown},
		OutputOther:     llmprotocol.TokenCount{Provenance: llmprotocol.UsageUnknown},
		InputTotal:      count(usage.promptTokens, usage.promptTokensReported),
		OutputTotal:     count(usage.completionTokens, usage.completionTokensReported),
		Total:           count(usage.totalTokens, usage.totalTokensReported),
	}
	// Messages states the fresh part of a prompt beside the cached part, so
	// the cached tokens have to come out of the number the client is billed
	// as input. The Chat side states a prompt total that already includes
	// them.
	fresh := usage.promptTokens - usage.cachedPromptTokens - usage.cacheWriteTokens
	if usage.promptTokensReported && fresh >= 0 {
		neutral.InputUncached = count(fresh, true)
	}
	return neutral
}
