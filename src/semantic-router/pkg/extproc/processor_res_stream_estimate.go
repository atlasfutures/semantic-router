package extproc

import (
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
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
	estimate := responseUsageMetrics{
		completionTokens:         completion,
		completionTokensReported: true,
		estimated:                true,
	}
	// The prompt the stream itself stated is a number the Router saw, and a
	// better one than its own byte count: Anthropic's message_start says the
	// input and cache legs before the first delta, and a prompt does not grow
	// after it is counted. Only a stream that said nothing is measured.
	if !statedInputLegs(ctx.SemanticStreamState, &estimate) {
		prompt := estimatedTokensForBytes(ctx.VSRContextTextBytes)
		estimate.promptTokens, estimate.promptTokensReported = prompt, prompt > 0
	}
	estimate.totalTokens = estimate.promptTokens + estimate.completionTokens
	estimate.totalTokensReported = true
	return estimate, true
}

// statedInputLegs copies the input legs the stream stated onto an estimate,
// and reports whether it stated any. Provenance is not required to be
// authoritative here: the turn is already an estimate, and a derived input
// total is still the stream's own statement rather than the Router's guess.
func statedInputLegs(state *semanticResponseStreamState, estimate *responseUsageMetrics) bool {
	if state == nil || state.usage.State != llmprotocol.UsageAvailable || state.usage.InputTotal.Value == nil {
		return false
	}
	estimate.promptTokens = int(*state.usage.InputTotal.Value)
	estimate.promptTokensReported = true
	if state.usage.InputCacheRead.Value != nil {
		estimate.cachedPromptTokens = int(*state.usage.InputCacheRead.Value)
		estimate.cachedPromptTokensReported = true
	}
	if state.usage.InputCacheWrite.Value != nil {
		estimate.cacheWriteTokens = int(*state.usage.InputCacheWrite.Value)
		estimate.cacheWriteTokensReported = true
	}
	return true
}

// outputRestatedPastSnapshot takes a count the upstream stated and asks
// whether the turn kept generating after it was stated.
//
// If it did not, the count is a settlement and stands. If it did, the output
// leg is a snapshot: Anthropic's message_start states output_tokens 1 before
// the first delta, and a turn the Router cut after 780 deltas was being billed
// as one token, marked authoritative. Measured live 2026-10-26. The output
// then takes the larger of the snapshot and the Router's own estimate of what
// it forwarded, and whichever wins the row says it is an estimate: the
// snapshot is known to be short, and the estimate is bytes over four, which is
// low on average rather than a bound. The input leg is left as stated: a
// prompt does not grow after it is counted.
func outputRestatedPastSnapshot(ctx *RequestContext, usage responseUsageMetrics) responseUsageMetrics {
	state := ctx.SemanticStreamState
	if state == nil || generatedTextBytes(state) <= state.generatedAtUsage {
		return usage
	}
	usage.estimated = true
	estimated, carried := estimatedTruncatedStreamUsage(ctx)
	if !carried || estimated.completionTokens <= usage.completionTokens {
		return usage
	}
	usage.completionTokens = estimated.completionTokens
	usage.completionTokensReported = true
	usage.totalTokens = usage.promptTokens + usage.completionTokens
	usage.totalTokensReported = true
	return usage
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
	// No count at all is no settlement: a refusal Anthropic's safeguards
	// issue before generating, or a provider error, states no usage, and its
	// null counts are unknown, not an authoritative zero.
	if !usage.promptTokensReported && !usage.completionTokensReported && !usage.totalTokensReported &&
		!usage.cachedPromptTokensReported && !usage.cacheWriteTokensReported {
		return usageSourceUnknown
	}
	return "authoritative"
}

// usageSourceUnknown is the llm_usage usage_source of a call whose provider
// stated no token counts.
const usageSourceUnknown = "unknown"

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
