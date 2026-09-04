package extproc

import (
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
