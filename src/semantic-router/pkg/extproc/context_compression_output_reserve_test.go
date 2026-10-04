package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func outputReserveCompressionRequest() *llmprotocol.Request {
	return semanticCompressionRequest(strings.Repeat("irrelevant inventory ", 300) +
		"authentication validator failed " + strings.Repeat("irrelevant billing ", 300))
}

// compressWithOutputReserve compresses the shared fixture for a Responses
// client routed to a model of the given format, with the given card limit and
// context window.
func compressWithOutputReserve(t *testing.T, target llmprotocol.WireFormat, card, window int) *RequestContext {
	t.Helper()
	router, model := routingTestRouterForFormat(target)
	params := router.Config.ModelConfig[model]
	params.MaxOutputTokens, params.ContextWindowSize = card, window
	router.Config.ModelConfig[model] = params
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "compression-reserve", SourceFormat: llmprotocol.OpenAIResponsesV1,
		VSRSelectedModel: model, VSRSelectedDecision: contextCompressionTestDecision(false),
	}
	if err := router.applySemanticContextCompression(ctx, outputReserveCompressionRequest()); err != nil {
		t.Fatalf("compress: %v", err)
	}
	if !ctx.ContextCompressionApplied {
		t.Fatalf("compression did not apply: %q", ctx.ContextCompressionSkipReason)
	}
	return ctx
}

// A request with no output limit is compressed for the limit dispatch will
// set. The prompt fits the window on its own, so without the reserve it is
// compressed only by the per-item threshold; with the card value reserved it
// no longer fits, compression is triggered by the context window, and the
// result fits beside the reserved output.
func TestCompressionReservesTheDispatchOutputBound(t *testing.T) {
	baseline := compressWithOutputReserve(t, llmprotocol.OpenAIChatV1, 0, 0)
	original := baseline.ContextCompressionBefore
	if original < 600 {
		t.Fatalf("fixture too small to test a reserve: %d tokens", original)
	}
	const card = 400
	window := original + card/2

	unreserved := compressWithOutputReserve(t, llmprotocol.OpenAIChatV1, 0, window)
	if unreserved.ContextCompressionTrigger != "item_threshold" {
		t.Fatalf("without a bound the trigger = %q, want item_threshold", unreserved.ContextCompressionTrigger)
	}

	reserved := compressWithOutputReserve(t, llmprotocol.AnthropicMessagesV1, card, window)
	if reserved.ContextCompressionTrigger != "context_window" {
		t.Fatalf("with the card reserved the trigger = %q, want context_window", reserved.ContextCompressionTrigger)
	}
	if after := reserved.ContextCompressionAfter; after > window-card {
		t.Fatalf("compressed to %d tokens, which leaves less than the card's %d of a %d window", after, card, window)
	}
}

// The reserve is the same plan dispatch applies: the card value, the Messages
// fallback, nothing for an optional target without a card, the decision's
// cap, and a caller's own limit as stated.
func TestCompressionOutputReserveMatchesTheDispatchPlan(t *testing.T) {
	for name, test := range map[string]struct {
		target   llmprotocol.WireFormat
		card     int
		limit    int
		stated   *int64
		expected int64
	}{
		"messages card":         {llmprotocol.AnthropicMessagesV1, 16000, 0, nil, 16000},
		"messages fallback":     {llmprotocol.AnthropicMessagesV1, 0, 0, nil, 32000},
		"messages capped":       {llmprotocol.AnthropicMessagesV1, 0, 2000, nil, 2000},
		"chat without a card":   {llmprotocol.OpenAIChatV1, 0, 0, nil, 0},
		"responses card":        {llmprotocol.OpenAIResponsesV1, 8000, 0, nil, 8000},
		"responses below 16":    {llmprotocol.OpenAIResponsesV1, 4, 0, nil, 16},
		"caller's stated limit": {llmprotocol.AnthropicMessagesV1, 16000, 0, llmprotocol.Int64(500), 500},
	} {
		t.Run(name, func(t *testing.T) {
			router, model := routingTestRouterForFormat(test.target)
			params := router.Config.ModelConfig[model]
			params.MaxOutputTokens = test.card
			router.Config.ModelConfig[model] = params
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, VSRSelectedModel: model}
			if test.limit > 0 {
				ctx.VSRSelectedDecision = maxTokensLimitDecision(t, test.limit)
			}
			request := testNeutralRequest(model, "hi")
			request.Sampling.MaxOutputTokens = test.stated
			if got := router.semanticContextCompressionCapabilities(ctx, request).RequestedOutput; int64(got) != test.expected {
				t.Fatalf("reserved %d output tokens, want %d", got, test.expected)
			}
		})
	}
}
