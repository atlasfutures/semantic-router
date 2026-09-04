package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The rare upstream_completion_was_empty outcome was joined against 13
// occurrences in 48 h on 2026-09-04. Every one billed completion_tokens = 1, a
// bare stop token, so the reasoning budget was not what ran out. Which upstream
// served the turn, what it said it stopped for, and how much of the completion
// was reasoning are the three facts that would attribute it, and none of them
// reached the logs.
func usageAttributionResponse(provider string, reasoning *int64) *llmprotocol.Response {
	usage := llmprotocol.Usage{
		State:       llmprotocol.UsageAvailable,
		InputTotal:  llmprotocol.TokenCount{Value: llmprotocol.Int64(31402), Provenance: llmprotocol.UsageAuthoritative},
		OutputTotal: llmprotocol.TokenCount{Value: llmprotocol.Int64(64), Provenance: llmprotocol.UsageAuthoritative},
		Total:       llmprotocol.TokenCount{Value: llmprotocol.Int64(31466), Provenance: llmprotocol.UsageAuthoritative},
	}
	if reasoning != nil {
		usage.OutputReasoning = llmprotocol.TokenCount{Value: reasoning, Provenance: llmprotocol.UsageAuthoritative}
	}
	return &llmprotocol.Response{
		Generation: 1, ID: "gen-1", Model: "deepseek/deepseek-v4-pro@thinking-off",
		Output: []llmprotocol.OutputItem{{
			ID: "item_0", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "an answer"}},
		}},
		StopReason: llmprotocol.StopEndTurn, SourceStopReason: "stop",
		UpstreamProvider: provider, Usage: usage,
	}
}

// A buffered turn that reasoned names its upstream, its stop and its reasoning
// split on the one line accounting already reads.
func TestUsageLineNamesTheUpstreamAttribution(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_9f21ab40-000", RequestModel: "deepseek/deepseek-v4-pro@thinking-off",
		SemanticResponse: usageAttributionResponse("Ionstream", llmprotocol.Int64(48)),
	}

	router.reportNonStreamingUsage(ctx, time.Second, router.takeNeutralResponseUsage(ctx))

	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["upstream_provider"].(string); got != "Ionstream" {
		t.Fatalf("upstream_provider = %v, want the provider OpenRouter reported", fields["upstream_provider"])
	}
	if got, _ := fields["stop_reason"].(string); got != "end_turn" {
		t.Fatalf("stop_reason = %v, want the resolved stop", fields["stop_reason"])
	}
	if got, _ := fields["native_stop_reason"].(string); got != "stop" {
		t.Fatalf("native_stop_reason = %v, want the reason the upstream sent", fields["native_stop_reason"])
	}
	if got, _ := fields["reasoning_tokens"].(int64); got != 48 {
		t.Fatalf("reasoning_tokens = %v, want the provider's split", fields["reasoning_tokens"])
	}
}

// A split the provider never stated is absent, not zero. A zero would assert a
// turn that did not reason, and the empties are being told apart from turns
// that reasoned to exhaustion.
func TestUsageLineOmitsAnUnknownReasoningSplit(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_9f21ab40-001", RequestModel: "deepseek/deepseek-v4-pro@thinking-off",
		SemanticResponse: usageAttributionResponse("Ionstream", nil),
	}

	router.reportNonStreamingUsage(ctx, time.Second, router.takeNeutralResponseUsage(ctx))

	fields := findLogEvent(t, logs, "llm_usage")
	if _, present := fields["reasoning_tokens"]; present {
		t.Fatalf("a reasoning split was invented for a turn that stated none: %v", fields)
	}
}

func attributedStreamContext(requestID string, deltas ...string) *RequestContext {
	ctx := &RequestContext{
		RequestID: requestID, RequestModel: "deepseek/deepseek-v4-pro@thinking-off",
		IsStreamingResponse: true,
		SemanticStreamState: &semanticResponseStreamState{
			requestID: requestID, items: map[int]*semanticStreamItem{},
		},
	}
	events := []llmprotocol.Event{
		{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "item_0", Role: llmprotocol.RoleAssistant},
	}
	for _, delta := range deltas {
		events = append(events, llmprotocol.Event{Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, Delta: delta})
	}
	events = append(events,
		llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0},
		llmprotocol.Event{
			Type: llmprotocol.EventResponseCompleted, ResponseID: "gen-2",
			Model:      "deepseek/deepseek-v4-pro@thinking-off",
			StopReason: llmprotocol.StopEndTurn, SourceStopReason: "stop",
			UpstreamProvider: "Venice",
			Usage: &llmprotocol.Usage{
				State:           llmprotocol.UsageAvailable,
				InputTotal:      llmprotocol.TokenCount{Value: llmprotocol.Int64(31402), Provenance: llmprotocol.UsageAuthoritative},
				OutputTotal:     llmprotocol.TokenCount{Value: llmprotocol.Int64(1), Provenance: llmprotocol.UsageAuthoritative},
				Total:           llmprotocol.TokenCount{Value: llmprotocol.Int64(31403), Provenance: llmprotocol.UsageAuthoritative},
				OutputReasoning: llmprotocol.TokenCount{Value: llmprotocol.Int64(0), Provenance: llmprotocol.UsageAuthoritative},
			},
		},
	)
	ctx.SemanticStreamState.observe(events)
	return ctx
}

// A streamed turn carries the same attribution as a buffered one. This is the
// path the measured empties took.
func TestStreamedUsageLineNamesTheUpstreamAttribution(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := attributedStreamContext("rt_9f21ab40-002", "an answer")

	router.finalizeSemanticStreamingResponse(ctx, nil)

	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["upstream_provider"].(string); got != "Venice" {
		t.Fatalf("upstream_provider = %v, want the provider the stream reported", fields["upstream_provider"])
	}
	if got, _ := fields["stop_reason"].(string); got != "end_turn" {
		t.Fatalf("stop_reason = %v, want the resolved stop", fields["stop_reason"])
	}
	if got, _ := fields["native_stop_reason"].(string); got != "stop" {
		t.Fatalf("native_stop_reason = %v, want the reason the upstream sent", fields["native_stop_reason"])
	}
	if got, _ := fields["reasoning_tokens"].(int64); got != 0 {
		t.Fatalf("reasoning_tokens = %v, want the split the stream stated", fields["reasoning_tokens"])
	}
}

// The empty-completion line named only an item index, so it could be joined to
// the turn it belongs to by same-millisecond timestamp and nothing else. It
// carries the request id every other line is joined on.
func TestEmptyCompletionLineNamesItsRequest(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := attributedStreamContext("rt_9f21ab40-003")

	router.finalizeSemanticStreamingResponse(ctx, nil)

	empty := findLogEvent(t, logs, "upstream_completion_was_empty")
	if got, _ := empty["request_id"].(string); got != "rt_9f21ab40-003" {
		t.Fatalf("request_id = %v, want the turn the empty completion belongs to", empty["request_id"])
	}
	// The turn that produced no content is the one being attributed, so its
	// usage line has to carry the same three facts.
	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["upstream_provider"].(string); got != "Venice" {
		t.Fatalf("upstream_provider = %v, want the provider that served the empty turn", fields["upstream_provider"])
	}
	if got, _ := fields["native_stop_reason"].(string); got != "stop" {
		t.Fatalf("native_stop_reason = %v, want the reason the upstream gave for an empty turn", fields["native_stop_reason"])
	}
	if got, _ := fields["completion_tokens"].(int64); got != 1 {
		t.Fatalf("completion_tokens = %v, want the bare stop token the provider billed", fields["completion_tokens"])
	}
}
