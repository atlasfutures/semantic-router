package extproc

import (
	"context"
	"strings"
	"testing"

	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A stream the platform cuts is never finalized, so the turn leaves no usage
// record at all. Measured on the dev cell 2026-09-03: three turns were cut at
// the 630 s deadline and none of the three has an llm_usage line. They are
// uncounted and unbilled in the cell's own telemetry, while the upstream
// charged for every token it generated.
//
// handleProcessReceiveError is where the cut arrives. It marks the stream
// aborted and returns without ever finalizing it.
func TestCutStreamIsCountedWithItsUsage(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_70472e24-000", RequestModel: "deepseek/deepseek-v4-flash@thinking-on",
		IsStreamingResponse: true,
		SemanticStreamState: &semanticResponseStreamState{
			responseID: "gen-1", model: "deepseek/deepseek-v4-flash@thinking-on",
			items: map[int]*semanticStreamItem{},
			usage: llmprotocol.Usage{
				State:       llmprotocol.UsageAvailable,
				InputTotal:  llmprotocol.TokenCount{Value: llmprotocol.Int64(220), Provenance: llmprotocol.UsageAuthoritative},
				OutputTotal: llmprotocol.TokenCount{Value: llmprotocol.Int64(37213), Provenance: llmprotocol.UsageAuthoritative},
				Total:       llmprotocol.TokenCount{Value: llmprotocol.Int64(37433), Provenance: llmprotocol.UsageAuthoritative},
			},
		},
	}
	// The stream carried a partial answer and then stopped.
	item := ctx.SemanticStreamState.item(0)
	item.text = "a partial answer"

	if err := router.handleProcessReceiveError(ctx, context.DeadlineExceeded); err != nil {
		t.Fatalf("handleProcessReceiveError returned %v", err)
	}

	fields := findLogEvent(t, logs, "llm_usage")
	if truncated, _ := fields["truncated"].(bool); !truncated {
		t.Fatalf("a cut turn was counted as a whole one: %v", fields)
	}
	if got, _ := fields["completion_tokens"].(int64); got != 37213 {
		t.Fatalf("completion_tokens = %v, want the tokens the stream carried", fields["completion_tokens"])
	}
	// The counts came from the upstream, so the line says so and billing can
	// settle on it.
	if source, _ := fields["usage_source"].(string); source != "authoritative" {
		t.Fatalf("usage_source = %v, want authoritative", fields["usage_source"])
	}
}

// A turn cut before any usage arrived is still a turn that generated. Measured
// on the dev cell 2026-09-04: the Router's own deadline fired at 590 s,
// stream_truncated_uncounted was written with reason
// authoritative_usage_invalid, no llm_usage line followed, and 11,244 events of
// generation went unbilled.
//
// So the counts the stream carried are reported, marked as what they are. What
// "carried" means is only what the Router forwarded downstream: the output
// text, refusal, reasoning and tool-call arguments it accumulated as it
// translated the stream, converted by the same character-based counter the
// Router already routes with, plus the request text it measured before
// dispatch. No provider number is guessed at, and usage_source says the line
// is an estimate so nothing downstream reads it as a settlement.
func TestCutStreamWithoutUsageIsCountedFromWhatItCarried(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_c5a1de25-000", RequestModel: "deepseek/deepseek-v4-flash@thinking-on",
		IsStreamingResponse: true,
		// 800 bytes of request text, which is 200 tokens at the Router's own
		// four bytes to a token.
		VSRContextTextBytes: 800,
		SemanticStreamState: &semanticResponseStreamState{
			responseID: "gen-2", model: "deepseek/deepseek-v4-flash@thinking-on",
			items: map[int]*semanticStreamItem{},
		},
	}
	// 100 bytes of answer and 300 of reasoning, which is 100 tokens.
	item := ctx.SemanticStreamState.item(0)
	item.text = strings.Repeat("x", 100)
	item.reasoning = strings.Repeat("y", 300)

	if err := router.handleProcessReceiveError(ctx, context.DeadlineExceeded); err != nil {
		t.Fatalf("handleProcessReceiveError returned %v", err)
	}

	fields := findLogEvent(t, logs, "llm_usage")
	if source, _ := fields["usage_source"].(string); source != "stream_estimate" {
		t.Fatalf("an estimated turn was not marked as one: %v", fields)
	}
	if truncated, _ := fields["truncated"].(bool); !truncated {
		t.Fatalf("an estimated turn was counted as a whole one: %v", fields)
	}
	if got, _ := fields["completion_tokens"].(int64); got != 100 {
		t.Fatalf("completion_tokens = %v, want the tokens the stream carried", fields["completion_tokens"])
	}
	if got, _ := fields["prompt_tokens"].(int64); got != 200 {
		t.Fatalf("prompt_tokens = %v, want the request text the Router measured", fields["prompt_tokens"])
	}
}

// A turn that carried nothing is still uncounted, and still says so. There is
// no number to estimate from, and inventing one would be inventing money.
func TestCutStreamThatCarriedNothingSaysSoRatherThanGuessing(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_c5a1de25-001", RequestModel: "deepseek/deepseek-v4-flash@thinking-on",
		IsStreamingResponse: true,
		VSRContextTextBytes: 800,
		SemanticStreamState: &semanticResponseStreamState{
			responseID: "gen-3", model: "deepseek/deepseek-v4-flash@thinking-on",
			items: map[int]*semanticStreamItem{},
		},
	}

	if err := router.handleProcessReceiveError(ctx, context.DeadlineExceeded); err != nil {
		t.Fatalf("handleProcessReceiveError returned %v", err)
	}

	fields := findLogEvent(t, logs, "stream_truncated_uncounted")
	if got, _ := fields["request_id"].(string); got != "rt_c5a1de25-001" {
		t.Fatalf("the uncounted turn was not named: %v", fields)
	}
	for _, forbidden := range []string{"completion_tokens", "prompt_tokens", "cost"} {
		if _, present := fields[forbidden]; present {
			t.Fatalf("a token count was invented for a turn that carried none: %v", fields)
		}
	}
}

// A stream that ended normally is untouched: no truncation marker.
func TestCompletedStreamCarriesNoTruncationMarker(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_ok", RequestModel: "m", IsStreamingResponse: true,
		SemanticStreamState: &semanticResponseStreamState{
			responseID: "gen-3", model: "m", stop: llmprotocol.StopEndTurn,
			items: map[int]*semanticStreamItem{}, terminal: true,
			usage: llmprotocol.Usage{
				State:       llmprotocol.UsageAvailable,
				InputTotal:  llmprotocol.TokenCount{Value: llmprotocol.Int64(10), Provenance: llmprotocol.UsageAuthoritative},
				OutputTotal: llmprotocol.TokenCount{Value: llmprotocol.Int64(20), Provenance: llmprotocol.UsageAuthoritative},
				Total:       llmprotocol.TokenCount{Value: llmprotocol.Int64(30), Provenance: llmprotocol.UsageAuthoritative},
			},
		},
	}
	item := ctx.SemanticStreamState.item(0)
	item.text, item.completed = "done", true

	router.finalizeSemanticStreamingResponse(ctx, nil)

	fields := findLogEvent(t, logs, "llm_usage")
	if _, present := fields["truncated"]; present {
		t.Fatalf("a completed turn was marked truncated: %v", fields)
	}
}

func findLogEvent(t *testing.T, logs *observer.ObservedLogs, event string) map[string]interface{} {
	t.Helper()
	for _, entry := range logs.All() {
		fields := entry.ContextMap()
		if name, _ := fields["event"].(string); name == event {
			return fields
		}
	}
	t.Fatalf("no %q event was logged", event)
	return nil
}
