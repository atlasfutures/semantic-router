package extproc

import (
	"context"
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
}

// When the cut arrives before any usage does, there is nothing to count. The
// turn still has to be visible: silence is what made these three invisible.
func TestCutStreamWithoutUsageSaysSoRatherThanGuessing(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_c5a1de25-000", RequestModel: "deepseek/deepseek-v4-flash@thinking-on",
		IsStreamingResponse: true,
		SemanticStreamState: &semanticResponseStreamState{
			responseID: "gen-2", model: "deepseek/deepseek-v4-flash@thinking-on",
			items: map[int]*semanticStreamItem{},
		},
	}
	ctx.SemanticStreamState.item(0).text = "a partial answer"

	if err := router.handleProcessReceiveError(ctx, context.DeadlineExceeded); err != nil {
		t.Fatalf("handleProcessReceiveError returned %v", err)
	}

	fields := findLogEvent(t, logs, "stream_truncated_uncounted")
	if got, _ := fields["request_id"].(string); got != "rt_c5a1de25-000" {
		t.Fatalf("the uncounted turn was not named: %v", fields)
	}
	for _, forbidden := range []string{"completion_tokens", "prompt_tokens", "cost"} {
		if _, present := fields[forbidden]; present {
			t.Fatalf("a token count was invented for a turn that reported none: %v", fields)
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
