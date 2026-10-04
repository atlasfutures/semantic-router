//go:build !windows && cgo

package extproc

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A refused turn's diagnostics say why it was refused: the provider's own
// stop string (OpenRouter's native_finish_reason) and Anthropic's
// stop_details type and category reach both the turn_failed line and the
// llm_usage record, for a buffered and a streamed refusal alike.
func TestRefusalStopDetailsReachTheTurnDiagnostics(t *testing.T) {
	assertStopFields := func(t *testing.T, event string, fields map[string]interface{}) {
		t.Helper()
		got := fmt.Sprint(fields["provider_stop_reason"], " ", fields["stop_details_type"], " ", fields["stop_details_category"])
		if got != "refusal refusal cyber" {
			t.Fatalf("%s stop telemetry = %q, want refusal refusal cyber: %v", event, got, fields)
		}
	}
	t.Run("buffered", func(t *testing.T) {
		logs := captureLogs(t)
		_, router, decision := statusCacheRouter()
		ctx := withSelectedDecision(&RequestContext{
			RequestID: "req-refusal-details", RequestModel: "test", RequestQuery: "hello",
			SemanticRequest: testNeutralRequest("test", "hello"),
			SourceFormat:    llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
			TraceContext: context.Background(), UpstreamStatusCode: 200,
		}, decision)
		body := `{"id":"gen-refused","object":"chat.completion","created":1757030000,"model":"test",` +
			`"choices":[{"index":0,"finish_reason":"content_filter","native_finish_reason":"refusal",` +
			`"stop_details":{"type":"refusal","category":"cyber","explanation":"Flagged."},` +
			`"message":{"role":"assistant","content":null,"refusal":"This request triggered restrictions."}}]}`
		router.handleNonStreamingResponseBody([]byte(body), ctx, time.Second)
		runPendingSelectionCompletion(ctx)
		finalizeSelectionProcessTerminal(ctx)
		assertStopFields(t, "turn_failed", findLogEvent(t, logs, "turn_failed"))
		assertStopFields(t, "llm_usage", findLogEvent(t, logs, "llm_usage"))
	})
	t.Run("streamed", func(t *testing.T) {
		logs := captureLogs(t)
		_, router, decision := statusCacheRouter()
		ctx := withSelectedDecision(&RequestContext{
			RequestID: "req-refusal-details-stream", RequestModel: "test", RequestQuery: "hello",
			SemanticRequest: testNeutralRequest("test", "hello"),
			SourceFormat:    llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
			TraceContext: context.Background(), UpstreamStatusCode: 200,
		}, decision)
		sendHeaders(t, router, ctx, streamingResponseHeaders("200"))
		ctx.SemanticStreamState = &semanticResponseStreamState{
			terminal: true, stop: llmprotocol.StopContentFilter, items: map[int]*semanticStreamItem{},
			providerStop: "refusal", stopDetails: &llmprotocol.StopDetails{Type: "refusal", Category: "cyber"},
		}
		router.finalizeSemanticStreamingResponse(ctx, nil)
		runPendingSelectionCompletion(ctx)
		assertStopFields(t, "turn_failed", findLogEvent(t, logs, "turn_failed"))
		assertStopFields(t, "llm_usage", findLogEvent(t, logs, "llm_usage"))
	})
}
