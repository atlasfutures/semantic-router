package extproc

import (
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// Every llm_usage line names the tier the upstream served the call at, as
// the response stated it, and null when it stated none: the requested tier
// is not the billed one (OpenRouter may serve a flex request at default).

func TestUsageLineNamesTheServedServiceTier(t *testing.T) {
	for name, test := range map[string]struct {
		tier string
		want interface{}
	}{
		"stated": {"flex", "flex"},
		"absent": {"", nil},
	} {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			router := &OpenAIRouter{}
			response := usageAttributionResponse("OpenAI", nil)
			response.UpstreamServiceTier = test.tier
			ctx := &RequestContext{
				RequestID: "rt_tier-buffered", RequestModel: "openai/gpt-luna", SemanticResponse: response,
			}

			router.reportNonStreamingUsage(ctx, time.Second, router.takeNeutralResponseUsage(ctx))

			fields := findLogEvent(t, logs, "llm_usage")
			if value, present := fields["upstream_service_tier"]; !present || value != test.want {
				t.Fatalf("upstream_service_tier = %v (present %v), want %v", value, present, test.want)
			}
		})
	}
}

func TestStreamedUsageLineNamesTheServedServiceTier(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := attributedStreamContext("rt_tier-streamed")
	// The tier arrives on a chunk; the stream state keeps it for the turn.
	ctx.SemanticStreamState.observe([]llmprotocol.Event{
		{Type: llmprotocol.EventUsageUpdated, UpstreamServiceTier: "priority"},
	})

	router.finalizeSemanticStreamingResponse(ctx, nil)

	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["upstream_service_tier"].(string); got != "priority" {
		t.Fatalf("upstream_service_tier = %v, want the tier the stream stated", fields["upstream_service_tier"])
	}
}

func TestStreamedUsageLineWithoutATierIsNull(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := attributedStreamContext("rt_tier-streamed-none", "an answer")

	router.finalizeSemanticStreamingResponse(ctx, nil)

	fields := findLogEvent(t, logs, "llm_usage")
	if value, present := fields["upstream_service_tier"]; !present || value != nil {
		t.Fatalf("upstream_service_tier = %v (present %v), want null", value, present)
	}
}
