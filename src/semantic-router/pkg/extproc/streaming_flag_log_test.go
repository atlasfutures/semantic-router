package extproc

import (
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// Whether a turn streamed is a routing fact that never reached a live log.
// The CP9x/decision-41 traffic-mix analysis had to infer the streaming share
// from a Firebase side effect, because llm_usage and routing_decision both
// omit it and router_replay_complete -- the one event that carries it -- is
// disabled on the ARC cell.
//
// The two events answer the question at different phases, so they answer
// different halves of it. recordResponseCost runs after the response headers,
// where ctx.IsStreamingResponse is the upstream's own answer. emitRoutingDecision
// runs at provider dispatch, before any response header exists, so the only
// truth there is the request's own declaration, ctx.ExpectStreamingResponse.
// The keys are named apart for that reason: joining them as one fact would
// silently equate a request that asked to stream with a response that did.

func streamingFlagUsage() responseUsageMetrics {
	return responseUsageMetrics{
		promptTokens: 220, promptTokensReported: true,
		completionTokens: 64, completionTokensReported: true,
		totalTokens: 284, totalTokensReported: true,
	}
}

// A streamed turn says so on the accounting line.
func TestUsageLineMarksAStreamedTurn(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_stream_yes", RequestModel: "deepseek/deepseek-v4-flash@thinking-on",
		IsStreamingResponse: true,
	}

	router.reportSemanticStreamingUsage(ctx, time.Second, streamingFlagUsage())

	fields := findLogEvent(t, logs, "llm_usage")
	streaming, present := fields["streaming"].(bool)
	if !present {
		t.Fatalf("the accounting line does not say whether the turn streamed: %v", fields)
	}
	if !streaming {
		t.Fatalf("streaming = false on a streamed turn: %v", fields)
	}
}

// A buffered turn carries the field too. An absent field is indistinguishable
// from a turn the Router did not classify, which is what made the traffic mix
// unreadable in the first place.
func TestUsageLineMarksABufferedTurn(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_stream_no", RequestModel: "deepseek/deepseek-v4-flash@thinking-off",
		IsStreamingResponse: false,
	}

	router.reportNonStreamingUsage(ctx, time.Second, streamingFlagUsage())

	fields := findLogEvent(t, logs, "llm_usage")
	streaming, present := fields["streaming"].(bool)
	if !present {
		t.Fatalf("the accounting line omitted streaming rather than answering it: %v", fields)
	}
	if streaming {
		t.Fatalf("streaming = true on a buffered turn: %v", fields)
	}
}

// The decision record says what the dispatched body asked the upstream for.
func TestRoutingDecisionMarksAStreamingRequest(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_route_stream", ProcessingStartTime: time.Now(),
		ExpectStreamingResponse: true,
	}

	router.logRoutingDecision(ctx, "entrypoint_routing", "auto", "gpt-5-mini", "arc", true)
	router.emitRoutingDecision(ctx)

	fields := findLogEvent(t, logs, "routing_decision")
	requested, present := fields["streaming_requested"].(bool)
	if !present {
		t.Fatalf("the decision record does not say whether the turn asked to stream: %v", fields)
	}
	if !requested {
		t.Fatalf("streaming_requested = false on a streaming request: %v", fields)
	}
}

// A non-streaming request carries the field as false, not as an omission.
func TestRoutingDecisionMarksABufferedRequest(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_route_buffered", ProcessingStartTime: time.Now(),
		ExpectStreamingResponse: false,
	}

	router.logRoutingDecision(ctx, "entrypoint_routing", "auto", "gpt-5-mini", "arc", false)
	router.emitRoutingDecision(ctx)

	fields := findLogEvent(t, logs, "routing_decision")
	requested, present := fields["streaming_requested"].(bool)
	if !present {
		t.Fatalf("the decision record omitted streaming_requested rather than answering it: %v", fields)
	}
	if requested {
		t.Fatalf("streaming_requested = true on a buffered request: %v", fields)
	}
}

// The two keys are deliberately not the same key, and this is the case that
// proves it. A turn can ask to stream and not get a stream: IsStreamingResponse
// is gated on a 2xx, so an upstream refusal leaves the request's declaration
// true and the response's observation false.
//
// Anyone tempted to "fix" the names to match should fail here first. Renaming
// either key to the other would make this turn report itself as streamed on
// one line and buffered on the other under a single name, or collapse the two
// facts into whichever phase wrote last.
func TestStreamingKeysDivergeWhenTheUpstreamDidNotStream(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID: "rt_diverge", RequestModel: "deepseek/deepseek-v4-flash@thinking-on",
		ProcessingStartTime:     time.Now(),
		ExpectStreamingResponse: true,
	}

	// The upstream refused. The SSE content-type on the refusal is deliberate:
	// the content-type alone must not make this a streamed turn, because the
	// status says nothing was served.
	outcome := evaluateResponseHeaderOutcome(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":status", Value: "503"},
				{Key: "content-type", Value: "text/event-stream"},
			}},
		},
	}, ctx)
	if outcome.isSuccessful {
		t.Fatalf("a 503 was read as a successful response: %+v", outcome)
	}
	if ctx.IsStreamingResponse {
		t.Fatal("a refused turn was recorded as having streamed")
	}

	router.logRoutingDecision(ctx, "entrypoint_routing", "auto", "gpt-5-mini", "arc", true)
	router.emitRoutingDecision(ctx)
	router.reportNonStreamingUsage(ctx, time.Second, streamingFlagUsage())

	decision := findLogEvent(t, logs, "routing_decision")
	requested, present := decision["streaming_requested"].(bool)
	if !present || !requested {
		t.Fatalf("streaming_requested must stay true: the request did ask to stream: %v", decision)
	}

	usage := findLogEvent(t, logs, "llm_usage")
	streamed, present := usage["streaming"].(bool)
	if !present {
		t.Fatalf("the accounting line dropped streaming: %v", usage)
	}
	if streamed {
		t.Fatalf("streaming must be false: the upstream refused and served no stream: %v", usage)
	}
}
