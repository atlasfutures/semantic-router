//go:build !windows && cgo

package extproc

import (
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// A streamed endpoint-picked turn whose pick answers 200 and then fails
// in-band before any content is re-sent on the arm's next endpoint, end to
// end through the extproc phases, against a fake OpenRouter that answers by
// the endpoint each call pins (newHostUpstream keys on provider.order[0]).

// endpointStreamErrorChunk is an in-band provider error before any content,
// as OpenRouter streams a provider's 5xx after answering 200.
const endpointStreamErrorChunk = `data: {"error":{"type":"api_error","code":"api_error","message":"provider overloaded"}}` + "\n\n"

// endpointStreamRouter is endpointPickRouter with the in-band re-send on or
// off.
func endpointStreamRouter(baseURL string, resend bool, pin ...string) *OpenAIRouter {
	router := endpointPickRouter(baseURL, pin...)
	router.Config.ResponseStreamEndpointStreamResend = resend
	return router
}

// routeStreamedEndpointTurn dispatches one streamed ARC turn picked among
// tags; prepare runs on the context before the response arrives.
func routeStreamedEndpointTurn(
	t *testing.T, router *OpenAIRouter, tags []string, prepare func(*RequestContext),
) (*RequestContext, *recordingSelectionTransaction) {
	t.Helper()
	request := testNeutralRequest("auto", "route this turn")
	request.Stream = true
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ExpectStreamingResponse = true
	ctx.StartTime = time.Now()
	ctx.ProcessingStartTime = ctx.StartTime
	ctx.VSRRaylineARC = &selection.RaylineARCTrace{EpisodeIDHash: "episode", SettlementPrices: endpointFeed(t, tags...)}
	ctx.RaylineARCSessionKey, ctx.RaylineARCSessionKeySource = endpointPickKey, sessionKeySourceHeader
	if _, err := router.handleEntrypointModelRouting(
		request, "auto", endpointPickDecision, entropy.ReasoningDecision{}, endpointPickArm, ctx,
	); err != nil {
		t.Fatalf("handleEntrypointModelRouting: %v", err)
	}
	if prepare != nil {
		prepare(ctx)
	}
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.True(t, ctx.IsStreamingResponse)
	transaction := &recordingSelectionTransaction{}
	ctx.SelectionTransaction = newSelectionTransactionOwner(configRaylineARC, transaction)
	return ctx, transaction
}

func sendUpstreamChunks(t *testing.T, router *OpenAIRouter, ctx *RequestContext, stream *MockStream, chunks ...string) {
	t.Helper()
	for _, chunk := range chunks {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
}

// The pick's in-band 5xx before any content is re-sent on the next endpoint,
// whose stream the client gets on the open response; each call writes its
// own usage line, and the served one says how long its frames were held.
func TestInBandFailureBeforeContentIsResentOnTheNextEndpoint(t *testing.T) {
	logs := captureLogs(t)
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	upstream := newHostUpstream(t, map[string]string{order[1]: "content"})
	router := endpointStreamRouter(upstream.server.URL, true)
	stream := NewMockStream(nil)

	ctx, transaction := routeStreamedEndpointTurn(t, router, endpointPickTags, nil)
	sendUpstreamChunks(t, router, ctx, stream, hostRetryPreContentChunk, endpointStreamErrorChunk)
	require.True(t, ctx.HostRetry.started(), "the in-band failure was not re-sent")
	assert.Nil(t, ctx.RaylineARCEndpointResend, "the status path could still re-send the turn")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "served on "+order[1])
	assert.NotContains(t, sent, "gen-stalled", "the failed pick's held frame reached the client")
	assert.NotContains(t, sent, "provider overloaded", "the failed pick's error reached the client")
	assert.Contains(t, sent, strings.TrimSpace(string(hostRetryHoldFrame)))
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.Equal(t, 0, transaction.aborts)

	calls := upstream.calls()
	require.Len(t, calls, 1)
	assert.Equal(t, []string{order[1]}, providerList(t, calls[0], "order"))
	assert.Equal(t, []string{order[1]}, providerList(t, calls[0], "only"))
	assert.True(t, slices.Equal(ctx.RaylineARCEndpointsTried, order[:2]), "tried %v", ctx.RaylineARCEndpointsTried)

	lines := usageLines(logs)
	require.Len(t, lines, 2, "want one usage line per call: %v", lines)
	assertFields(t, "failed pick", lines[0], map[string]interface{}{
		"endpoint_resend": endpointResendAttempt, "provider_endpoint": order[0],
		"failure_class": turnFailureUpstream5xx, "content_sent_before_failure": false,
	})
	assert.Nil(t, lines[0]["host_retry"])
	assert.Nil(t, lines[0]["stream_hold_ms"], "a replaced call's line carries the hold")
	assertFields(t, "served re-send", lines[1], map[string]interface{}{
		"endpoint_resend": endpointResendServed, "provider_endpoint": order[1],
		"upstream_response_id": "gen-served",
	})
	assert.Nil(t, lines[1]["failure_class"])
	assert.Nil(t, lines[1]["host_retry"])
	assert.Nil(t, lines[1]["host_attempt"])
	if tried := lines[1]["provider_endpoints_tried"]; !sameJSON(t, mustJSONRaw(t, tried), mustJSON(t, order[:2])) {
		t.Fatalf("provider_endpoints_tried = %v, want %v", tried, order[:2])
	}
	hold, ok := lines[1]["stream_hold_ms"].(int64)
	require.True(t, ok, "stream_hold_ms = %#v, want the hold", lines[1]["stream_hold_ms"])
	assert.GreaterOrEqual(t, hold, int64(0))
	event := findLogEvent(t, logs, "endpoint_stream_resend")
	assertFields(t, "re-send event", event, map[string]interface{}{
		"failed_endpoint": order[0], "next_endpoint": order[1], "trigger": turnFailureUpstream5xx, "attempt": 1,
	})
}

// A re-send to an endpoint of another host family is sent the history with
// the pick's encrypted reasoning stripped, logged as the status path logs it,
// and the run restarts at its reply.
func TestCrossFamilyInBandResendStripsTheEncryptedReasoning(t *testing.T) {
	logs := captureLogs(t)
	tags := []string{"azure", "openai"}
	order := raylinearc.PickEndpoint(endpointPickKey, tags)
	upstream := newHostUpstream(t, map[string]string{order[1]: "content"})
	router := endpointStreamRouter(upstream.server.URL, true, tags...)
	stream := NewMockStream(nil)
	transaction := &raylineARCEpisodeTransaction{thinkingWorker: "run", thinkingAdopted: "adopted", borrowed: true}

	ctx, _ := routeStreamedEndpointTurn(t, router, tags, func(ctx *RequestContext) {
		withHeldReasoning(t, ctx)
		ctx.HostRetry.dispatched = ctx.RaylineARCEndpointResend.dispatched
		ctx.RaylineARCTransaction = transaction
	})
	sendUpstreamChunks(t, router, ctx, stream, endpointStreamErrorChunk)
	require.True(t, ctx.HostRetry.started())
	drainSpliced(t, router, ctx, stream)

	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "served on "+order[1])
	bodies := upstream.received()
	require.Len(t, bodies, 1)
	history := resentHistory(t, bodies[0])
	require.Len(t, history, 1)
	assert.NotContains(t, history[0], "reasoning_details", "the re-send carried the pick's encrypted reasoning")
	assert.Equal(t, `"earlier answer"`, string(history[0]["content"]))
	assert.True(t, transaction.thinkingRestart, "the run did not restart at the stripped re-send")
	assert.Empty(t, transaction.thinkingAdopted)
	strip := findLogEvent(t, logs, "encrypted_reasoning_strip")
	assertFields(t, "strip", strip, map[string]interface{}{
		"from_endpoint": order[0], "to_endpoint": order[1],
		"from_host_family": signatureHostFamily(order[0]), "to_host_family": signatureHostFamily(order[1]),
	})
	assert.True(t, sameJSON(t, mustJSONRaw(t, strip["stripped"]), "1"), "stripped = %v", strip["stripped"])
	event := findLogEvent(t, logs, "endpoint_stream_resend")
	assert.True(t, sameJSON(t, mustJSONRaw(t, event["reasoning_stripped"]), "1"))
}

// Content already reached the client: the failure is the client's, and
// nothing is re-sent.
func TestInBandFailureAfterContentIsNotResent(t *testing.T) {
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	upstream := newHostUpstream(t, map[string]string{order[1]: "content"})
	router := endpointStreamRouter(upstream.server.URL, true)
	stream := NewMockStream(nil)

	ctx, _ := routeStreamedEndpointTurn(t, router, endpointPickTags, nil)
	sendUpstreamChunks(t, router, ctx, stream, endlessUpstreamChunk(1))
	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "still thinking", "content was held")
	sendUpstreamChunks(t, router, ctx, stream, endpointStreamErrorChunk)
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest("", true), ctx))

	assert.False(t, ctx.HostRetry.started())
	assert.Empty(t, upstream.calls())
	assert.NotEmpty(t, ctx.ResponseFailureClass)
	assert.Equal(t, order[:1], ctx.RaylineARCEndpointsTried)
}

// With the switch off nothing is staged or held, and the in-band failure is
// the client's as before.
func TestEndpointStreamResendOffLeavesTheFailureAsBefore(t *testing.T) {
	logs := captureLogs(t)
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	upstream := newHostUpstream(t, map[string]string{order[1]: "content"})
	router := endpointStreamRouter(upstream.server.URL, false)
	stream := NewMockStream(nil)

	ctx, _ := routeStreamedEndpointTurn(t, router, endpointPickTags, nil)
	require.NotNil(t, ctx.RaylineARCEndpointResend, "the status re-send is staged either way")
	assert.Nil(t, ctx.HostRetry)
	sendUpstreamChunks(t, router, ctx, stream, hostRetryPreContentChunk)
	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "gen-stalled", "a frame was held with the switch off")
	sendUpstreamChunks(t, router, ctx, stream, endpointStreamErrorChunk)
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest("", true), ctx))

	assert.Empty(t, upstream.calls())
	assert.Equal(t, turnFailureUpstream5xx, ctx.ResponseFailureClass)
	lines := usageLines(logs)
	require.Len(t, lines, 1)
	assert.Nil(t, lines[0]["endpoint_resend"])
	assert.Contains(t, lines[0], "stream_hold_ms")
	assert.Nil(t, lines[0]["stream_hold_ms"], "a turn that held nothing reports a hold")
}

// A first-content stall is not an endpoint re-send's trigger: it is cut as
// a timeout, as without the switch.
func TestFirstContentStallIsNotResentOnTheNextEndpoint(t *testing.T) {
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	upstream := newHostUpstream(t, map[string]string{order[1]: "content"})
	router := endpointStreamRouter(upstream.server.URL, true)
	router.Config.ResponseStreamFirstContentSec = 90
	stream := NewMockStream(nil)

	ctx, _ := routeStreamedEndpointTurn(t, router, endpointPickTags, nil)
	sendUpstreamChunks(t, router, ctx, stream, hostRetryPreContentChunk, openRouterKeepaliveChunk)
	ctx.StreamContentSince = time.Now().Add(-91 * time.Second)
	require.True(t, router.streamSilenceExceeded(ctx))
	require.NoError(t, router.endSilentStream(stream, ctx))

	assert.False(t, ctx.HostRetry.started())
	assert.Empty(t, upstream.calls())
	assert.True(t, ctx.StreamingComplete)
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
	assert.Equal(t, order[:1], ctx.RaylineARCEndpointsTried)
}

// The in-band re-sends stop after endpointResendMaxExtra further endpoints:
// a re-send's own in-band failure moves on, and the last one's failure (here
// a stream that ended before any content) is the client's.
func TestInBandResendsStopAfterTwoFurtherEndpoints(t *testing.T) {
	logs := captureLogs(t)
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	upstream := newHostUpstream(t, map[string]string{order[1]: "sse_api_error", order[2]: "sse_error", order[3]: "content"})
	router := endpointStreamRouter(upstream.server.URL, true)
	stream := NewMockStream(nil)

	ctx, _ := routeStreamedEndpointTurn(t, router, endpointPickTags, nil)
	sendUpstreamChunks(t, router, ctx, stream, endpointStreamErrorChunk)
	drainSpliced(t, router, ctx, stream)

	var called []string
	for _, call := range upstream.calls() {
		called = append(called, providerList(t, call, "order")...)
	}
	assert.Equal(t, order[1:3], called)
	assert.Equal(t, order[:3], ctx.RaylineARCEndpointsTried)
	assert.NotEmpty(t, ctx.ResponseFailureClass)
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.NotContains(t, sent, "served on")
	lines := usageLines(logs)
	require.Len(t, lines, 3)
	assertFields(t, "pick", lines[0], map[string]interface{}{"endpoint_resend": endpointResendAttempt, "provider_endpoint": order[0]})
	assertFields(t, "second", lines[1], map[string]interface{}{
		"endpoint_resend": endpointResendAttempt, "provider_endpoint": order[1],
		"failure_class": turnFailureUpstream5xx, "failure_detail": "provider:api_error",
	})
	assertFields(t, "last", lines[2], map[string]interface{}{"endpoint_resend": endpointResendServed, "provider_endpoint": order[2]})
	// The last re-send ended before any content: an upstream error, not a
	// stall.
	assertFields(t, "last failure", lines[2], map[string]interface{}{
		"failure_class": turnFailureUpstreamError, "failure_detail": endpointResendExhaustedCode,
	})
}

// A turn the status path re-sent is answered whole: its in-band plan is
// dropped, so no stream failure can re-send it past the shared bound.
func TestStatusResendLeavesNoInBandResend(t *testing.T) {
	order := raylinearc.PickEndpoint(endpointPickKey, endpointPickTags)
	upstream := newEndpointUpstream(t, map[string]int{order[1]: http.StatusServiceUnavailable})
	router := endpointStreamRouter(upstream.server.URL, true)
	ctx, _ := routeEndpointTurn(t, router, endpointTurn{key: endpointPickKey, feed: endpointFeed(t, endpointPickTags...)})
	require.NotNil(t, ctx.HostRetry)
	require.NotNil(t, ctx.HostRetry.endpoint)

	reply := answerHeaders(t, router, ctx, http.StatusServiceUnavailable)

	require.NotNil(t, reply.GetImmediateResponse())
	assert.Equal(t, order[1:3], upstream.calls())
	assert.Equal(t, order[:3], ctx.RaylineARCEndpointsTried)
	assert.Nil(t, ctx.RaylineARCEndpointResend)
	assert.Nil(t, ctx.HostRetry, "the in-band plan survived the status re-send")
	assert.Empty(t, endpointResendTags(ctx), "the shared bound has room left")
	assert.False(t, endpointStreamRetries(turnFailureUpstreamError, "upstream_status_400"))
	assert.False(t, endpointStreamRetries(turnFailureTimeout, streamFirstContentTimeoutCode))
	assert.True(t, endpointStreamRetries(turnFailureUpstream5xx, "provider:api_error"))
}

// Frames held until the first content are released with it, and the served
// line says how long they were held.
func TestHeldFramesReleasedWithTheFirstContentRecordTheHold(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{})
	router := endpointStreamRouter(upstream.server.URL, true)
	stream := NewMockStream(nil)

	ctx, _ := routeStreamedEndpointTurn(t, router, endpointPickTags, nil)
	sendUpstreamChunks(t, router, ctx, stream, hostRetryPreContentChunk)
	sent, _ := clientBytes(stream)
	assert.NotContains(t, sent, "gen-stalled", "the opening frame was not held")
	time.Sleep(20 * time.Millisecond)
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(strings.ReplaceAll(hostRetryContentStream("answer"), "gen-served", "gen-stalled"), true), ctx))

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "gen-stalled", "the held frame was never released")
	assert.Contains(t, sent, "answer")
	assert.Empty(t, upstream.calls())
	lines := usageLines(logs)
	require.Len(t, lines, 1)
	hold, ok := lines[0]["stream_hold_ms"].(int64)
	require.True(t, ok, "stream_hold_ms = %#v", lines[0]["stream_hold_ms"])
	assert.GreaterOrEqual(t, hold, int64(20))
	assert.Nil(t, lines[0]["endpoint_resend"])
}

// An endpoint re-send in flight keeps its first-content limit: host retry's
// budget, counted from the turn's start, is not its bound.
func TestHostRetryBudgetDoesNotShortenAnEndpointReSend(t *testing.T) {
	router := hostRetryRouter("http://127.0.0.1:1", true, 30)
	router.Config.ResponseStreamHostRetryBudgetSec = 30
	ctx, _ := silentStreamContext(t, 0)
	ctx.RequestModel = hostRetryArm
	ctx.StartTime = time.Now().Add(-40 * time.Second)
	ctx.StreamContentSince = time.Now()
	ctx.HostRetry = &hostRetryPlan{splice: &hostRetrySplice{}, endpoint: &endpointResendPlan{}}
	assert.Equal(t, 30*time.Second, router.streamSilenceLimit(ctx))
}
