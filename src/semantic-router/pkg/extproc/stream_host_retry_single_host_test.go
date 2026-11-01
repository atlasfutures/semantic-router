//go:build !windows && cgo

package extproc

import (
	"bytes"
	"net/http"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// An arm whose pin names a single host (a privacy pin such as only:[azure])
// has no other host to move to. With response_stream.single_host_resend on, a
// turn that fails before any content is re-sent once on that same host, with
// the dispatched body unchanged.

const singleHost = "azure"

// hostRetryRateLimitChunk is an in-band rate limit before any content, as
// OpenRouter streams a provider's rate limit after answering 200 (prod logs
// it as provider:rate_limit_error).
const hostRetryRateLimitChunk = `data: {"error":{"type":"rate_limit_error","code":"rate_limit_error","message":"Rate limit exceeded"}}` + "\n\n"

func singleHostRouter(baseURL string, resend bool) *OpenAIRouter {
	router := hostRetryRouter(baseURL, true, 0, singleHost)
	router.Config.ResponseStreamSingleHostResend = resend
	return router
}

func shortenSameHostWaits(t *testing.T, wait, maxWait time.Duration) {
	t.Helper()
	oldWait, oldMax := sameHostRateLimitWait, sameHostRateLimitMaxWait
	sameHostRateLimitWait, sameHostRateLimitMaxWait = wait, maxWait
	t.Cleanup(func() { sameHostRateLimitWait, sameHostRateLimitMaxWait = oldWait, oldMax })
}

// A first-content stall on the only host is re-sent once on it, with the pin
// exactly as it was dispatched, and served.
func TestSingleHostStallIsReSentOnTheSameHost(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{singleHost: "content"})
	router := singleHostRouter(upstream.server.URL, true)
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, hostRetryPreContentChunk, openRouterKeepaliveChunk)
	require.True(t, ctx.HostRetry.started(), "the stall was not re-sent")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "served on "+singleHost)
	assert.NotContains(t, sent, "gen-stalled")
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.Equal(t, 0, transaction.aborts)

	calls := upstream.calls()
	require.Len(t, calls, 1)
	assert.Equal(t, []string{singleHost}, providerList(t, calls[0], "order"), "the re-send's pin was narrowed")
	assert.Equal(t, []string{"anthropic/flex"}, providerList(t, calls[0], "ignore"), "the re-send ignores its own host")

	lines := usageLines(logs)
	require.Len(t, lines, 2)
	assertFields(t, "stalled call", lines[0], map[string]interface{}{
		"host_retry": hostRetryAttempt, "host_attempt": 1, "upstream_host": singleHost,
		"failure_class": turnFailureTimeout,
	})
	assertFields(t, "served re-send", lines[1], map[string]interface{}{
		"host_retry": hostRetryServed, "host_attempt": 2, "upstream_host": singleHost,
	})
	event := findLogEvent(t, logs, "stream_host_retry")
	assertFields(t, "retry event", event, map[string]interface{}{"same_host": true, "wait_ms": 0, "attempt": 1})
}

// The same-host re-send is made once: when it too fails before any content,
// the turn ends as it would have without it.
func TestSingleHostReSendIsMadeOnlyOnce(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{singleHost: "sse_error"})
	router := singleHostRouter(upstream.server.URL, true)
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, openRouterKeepaliveChunk)
	drainSpliced(t, router, ctx, stream)

	assert.Len(t, upstream.calls(), 1, "the same host was asked more than once more")
	assert.NotEmpty(t, ctx.ResponseFailureClass)
	assert.Equal(t, 1, transaction.aborts)
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.NotContains(t, sent, "provider overloaded")
}

// With the switch off, a single-host arm stages no re-send: the stall is cut
// as before.
func TestSingleHostResendOffLeavesTheStallAsBefore(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{singleHost: "content"})
	router := singleHostRouter(upstream.server.URL, false)
	stream := NewMockStream(nil)

	ctx, _ := startStalledTurn(t, router, stream, openRouterKeepaliveChunk)
	assert.Nil(t, ctx.HostRetry)
	assert.True(t, ctx.StreamingComplete)
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
	assert.Empty(t, upstream.calls())
}

// An in-band rate limit before any content on the only host waits before the
// same-host re-send dials, and the client never sees the error.
func TestInBandRateLimitWaitsBeforeTheSameHostReSend(t *testing.T) {
	shortenSameHostWaits(t, 150*time.Millisecond, time.Second)
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{singleHost: "content"})
	router := singleHostRouter(upstream.server.URL, true)
	stream := NewMockStream(nil)

	ctx, transaction := routeFlexTurn(t, router, hostRetryArm, true)
	ctx.ProcessingStartTime = ctx.StartTime
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, false), ctx))
	limited := time.Now()
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(hostRetryRateLimitChunk, false), ctx))
	require.True(t, ctx.HostRetry.started(), "the rate limit was not re-sent")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "served on "+singleHost)
	assert.NotContains(t, sent, "Rate limit exceeded")
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.GreaterOrEqual(t, time.Since(limited), 150*time.Millisecond, "the re-send did not wait")
	_ = transaction
	require.Len(t, upstream.calls(), 1)
	event := findLogEvent(t, logs, "stream_host_retry")
	assertFields(t, "retry event", event, map[string]interface{}{"same_host": true, "wait_ms": 150, "trigger": "stream_error"})
}

// The wait follows the provider's stated delay, bounded; anything but a rate
// limit, and any plan with another host, dials at once.
func TestSameHostResendWait(t *testing.T) {
	shortenSameHostWaits(t, 2*time.Second, 10*time.Second)
	limited := func(retryAfter int64) *RequestContext {
		return &RequestContext{SemanticStreamState: &semanticResponseStreamState{
			upstreamFailed: &llmprotocol.ProtocolError{Category: llmprotocol.ErrorRateLimited, RetryAfter: retryAfter},
		}}
	}
	same := &hostRetryPlan{sameHost: true, hosts: []string{singleHost}}
	assert.Equal(t, 2*time.Second, same.resendWait(limited(0)))
	assert.Equal(t, 4*time.Second, same.resendWait(limited(4)))
	assert.Equal(t, 10*time.Second, same.resendWait(limited(60)))
	other := &RequestContext{SemanticStreamState: &semanticResponseStreamState{
		upstreamFailed: &llmprotocol.ProtocolError{Category: llmprotocol.ErrorUpstreamUnavailable},
	}}
	assert.Zero(t, same.resendWait(other))
	assert.Zero(t, same.resendWait(&RequestContext{}))
	multi := &hostRetryPlan{hosts: []string{hostRetryHostA, hostRetryHostB}}
	assert.Zero(t, multi.resendWait(limited(4)))
}

// A wait is not spent from the re-send's first-content limit: the re-send of
// an arm whose limit is shorter than the wait still dials and serves.
func TestSameHostWaitDoesNotSpendTheFirstContentLimit(t *testing.T) {
	shortenSameHostWaits(t, 300*time.Millisecond, time.Second)
	upstream := newHostUpstream(t, map[string]string{singleHost: "content"})
	router := singleHostRouter(upstream.server.URL, true)
	stream := NewMockStream(nil)

	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	ctx.ProcessingStartTime = ctx.StartTime
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(hostRetryRateLimitChunk, false), ctx))
	require.True(t, ctx.HostRetry.started())
	// Mid-wait, with a 90 s limit and a clock that starts at the dial, the
	// re-send has not been silent at all yet.
	assert.False(t, router.streamSilenceExceeded(ctx))
	assert.True(t, ctx.StreamContentSince.After(time.Now()), "the first-content clock started before the dial")
	drainSpliced(t, router, ctx, stream)
	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "served on "+singleHost)
}

// A wait that would carry the re-send past the turn's deadline dials nothing:
// the rate limit ends the turn as it did before.
func TestSameHostWaitPastTheDeadlineDialsNothing(t *testing.T) {
	shortenSameHostWaits(t, 2*time.Second, 10*time.Second)
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{singleHost: "content"})
	router := singleHostRouter(upstream.server.URL, true)
	router.Config.ResponseStreamDeadlineSec = 600
	stream := NewMockStream(nil)

	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	ctx.ProcessingStartTime = ctx.StartTime
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, false), ctx))
	// One second of the turn's 600 s is left, and the wait is two.
	ctx.StartTime = time.Now().Add(-599 * time.Second)
	ctx.StreamContentSince = time.Now()
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(hostRetryRateLimitChunk, false), ctx))

	assert.False(t, ctx.HostRetry.started(), "a re-send was scheduled past the deadline")
	assert.Empty(t, upstream.calls())
	event := findLogEvent(t, logs, "stream_host_retry_unavailable")
	assertFields(t, "unavailable", event, map[string]interface{}{"reason": "deadline_within_wait", "wait_ms": 2000})
}

// The privacy invariant: whatever ends the first call before content, the
// same-host re-send carries the exact bytes that call dispatched, so its
// provider object (only, ignore, order, allow_fallbacks, zdr,
// data_collection) is the one privacy routing cleared. A change that rebuilds
// the provider object for the re-send must fail here.
func TestSameHostReSendSendsTheDispatchedBytes(t *testing.T) {
	cases := map[string][]string{
		"first-content stall": {hostRetryPreContentChunk, openRouterKeepaliveChunk},
		"in-band rate limit":  {openRouterKeepaliveChunk, hostRetryRateLimitChunk},
	}
	for name, chunks := range cases {
		t.Run(name, func(t *testing.T) {
			shortenSameHostWaits(t, time.Millisecond, time.Millisecond)
			upstream := newHostUpstream(t, map[string]string{singleHost: "content"})
			router := singleHostRouter(upstream.server.URL, true)
			stream := NewMockStream(nil)

			ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
			require.NotNil(t, ctx.HostRetry, "no same-host plan was staged")
			dispatched := bytes.Clone(ctx.HostRetry.dispatched)
			require.NotEmpty(t, dispatched)
			ctx.ProcessingStartTime = ctx.StartTime
			require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
			for _, chunk := range chunks {
				require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
			}
			if !ctx.HostRetry.started() {
				ctx.StreamContentSince = time.Now().Add(-91 * time.Second)
				require.True(t, router.streamSilenceExceeded(ctx))
				require.NoError(t, router.endSilentStream(stream, ctx))
			}
			require.True(t, ctx.HostRetry.started(), "nothing was re-sent")
			drainSpliced(t, router, ctx, stream)

			received := upstream.received()
			require.Len(t, received, 1)
			assert.Equal(t, string(dispatched), string(received[0]), "the same-host re-send changed the dispatched body")
		})
	}
}
