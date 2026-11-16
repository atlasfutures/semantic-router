//go:build !windows && cgo

package extproc

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The no-content re-send (stream_host_retry.go): with host retry off, an
// OpenRouter turn whose call fails before any content is sent again once,
// without the host that failed when the stream named one.

// noContentNamedChunk is a response-start frame that names the host serving
// the call: no content, so it is held from the client.
const noContentNamedChunk = `data: {"id":"gen-stalled","object":"chat.completion.chunk","created":1,` +
	`"model":"anthropic/claude-haiku-5.5","provider":"Anthropic",` +
	`"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n"

// startIncompleteTurn routes a streamed turn, answers its headers with a
// 200 stream and sends it chunks, the last of them ending the stream.
func startIncompleteTurn(t *testing.T, router *OpenAIRouter, stream *MockStream, chunks ...string) (*RequestContext, *recordingSelectionTransaction) {
	t.Helper()
	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	ctx.ProcessingStartTime = ctx.StartTime
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	transaction := &recordingSelectionTransaction{}
	ctx.SelectionTransaction = newSelectionTransactionOwner(configRaylineARC, transaction)
	for index, chunk := range chunks {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, index == len(chunks)-1), ctx))
	}
	return ctx, transaction
}

// (a) A first-content timeout with nothing sent and no host named: the turn
// is re-sent once with the dispatched pin unchanged, and the re-send serves.
func TestNoContentTimeoutIsReSentOnce(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{hostRetryHostA: "content"})
	router := hostRetryRouter(upstream.server.URL, false, 0)
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, hostRetryPreContentChunk, openRouterKeepaliveChunk)
	require.True(t, ctx.HostRetry.started(), "the stall was not re-sent")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "served on "+hostRetryHostA)
	assert.NotContains(t, sent, "gen-stalled", "the stalled call's held frame reached the client")
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.Equal(t, 0, transaction.aborts)

	calls := upstream.calls()
	require.Len(t, calls, 1)
	assert.Equal(t, []string{hostRetryHostA, hostRetryHostB}, providerList(t, calls[0], "order"))
	assert.Equal(t, []string{"anthropic/flex"}, providerList(t, calls[0], "ignore"))

	lines := usageLines(logs)
	require.Len(t, lines, 2, "want one usage line per call: %v", lines)
	assertFields(t, "stalled call", lines[0], map[string]interface{}{
		"host_retry": hostRetryAttempt, "host_attempt": 1, "failure_class": turnFailureTimeout,
		"failure_detail": streamFirstContentTimeoutCode, "content_sent_before_failure": false,
		"host_retry_reason": streamFirstContentTimeoutCode,
	})
	assert.Nil(t, lines[0]["upstream_host"])
	assertFields(t, "served re-send", lines[1], map[string]interface{}{
		"host_retry": hostRetryServed, "host_attempt": 2, "host_retry_reason": streamFirstContentTimeoutCode,
		"upstream_attempts": 1,
	})
	assert.Nil(t, lines[1]["failure_class"])
	assert.Nil(t, lines[1]["host_retry_excluded"])
	assertFields(t, "retry event", findLogEvent(t, logs, "stream_host_retry"), map[string]interface{}{
		"once": true, "trigger": streamFirstContentTimeoutCode, "attempt": 1,
	})
}

// (b) A stream that ends before any content without its terminal event is
// re-sent once; when the re-send fails too, nothing more is sent and the
// turn fails with nothing delivered and both calls logged.
func TestNoContentIncompleteStreamIsReSentOnlyOnce(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{hostRetryHostA: "keepalive_close"})
	router := hostRetryRouter(upstream.server.URL, false, 0)
	stream := NewMockStream(nil)

	ctx, transaction := startIncompleteTurn(t, router, stream, hostRetryPreContentChunk, "")
	require.True(t, ctx.HostRetry.started(), "the incomplete stream was not re-sent")
	drainSpliced(t, router, ctx, stream)

	require.True(t, ctx.StreamingComplete)
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.NotContains(t, sent, "gen-stalled")
	assert.NotEmpty(t, ctx.ResponseFailureClass)
	require.NotNil(t, ctx.ContentSentBeforeFailure)
	assert.False(t, *ctx.ContentSentBeforeFailure)
	assert.Equal(t, 1, transaction.aborts)
	assert.Len(t, upstream.calls(), 1, "the turn was not re-sent exactly once")

	lines := usageLines(logs)
	require.Len(t, lines, 2, "want one usage line per call: %v", lines)
	assertFields(t, "incomplete call", lines[0], map[string]interface{}{
		"host_retry": hostRetryAttempt, "host_attempt": 1, "failure_class": turnFailureUpstream5xx,
		"failure_detail": "stream_incomplete", "content_sent_before_failure": false,
		"host_retry_reason": "stream_incomplete",
	})
	assertFields(t, "failed re-send", lines[1], map[string]interface{}{
		"host_retry": hostRetryServed, "host_attempt": 2, "content_sent_before_failure": false,
		"host_retry_reason": "stream_incomplete",
	})
	assert.NotNil(t, lines[1]["failure_class"])
	assertFields(t, "turn_failed", findLogEvent(t, logs, "turn_failed"), map[string]interface{}{
		"host_attempt": 2, "host_retry_reason": "stream_incomplete", "content_sent_before_failure": false,
	})
}

// (c) Content reached the client before the stream was cut: never re-sent.
func TestContentSentThenCutIsNotReSent(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{hostRetryHostA: "content"})
	router := hostRetryRouter(upstream.server.URL, false, 0)
	stream := NewMockStream(nil)

	ctx, _ := startIncompleteTurn(t, router, stream, endlessUpstreamChunk(1), "")

	assert.False(t, ctx.HostRetry.started())
	assert.True(t, ctx.StreamingComplete)
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "still thinking")
	assert.Equal(t, turnFailureUpstream5xx, ctx.ResponseFailureClass)
	require.NotNil(t, ctx.ContentSentBeforeFailure)
	assert.True(t, *ctx.ContentSentBeforeFailure)
	assert.Empty(t, upstream.calls())
}

// (d) The stream named the pin's host that failed: the re-send leaves it out
// of order and adds it to ignore, as host retry does.
func TestNoContentReSendExcludesTheNamedHost(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{hostRetryHostB: "content"})
	router := hostRetryRouter(upstream.server.URL, false, 0)
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, noContentNamedChunk)
	require.True(t, ctx.HostRetry.started())
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "served on "+hostRetryHostB)
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.Equal(t, 0, transaction.aborts)
	calls := upstream.calls()
	require.Len(t, calls, 1)
	assert.Equal(t, []string{hostRetryHostB}, providerList(t, calls[0], "order"))
	assert.Equal(t, []string{"anthropic/flex", hostRetryHostA}, providerList(t, calls[0], "ignore"))

	lines := usageLines(logs)
	require.Len(t, lines, 2)
	assertFields(t, "stalled call", lines[0], map[string]interface{}{
		"host_retry": hostRetryAttempt, "upstream_host": hostRetryHostA, "upstream_host_source": hostSourceResponse,
	})
	if excluded := lines[1]["host_retry_excluded"]; !sameJSON(t, mustJSONRaw(t, excluded), `["anthropic"]`) {
		t.Fatalf("host_retry_excluded = %v", excluded)
	}
}

// (e) Too little of the turn's deadline is left for a re-send to reach its
// first content: the stall is cut as before and nothing is dialled.
func TestNoContentReSendSkippedNearTheDeadline(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{hostRetryHostA: "content"})
	router := hostRetryRouter(upstream.server.URL, false, 0)
	// 60 s left of the turn against a 90 s first-content limit.
	router.Config.ResponseStreamDeadlineSec = 60
	stream := NewMockStream(nil)

	ctx, transaction := startStalledTurn(t, router, stream, hostRetryPreContentChunk)

	assert.False(t, ctx.HostRetry.started())
	assert.True(t, ctx.StreamingComplete)
	_, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
	assert.Equal(t, 1, transaction.aborts)
	assert.Empty(t, upstream.calls())
	assertFields(t, "skipped", findLogEvent(t, logs, "stream_host_retry_unavailable"), map[string]interface{}{
		"reason": "deadline_too_short",
	})
}
