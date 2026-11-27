//go:build !windows && cgo

package extproc

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// An arm with head_retry on holds its stream's first completion tokens from
// the client. A stream the upstream ends without its terminal event inside
// that window is re-sent once, unchanged, and the re-send serves the client.

func headChunk(id, text string) string {
	return `data: {"id":"` + id + `","object":"chat.completion.chunk","created":1,"model":"anthropic/claude-haiku-5.5",` +
		`"choices":[{"index":0,"delta":{"role":"assistant","content":"` + text + `"},"finish_reason":null}]}` + "\n\n"
}

func headStop(id string) string {
	return `data: {"id":"` + id + `","object":"chat.completion.chunk","created":1,"model":"anthropic/claude-haiku-5.5",` +
		`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}` + "\n\n" + "data: [DONE]\n\n"
}

// headUpstream is a fake OpenRouter that answers its nth call with the nth
// body, as a 200 stream, or with a 503 when the body is "503".
type headUpstream struct {
	server *httptest.Server
	mu     sync.Mutex
	bodies [][]byte
}

func newHeadUpstream(t *testing.T, answers ...string) *headUpstream {
	t.Helper()
	upstream := &headUpstream{}
	upstream.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received, _ := io.ReadAll(r.Body)
		upstream.mu.Lock()
		index := len(upstream.bodies)
		upstream.bodies = append(upstream.bodies, received)
		upstream.mu.Unlock()
		answer := "503"
		if index < len(answers) {
			answer = answers[index]
		}
		if answer == "503" {
			w.Header().Set("content-type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":{"message":"no capacity"}}`)
			return
		}
		w.Header().Set("content-type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, answer)
	}))
	t.Cleanup(upstream.server.Close)
	return upstream
}

func (upstream *headUpstream) received() [][]byte {
	upstream.mu.Lock()
	defer upstream.mu.Unlock()
	return append([][]byte(nil), upstream.bodies...)
}

// headRetryRouter is a cell whose one arm pins a single host with host retry
// off, and has head_retry on with the given window (0: the default) when on.
func headRetryRouter(baseURL string, on bool, tokens int) *OpenAIRouter {
	router := hostRetryRouter(baseURL, false, 0, singleHost)
	params := router.Config.ModelConfig[hostRetryArm]
	params.HeadRetry = on
	params.HeadRetryTokens = tokens
	router.Config.ModelConfig[hostRetryArm] = params
	return router
}

// runHeadTurn routes a streamed turn to the arm, answers its headers with a
// 200 stream and sends it the chunks, the last one marked as the end.
func runHeadTurn(t *testing.T, router *OpenAIRouter, stream *MockStream, chunks ...string) *RequestContext {
	t.Helper()
	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	ctx.ProcessingStartTime = ctx.StartTime
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.True(t, ctx.IsStreamingResponse)
	for index, chunk := range chunks {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, index == len(chunks)-1), ctx))
	}
	return ctx
}

func lastUsageLine(t *testing.T, logs *observer.ObservedLogs) map[string]interface{} {
	t.Helper()
	lines := usageLines(logs)
	require.NotEmpty(t, lines)
	return lines[len(lines)-1]
}

// Off, an early end is today's failure: the content reached the client, the
// turn ends stream_incomplete, nothing is re-sent and no plan is staged.
func TestHeadRetryOffLeavesAnEarlyEndAsBefore(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHeadUpstream(t, headChunk("gen-served", "served")+headStop("gen-served"))
	router := headRetryRouter(upstream.server.URL, false, 32)
	stream := NewMockStream(nil)

	ctx := runHeadTurn(t, router, stream, headChunk("gen-cut", "Hi"), openRouterKeepaliveChunk)

	assert.Nil(t, ctx.HostRetry, "a plan was staged with head_retry off")
	assert.True(t, ctx.StreamingComplete)
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, `"content":"Hi"`)
	assert.NotContains(t, sent, ": ROUTER PROCESSING")
	assert.Equal(t, turnFailureUpstream5xx, ctx.ResponseFailureClass)
	assert.Equal(t, "stream_incomplete", ctx.ResponseFailureDetail)
	require.NotNil(t, ctx.ContentSentBeforeFailure)
	assert.True(t, *ctx.ContentSentBeforeFailure)
	assert.Empty(t, upstream.received())
	line := lastUsageLine(t, logs)
	for _, field := range []string{"head_retry", "head_retry_tokens", "head_retry_buffered_tokens"} {
		assert.Nil(t, line[field], field)
	}
}

// A cut inside the window is re-sent once with the dispatched bytes, and the
// client is served the re-send's stream with none of the cut call's content.
func TestHeadRetryReSendsACutInsideTheWindow(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHeadUpstream(t, headChunk("gen-served", "served again")+headStop("gen-served"))
	router := headRetryRouter(upstream.server.URL, true, 0)
	stream := NewMockStream(nil)

	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	require.NotNil(t, ctx.HostRetry)
	assert.Equal(t, config.DefaultStreamHeadRetryTokens, ctx.HostRetry.headTokens)
	dispatched := string(ctx.HostRetry.dispatched)
	ctx.ProcessingStartTime = ctx.StartTime
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(headChunk("gen-cut", "Hi"), false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, true), ctx))
	require.True(t, ctx.HostRetry.started(), "the early end was not re-sent")
	assert.False(t, ctx.StreamingComplete)
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "served again")
	assert.NotContains(t, sent, "gen-cut")
	assert.NotContains(t, sent, `"content":"Hi"`)
	assert.NotContains(t, sent, "stream_incomplete")
	assert.Empty(t, ctx.ResponseFailureClass)
	received := upstream.received()
	require.Len(t, received, 1)
	assert.Equal(t, dispatched, string(received[0]), "the head re-send changed the dispatched body")

	lines := usageLines(logs)
	require.Len(t, lines, 2)
	assertFields(t, "cut call", lines[0], map[string]interface{}{
		"head_retry": headRetryAttempt, "head_retry_tokens": 8, "head_retry_buffered_tokens": 1,
		"failure_class": turnFailureUpstream5xx, "failure_detail": "stream_incomplete",
		"content_sent_before_failure": false,
	})
	assertFields(t, "served re-send", lines[1], map[string]interface{}{
		"head_retry": headRetryServed, "head_retry_tokens": 8, "head_retry_buffered_tokens": 1,
	})
	assert.Nil(t, lines[1]["failure_class"])
	event := findLogEvent(t, logs, "stream_head_retry")
	assertFields(t, "retry event", event, map[string]interface{}{"buffered_tokens": 1, "window_tokens": 8})
}

// Past the window the held frames are sent, and a later cut is today's
// failure: no re-send.
func TestHeadRetryLeavesACutAfterTheWindowAsBefore(t *testing.T) {
	upstream := newHeadUpstream(t, headChunk("gen-served", "served")+headStop("gen-served"))
	router := headRetryRouter(upstream.server.URL, true, 8)
	stream := NewMockStream(nil)

	long := strings.Repeat("abcd", 10) // 10 tokens by the Router's estimate
	ctx := runHeadTurn(t, router, stream, headChunk("gen-cut", "Hi"), headChunk("gen-cut", long), openRouterKeepaliveChunk)

	assert.False(t, ctx.HostRetry.started())
	assert.True(t, ctx.StreamingComplete)
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, `"content":"Hi"`)
	assert.Contains(t, sent, long)
	assert.Equal(t, turnFailureUpstream5xx, ctx.ResponseFailureClass)
	assert.Equal(t, "stream_incomplete", ctx.ResponseFailureDetail)
	require.NotNil(t, ctx.ContentSentBeforeFailure)
	assert.True(t, *ctx.ContentSentBeforeFailure)
	assert.Empty(t, upstream.received())
}

// A re-send that is itself cut after content ends the turn as any cut does,
// and is not re-sent again.
func TestHeadRetryReSendCutIsTodaysFailure(t *testing.T) {
	upstream := newHeadUpstream(t, headChunk("gen-recut", "partial"), headChunk("gen-third", "third")+headStop("gen-third"))
	router := headRetryRouter(upstream.server.URL, true, 32)
	stream := NewMockStream(nil)

	ctx := runHeadTurn(t, router, stream, headChunk("gen-cut", "Hi"), openRouterKeepaliveChunk)
	require.True(t, ctx.HostRetry.started())
	drainSpliced(t, router, ctx, stream)

	assert.Len(t, upstream.received(), 1, "the turn was re-sent more than once")
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "partial")
	assert.NotContains(t, sent, "third")
	assert.Equal(t, turnFailureUpstream5xx, ctx.ResponseFailureClass)
	assert.Equal(t, "stream_incomplete", ctx.ResponseFailureDetail)
	require.NotNil(t, ctx.ContentSentBeforeFailure)
	assert.True(t, *ctx.ContentSentBeforeFailure)
}

// A re-send that fails before any content ends the turn with an error, and
// is not re-sent again.
func TestHeadRetryReSendRefusedEndsTheTurn(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHeadUpstream(t, "503", headChunk("gen-third", "third")+headStop("gen-third"))
	router := headRetryRouter(upstream.server.URL, true, 0)
	stream := NewMockStream(nil)

	ctx := runHeadTurn(t, router, stream, headChunk("gen-cut", "Hi"), openRouterKeepaliveChunk)
	require.True(t, ctx.HostRetry.started())
	drainSpliced(t, router, ctx, stream)

	assert.Len(t, upstream.received(), 1, "the turn was re-sent more than once")
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.NotContains(t, sent, `"content":"Hi"`)
	assert.NotContains(t, sent, "third")
	assert.Equal(t, turnFailureUpstream5xx, ctx.ResponseFailureClass)
	assert.Equal(t, streamHeadRetryFailedCode, ctx.ResponseFailureDetail)
	turn := findLogEvent(t, logs, "turn_failed")
	assertFields(t, "turn_failed", turn, map[string]interface{}{
		"head_retry": headRetryServed, "head_retry_tokens": 8, "head_retry_buffered_tokens": 1,
	})
}

// A stream that ends cleanly inside the window is sent whole, and nothing is
// re-sent.
func TestHeadRetryFlushesACleanShortStream(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHeadUpstream(t)
	router := headRetryRouter(upstream.server.URL, true, 32)
	stream := NewMockStream(nil)

	ctx := runHeadTurn(t, router, stream, headChunk("gen-short", "Hi"), headStop("gen-short"))

	assert.False(t, ctx.HostRetry.started())
	assert.True(t, ctx.StreamingComplete)
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, `"content":"Hi"`)
	assert.Contains(t, sent, `"finish_reason":"stop"`)
	assert.Contains(t, sent, "[DONE]")
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.Empty(t, upstream.received())
	line := lastUsageLine(t, logs)
	assertFields(t, "usage", line, map[string]interface{}{"head_retry_tokens": 32})
	assert.Nil(t, line["head_retry"])
}

// A body that ends at its trailers is re-sent the same way, and the
// trailers' reply waits for the re-send's end.
func TestHeadRetryReSendsACutEndedAtTheTrailers(t *testing.T) {
	upstream := newHeadUpstream(t, headChunk("gen-served", "served again")+headStop("gen-served"))
	router := headRetryRouter(upstream.server.URL, true, 0)
	stream := NewMockStream(nil)

	ctx, _ := routeFlexTurn(t, router, hostRetryArm, true)
	ctx.ProcessingStartTime = ctx.StartTime
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(headChunk("gen-cut", "Hi"), false), ctx))
	require.NoError(t, router.processResponseTrailers(stream, &ext_proc.ProcessingRequest_ResponseTrailers{}, ctx))
	require.True(t, ctx.HostRetry.started(), "the early end was not re-sent")
	require.True(t, ctx.HostRetry.trailersPending)
	drainSpliced(t, router, ctx, stream)

	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "served again")
	assert.NotContains(t, sent, `"content":"Hi"`)
	assert.Empty(t, ctx.ResponseFailureClass)
	require.NotEmpty(t, stream.Responses)
	assert.NotNil(t, stream.Responses[len(stream.Responses)-1].GetResponseTrailers(), "the trailers were not answered last")
	assert.Len(t, upstream.received(), 1)
}
