//go:build !windows && cgo

package extproc

import (
	"net/http"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// A held turn's headers reply waits for the hold, so the selected model it
// names is the arm that served.

// startServedHeadersTurn starts a held reserve turn on a stream that defers
// its headers, as processWithContext wraps every stream.
func startServedHeadersTurn(t *testing.T, router *OpenAIRouter) (*RequestContext, *MockStream, ext_proc.ExternalProcessor_ProcessServer) {
	t.Helper()
	mock := NewMockStream(nil)
	var stream ext_proc.ExternalProcessor_ProcessServer
	ctx, _ := startReserveTurnOn(t, router, func(ctx *RequestContext) ext_proc.ExternalProcessor_ProcessServer {
		// Envoy states the full-duplex mode on the stream's first message,
		// ahead of the response headers.
		ctx.FullDuplexResponseBody = true
		stream = router.newServedHeadersStream(mock, ctx)
		return stream
	})
	return ctx, mock, stream
}

// headersReplies are the headers replies sent, and the index of the first.
func headersReplies(mock *MockStream) ([]*ext_proc.HeadersResponse, int) {
	var replies []*ext_proc.HeadersResponse
	first := -1
	for i, response := range mock.Responses {
		if reply := response.GetResponseHeaders(); reply != nil {
			if first < 0 {
				first = i
			}
			replies = append(replies, reply)
		}
	}
	return replies, first
}

func selectedModelOf(reply *ext_proc.HeadersResponse) string {
	return headerValueForTest(reply.GetResponse().GetHeaderMutation(), headers.VSRSelectedModel)
}

func TestReservedTurnHeadersNameTheServedArm(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	ctx, mock, stream := startServedHeadersTurn(t, router)
	assert.Empty(t, mock.Responses, "the headers reply was not deferred over the hold")

	for _, chunk := range []string{reserveRoleChunk, reserveThinkingChunk} {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	require.NoError(t, sendRefusalReserveKeepAlive(stream, ctx))
	assert.Empty(t, mock.Responses, "a held placeholder or keepalive was sent ahead of the headers")

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	require.True(t, ctx.HostRetry.started(), "the refusal was not re-served")
	drainSpliced(t, router, ctx, stream)

	replies, first := headersReplies(mock)
	require.Len(t, replies, 1, "want exactly one headers reply")
	assert.Equal(t, 0, first, "the headers reply did not come first")
	assert.Equal(t, reserveGLMArm, selectedModelOf(replies[0]))
	sent, ended := clientBytes(mock)
	assert.True(t, ended)
	assert.Contains(t, sent, "served on "+reserveGLMHost)
}

func TestReleasedTurnHeadersNameTheHeldArm(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{})
	router := reserveRouter(upstream.server.URL)
	ctx, mock, stream := startServedHeadersTurn(t, router)

	for _, chunk := range []string{reserveRoleChunk, reserveTextChunk} {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	replies, first := headersReplies(mock)
	require.Len(t, replies, 1)
	assert.Equal(t, 0, first)
	assert.Equal(t, reserveClaudeArm, selectedModelOf(replies[0]))
	sent, _ := clientBytes(mock)
	assert.Contains(t, sent, "partial answer")
}

// Once the hold lapses the arm is settled, so the next keepalive goes out,
// headers first: the client's wait for headers ends with the hold.
func TestLapsedHoldKeepaliveSendsTheHeaders(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{})
	router := reserveRouter(upstream.server.URL)
	ctx, mock, stream := startServedHeadersTurn(t, router)
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	assert.Empty(t, mock.Responses)

	ctx.StartTime = time.Now().Add(-ctx.RefusalReserve.hold - time.Second)
	require.False(t, router.holdingForRefusalReserve(ctx))
	require.NoError(t, sendRefusalReserveKeepAlive(stream, ctx))
	replies, first := headersReplies(mock)
	require.Len(t, replies, 1)
	assert.Equal(t, 0, first)
	assert.Equal(t, reserveClaudeArm, selectedModelOf(replies[0]))
	sent, _ := clientBytes(mock)
	assert.Equal(t, string(hostRetryHoldFrame), sent)
}

// An immediate response answers in place of the deferred headers.
func TestImmediateResponseReplacesDeferredHeaders(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{})
	router := reserveRouter(upstream.server.URL)
	_, mock, stream := startServedHeadersTurn(t, router)
	require.NoError(t, stream.Send(router.createErrorResponse(http.StatusBadGateway, "failed")))
	require.Len(t, mock.Responses, 1)
	assert.NotNil(t, mock.Responses[0].GetImmediateResponse())
	require.NoError(t, stream.Send(heldResponseBodyChunk()))
	assert.Len(t, mock.Responses, 2, "the stream still deferred after the immediate response")
}

// A turn that is not held answers its headers at once.
func TestUnheldTurnHeadersAreNotDeferred(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{})
	router := reserveRouter(upstream.server.URL)
	mock := NewMockStream(nil)
	ctx, _ := routeFlexTurn(t, router, reserveClaudeArm, true)
	ctx.VSRSelectedModel = reserveClaudeArm
	ctx.FullDuplexResponseBody = true
	stream := router.newServedHeadersStream(mock, ctx)
	require.NoError(t, router.processResponseHeaders(stream, flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	replies, first := headersReplies(mock)
	require.Len(t, replies, 1)
	assert.Equal(t, 0, first)
	assert.Equal(t, reserveClaudeArm, selectedModelOf(replies[0]))
}
