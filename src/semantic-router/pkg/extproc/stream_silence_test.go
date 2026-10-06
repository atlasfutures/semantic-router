package extproc

import (
	"bytes"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// semantic-router #210: an arm that accepts a streamed turn and then sends
// only keepalives is cut at the first-content limit, as a timeout with nothing
// sent, and its episode lease is released at once.

const openRouterKeepaliveChunk = ": OPENROUTER PROCESSING\n\n"

func silenceRouter(firstContentSec, idleSec int) *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{RouterOptions: config.RouterOptions{
		ResponseStreamFirstContentSec: firstContentSec,
		ResponseStreamIdleSec:         idleSec,
	}}}
}

// silentStreamContext is a full-duplex streamed turn whose upstream headers
// arrived silentFor ago, holding an uncommitted selection.
func silentStreamContext(t *testing.T, silentFor time.Duration) (*RequestContext, *recordingSelectionTransaction) {
	t.Helper()
	ctx := &RequestContext{Headers: make(map[string]string)}
	seedOverrunningStreamContext(t, ctx, silentFor)
	ctx.StreamContentSince = time.Now().Add(-silentFor)
	transaction := &recordingSelectionTransaction{}
	ctx.SelectionTransaction = newSelectionTransactionOwner(configRaylineARC, transaction)
	return ctx, transaction
}

func TestKeepalivesPastTheFirstContentLimitCutTheTurn(t *testing.T) {
	router := silenceRouter(60, 0)
	ctx, transaction := silentStreamContext(t, 90*time.Second)
	stream := NewMockStream(nil)

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, false), ctx))

	require.Len(t, stream.Responses, 1)
	streamed := stream.Responses[0].GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.True(t, streamed.GetEndOfStream(), "the silent turn was left open")
	if !bytes.Contains(streamed.GetBody(), []byte(`"type":"error"`)) {
		t.Fatalf("the cut carried no error frame:\n%s", streamed.GetBody())
	}
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass, "a silent arm must be classed a timeout so it is excluded")
	require.NotNil(t, ctx.ContentSentBeforeFailure)
	assert.False(t, *ctx.ContentSentBeforeFailure)
	assert.Equal(t, 1, transaction.aborts, "the lease must be released when the turn is cut, not when the exchange closes")
	assert.Equal(t, 0, transaction.commits)
}

func TestContentArrivingLateIsNotCut(t *testing.T) {
	router := silenceRouter(60, 0)
	ctx, transaction := silentStreamContext(t, 90*time.Second)
	stream := NewMockStream(nil)

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(endlessUpstreamChunk(1), false), ctx))

	streamed := stream.Responses[0].GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	assert.False(t, streamed.GetEndOfStream(), "a chunk that carries content was cut for the silence before it")
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.True(t, ctx.StreamContentSeen)
	assert.WithinDuration(t, time.Now(), ctx.StreamContentSince, 5*time.Second)
	assert.Equal(t, 0, transaction.aborts)
}

func TestSilenceAfterContentIsCutAtTheIdleLimit(t *testing.T) {
	router := silenceRouter(60, 30)
	ctx, _ := silentStreamContext(t, 45*time.Second)
	ctx.StreamContentSeen = true
	stream := NewMockStream(nil)
	// The idle limit (30 s) applies once content was seen, not the first-content one (60 s).
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, false), ctx))
	streamed := stream.Responses[0].GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	assert.True(t, streamed.GetEndOfStream())
	assert.Contains(t, string(streamed.GetBody()), "stopped sending content")
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
}

func TestSilenceLimitsAreOffByDefault(t *testing.T) {
	router := silenceRouter(0, 0)
	ctx, transaction := silentStreamContext(t, 300*time.Second)
	ctx.StartTime = time.Now().Add(-300 * time.Second) // short of the 590 s stream deadline
	stream := NewMockStream(nil)
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, false), ctx))
	streamed := stream.Responses[0].GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	assert.False(t, streamed.GetEndOfStream())
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.Equal(t, 0, transaction.aborts)
	assert.Zero(t, router.loopWait(ctx))
}

// With nothing arriving at all, the loop's own timer ends the turn.
func TestTheLoopTimerEndsASilentTurnWithNoChunkToAnswer(t *testing.T) {
	router := silenceRouter(60, 0)
	ctx, transaction := silentStreamContext(t, 30*time.Second)
	ctx.FullDuplexResponseBody = true
	assert.Zero(t, router.loopWait(ctx), "the timer must not be armed before Envoy streams the body")
	// The first body message: a keepalive, inside the limit.
	require.NoError(t, router.handleProcessRequest(NewMockStream(nil), fullDuplexResponseBodyRequest(openRouterKeepaliveChunk, false), ctx))
	require.False(t, ctx.StreamingComplete)
	wait := router.loopWait(ctx)
	assert.InDelta(t, (30 * time.Second).Seconds(), wait.Seconds(), 2, "the loop must wake when the limit falls due")

	ctx.StreamContentSince = time.Now().Add(-61 * time.Second)
	require.True(t, router.streamSilenceExceeded(ctx))
	stream := NewMockStream(nil)
	require.NoError(t, router.endSilentStream(stream, ctx))
	require.Len(t, stream.Responses, 1)
	streamed := stream.Responses[0].GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.True(t, streamed.GetEndOfStream())
	assert.Equal(t, turnFailureTimeout, ctx.ResponseFailureClass)
	assert.Equal(t, 1, transaction.aborts)
	assert.Zero(t, router.loopWait(ctx), "an ended turn must not wake the loop again")
}

// Without full duplex no reply can be sent unprompted, so only the next chunk
// can end the turn.
func TestTheLoopTimerIsFullDuplexOnly(t *testing.T) {
	router := silenceRouter(60, 0)
	ctx, _ := silentStreamContext(t, 30*time.Second)
	assert.Zero(t, router.loopWait(ctx))
}

// An item's start is not content: a Responses reasoning model announces its
// reasoning item and then thinks in silence, so the first-content limit must
// still apply.
func TestAnItemStartIsNotContent(t *testing.T) {
	ctx := &RequestContext{}
	observeStreamContent(ctx, []llmprotocol.Event{
		{Type: llmprotocol.EventResponseStarted},
		{Type: llmprotocol.EventOutputItemStarted},
		{Type: llmprotocol.EventKeepalive},
		{Type: llmprotocol.EventUsageUpdated},
	})
	assert.False(t, ctx.StreamContentSeen)
	observeStreamContent(ctx, []llmprotocol.Event{{Type: llmprotocol.EventReasoningDelta}})
	assert.True(t, ctx.StreamContentSeen)
}

// A message that falls due with the timer is taken, not lost to the cut.
func TestPollTakesAWaitingMessage(t *testing.T) {
	receiver := &streamReceiver{messages: make(chan receivedMessage, 1)}
	_, ok := receiver.poll()
	assert.False(t, ok)
	receiver.messages <- receivedMessage{}
	_, ok = receiver.poll()
	assert.True(t, ok)
}
