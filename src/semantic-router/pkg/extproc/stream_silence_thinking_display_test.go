package extproc

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// summarizedThinkingChunk is how a Claude 5.x turn asked for thinking:
// {type: adaptive, display: summarized} opens on OpenRouter's Messages API
// (measured 2026-10-28): the message, a thinking block with an empty
// signature, then the summary's first thinking_delta, a few seconds in.
// With the display omitted the same turn sent nothing until its text.
func summarizedThinkingChunk() string {
	frames := [][2]string{
		{"message_start", `{"type":"message_start","message":{"id":"msg_vrtx_1","type":"message","role":"assistant","model":"claude-haiku-5.5","content":[],"stop_reason":null,"usage":{"input_tokens":93,"output_tokens":1}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The user wants the permutations of 1..8 with no i followed by i+1."}}`},
	}
	var chunk strings.Builder
	for _, frame := range frames {
		chunk.WriteString("event: " + frame[0] + "\ndata: " + frame[1] + "\n\n")
	}
	return chunk.String()
}

// D-haiku-thinking-fix: the summary's thinking_delta decodes to
// EventReasoningDelta, which is content for the first-content limit, so a
// thinking Claude turn is not cut as a stall once its summary streams.
func TestASummarizedThinkingDeltaIsFirstContent(t *testing.T) {
	router := silenceRouter(90, 120)
	ctx, transaction := silentStreamContext(t, 100*time.Second)
	ctx.SourceFormat, ctx.TargetFormat = llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1
	ctx.RequestModel = "anthropic/claude-haiku-5.5"
	stream := NewMockStream(nil)

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(summarizedThinkingChunk(), false), ctx))

	require.Len(t, stream.Responses, 1)
	streamed := stream.Responses[0].GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.False(t, streamed.GetEndOfStream(), "a chunk carrying the thinking summary was cut for the silence before it")
	assert.Contains(t, string(streamed.GetBody()), `"thinking_delta"`, "the summary did not reach the Messages client")
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.True(t, ctx.StreamContentSeen)
	assert.WithinDuration(t, time.Now(), ctx.StreamContentSince, 5*time.Second)
	assert.Equal(t, 0, transaction.aborts)
}
