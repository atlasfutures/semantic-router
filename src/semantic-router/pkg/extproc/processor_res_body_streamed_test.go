package extproc

import (
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Under FULL_DUPLEX_STREAMED a non-SSE response body arrives in as many chunks
// as Envoy chooses to send. The Router decoded each one as though it were the
// whole wire response, so the first partial chunk failed to decode and the
// turn was answered with a decode failure. It also read the end of the body
// from the response Content-Type rather than from the chunk, so a non-SSE
// response had no end at all.

func TestFullDuplexJoinsAResponseSplitAcrossChunks(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID: "request_1", RequestModel: "public-model",
		FullDuplexResponseBody: true, TraceContext: t.Context(),
	}
	split := len(openAIChatCompletionFixture) / 2

	held := fullDuplexResponseChunk(t, router, ctx, []byte(openAIChatCompletionFixture[:split]), false)
	heldStreamed := requireStreamedReply(t, held)
	assert.Empty(t, heldStreamed.GetBody(), "half a response document was forwarded to the client")
	assert.False(t, heldStreamed.GetEndOfStream(), "the response ended on a partial body")

	final := requireStreamedReply(t,
		fullDuplexResponseChunk(t, router, ctx, []byte(openAIChatCompletionFixture[split:]), true))
	assert.Contains(t, string(final.GetBody()), "hello",
		"the joined response never reached the client")
	assert.NotContains(t, string(final.GetBody()), "invalid",
		"a chunk was decoded as a whole response and failed")
	assert.True(t, final.GetEndOfStream(), "the joined response was left open")
}

// The end of a non-SSE body is what Envoy said on the chunk. Nothing about the
// Content-Type can say it.
func TestFullDuplexEndsANonStreamingResponseFromTheChunk(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID: "request_1", RequestModel: "public-model",
		IsStreamingResponse:    false,
		FullDuplexResponseBody: true, TraceContext: t.Context(),
	}

	streamed := requireStreamedReply(t, fullDuplexResponseChunk(t, router, ctx,
		[]byte(openAIChatCompletionFixture), true))
	assert.True(t, streamed.GetEndOfStream(),
		"a non-SSE response was left for the platform to end")
}

// An SSE turn is incremental on purpose. Accumulating it would hold every
// delta back to the end of the turn, which is the opposite of streaming.
func TestFullDuplexDoesNotAccumulateAnSSEResponse(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{FullDuplexResponseBody: true, IsStreamingResponse: true}
	chunk := &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte("data: {}\n\n"), EndOfStream: false},
	}

	complete, breach := router.completeResponseBody(chunk, ctx)
	require.Nil(t, breach)
	require.Same(t, chunk, complete, "a streamed turn was buffered instead of forwarded")
}

// Outside full duplex Envoy delivers a whole body and the Router must not
// start buffering.
func TestWithoutFullDuplexNoChunkIsHeld(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{}
	chunk := &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(`{"a":1}`), EndOfStream: false},
	}

	complete, breach := router.completeResponseBody(chunk, ctx)
	require.Nil(t, breach)
	require.Same(t, chunk, complete)
}
