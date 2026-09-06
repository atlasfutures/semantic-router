package extproc

import (
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Under FULL_DUPLEX_STREAMED every response-body reply has to carry a
// streamed_response. Envoy's processor_state.cc refuses a plain body mutation
// in that mode, and a CommonResponse carrying no mutation at all leaves the
// chunk unaccounted -- there is no pass-through in full duplex, because the
// reply *is* the response body rather than a mutation of it.
//
// Four response-body paths sent one of those two illegal shapes. Each is
// exercised here through processResponseBody, which is the seam the reply
// actually leaves by.

func fullDuplexResponseChunk(
	t *testing.T,
	router *OpenAIRouter,
	ctx *RequestContext,
	body []byte,
	endOfStream bool,
) *ext_proc.ProcessingResponse {
	t.Helper()
	stream := NewMockStream(nil)
	v := &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: body, EndOfStream: endOfStream},
	}
	require.NoError(t, router.processResponseBody(stream, v, ctx))
	require.Len(t, stream.Responses, 1)
	return stream.Responses[0]
}

func requireStreamedReply(t *testing.T, response *ext_proc.ProcessingResponse) *ext_proc.StreamedBodyResponse {
	t.Helper()
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, mutation, "a full-duplex reply carried no body mutation at all")
	streamed := mutation.GetStreamedResponse()
	require.NotNil(t, streamed,
		"a full-duplex reply carried a plain body mutation, which Envoy refuses in this mode")
	return streamed
}

// req_filter_skip_processing.go: a skipped turn replied CONTINUE with no
// mutation, which under full duplex drops the chunk instead of forwarding it.
func TestFullDuplexSkipProcessingForwardsTheChunk(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SkipProcessing: true, FullDuplexResponseBody: true, TraceContext: t.Context(),
	}
	chunk := []byte(`{"passed":"through"}`)

	streamed := requireStreamedReply(t, fullDuplexResponseChunk(t, router, ctx, chunk, true))
	assert.Equal(t, chunk, streamed.GetBody(), "the skipped turn's bytes never reached the client")
	assert.True(t, streamed.GetEndOfStream())
}

// processor_res_body.go: the looper captures the body for replay and replies
// CONTINUE with no mutation.
func TestFullDuplexLooperForwardsTheChunk(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		LooperRequest: true, FullDuplexResponseBody: true, TraceContext: t.Context(),
	}
	chunk := []byte(`{"looper":"body"}`)

	streamed := requireStreamedReply(t, fullDuplexResponseChunk(t, router, ctx, chunk, true))
	assert.Equal(t, chunk, streamed.GetBody())
	assert.True(t, streamed.GetEndOfStream())
}

// processor_res_transport_error.go: the translated error body went out as a
// plain body mutation.
func TestFullDuplexTransportErrorUsesStreamedReply(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		UpstreamStatusCode: 503, RequestID: "request_1", RequestModel: "public-model",
		FullDuplexResponseBody: true, TraceContext: t.Context(),
	}

	streamed := requireStreamedReply(t, fullDuplexResponseChunk(t, router, ctx,
		[]byte(`{"error":{"message":"upstream unavailable","type":"server_error"}}`), true))
	assert.NotEmpty(t, streamed.GetBody(), "the translated error never reached the client")
	assert.True(t, streamed.GetEndOfStream())
}

// processor_res_body_pipeline.go: setResponseBodyMutation always built a plain
// body mutation, so every rewritten non-streaming response was illegal here.
func TestFullDuplexRewrittenResponseUsesStreamedReply(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID: "request_1", RequestModel: "public-model",
		FullDuplexResponseBody: true, TraceContext: t.Context(),
	}

	streamed := requireStreamedReply(t, fullDuplexResponseChunk(t, router, ctx,
		[]byte(openAIChatCompletionFixture), true))
	assert.Contains(t, string(streamed.GetBody()), "message",
		"the rewritten response body never reached the client")
	assert.True(t, streamed.GetEndOfStream())
}

// The mode is Envoy's to declare. With STREAMED or BUFFERED the Router must
// send exactly what it sent before -- a plain body mutation where it rewrote,
// and a bare CONTINUE where it did not.
func TestWithoutFullDuplexTheReplyShapeIsUnchanged(t *testing.T) {
	router := &OpenAIRouter{}

	skipped := fullDuplexResponseChunk(t, router,
		&RequestContext{SkipProcessing: true, TraceContext: t.Context()},
		[]byte(`{"passed":"through"}`), true)
	assert.Nil(t, skipped.GetResponseBody().GetResponse().GetBodyMutation(),
		"a skipped turn gained a mutation outside full duplex")

	rewritten := fullDuplexResponseChunk(t, router, &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID: "request_1", RequestModel: "public-model", TraceContext: t.Context(),
	}, []byte(openAIChatCompletionFixture), true)
	mutation := rewritten.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, mutation)
	assert.NotEmpty(t, mutation.GetBody(), "the rewritten body stopped being a plain body mutation")
	assert.Nil(t, mutation.GetStreamedResponse(),
		"a streamed_response was sent in a mode that refuses it")
}

const openAIChatCompletionFixture = `{"id":"chatcmpl-1","object":"chat.completion",` +
	`"created":1788474769,"model":"public-model","choices":[{"index":0,` +
	`"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`
