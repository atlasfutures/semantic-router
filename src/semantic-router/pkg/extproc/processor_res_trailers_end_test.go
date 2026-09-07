package extproc

import (
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// HttpBody.end_of_stream is not "the last body chunk". The ext_proc contract
// (v1.36 external_processor.proto) is that it means the last body message AND
// that no trailers will follow. An upstream response that carries trailers
// therefore never sets it on any chunk: the HttpTrailers message is the end of
// the body.
//
// The accumulator flushed only on that flag, so a trailered response left the
// whole body stranded in ctx.ResponseBodyChunks. Every chunk was answered with
// an empty streamed_response, the trailer was answered with a bare
// TrailersResponse, and Envoy's continueIfNecessary completed the exchange:
// the client got an empty 200 and the pipeline behind it -- decode, usage,
// cache, replay -- never ran at all.

func sendTrailers(t *testing.T, router *OpenAIRouter, ctx *RequestContext, stream *MockStream) {
	t.Helper()
	require.NoError(t, router.handleProcessRequest(stream, &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_ResponseTrailers{
			ResponseTrailers: &ext_proc.HttpTrailers{},
		},
	}, ctx))
}

func TestTrailersFlushAResponseBodyEnvoyNeverMarkedAsEnded(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		Headers:      make(map[string]string),
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID: "request_1", RequestModel: "public-model",
		FullDuplexResponseBody: true, TraceContext: t.Context(),
	}
	stream := NewMockStream(nil)
	split := len(openAIChatCompletionFixture) / 2

	// Two chunks, neither marked as the end, because trailers are coming.
	for _, half := range []string{openAIChatCompletionFixture[:split], openAIChatCompletionFixture[split:]} {
		require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(half), EndOfStream: false},
		}, ctx))
	}
	require.Len(t, stream.Responses, 2)

	sendTrailers(t, router, ctx, stream)

	require.Len(t, stream.Responses, 4, "the trailer did not flush the held body")
	flushed := stream.Responses[2].GetResponseBody().GetResponse().
		GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, flushed, "the held body was not flushed as a streamed response")
	assert.Contains(t, string(flushed.GetBody()), "hello",
		"the whole response body was dropped and the client saw an empty 200")
	assert.True(t, flushed.GetEndOfStream(), "the flushed body did not end the response")
	assert.NotNil(t, stream.Responses[3].GetResponseTrailers(),
		"the trailer itself was left unanswered")
}

// The flush happens once. A body Envoy did mark as ended is not sent again
// when a trailer follows it anyway.
func TestTrailersDoNotResendABodyAlreadyEnded(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		Headers:      make(map[string]string),
		SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID: "request_1", RequestModel: "public-model",
		FullDuplexResponseBody: true, TraceContext: t.Context(),
	}
	stream := NewMockStream(nil)
	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(openAIChatCompletionFixture), EndOfStream: true},
	}, ctx))
	require.Len(t, stream.Responses, 1)

	sendTrailers(t, router, ctx, stream)

	require.Len(t, stream.Responses, 2, "an ended response body was sent a second time")
	assert.NotNil(t, stream.Responses[1].GetResponseTrailers())
}

// Outside full duplex the trailer mode is SKIP, so this message does not
// arrive; if it ever did, it must still not invent a body reply.
func TestTrailersFlushNothingOutsideFullDuplex(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string), TraceContext: t.Context()}
	stream := NewMockStream(nil)

	sendTrailers(t, router, ctx, stream)

	require.Len(t, stream.Responses, 1)
	assert.NotNil(t, stream.Responses[0].GetResponseTrailers())
}
