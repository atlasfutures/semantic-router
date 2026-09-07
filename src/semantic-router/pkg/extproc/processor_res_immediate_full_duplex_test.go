package extproc

import (
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A body-phase ImmediateResponse is legal only while Envoy still owns the
// response headers.
//
// Under BUFFERED the headers are held, so a decode failure or any other
// body-phase refusal reaches the client as a real 502 with its JSON body.
// Under FULL_DUPLEX_STREAMED the headers have already gone downstream, and
// sendLocalReply on a started response resets the stream instead: the client
// sees a truncated 200 and is told nothing at all.
//
// The refusal has to travel as the last body there, on a reply that ends the
// response. The status is already spent -- that is what "the headers have
// gone" means -- so what the conversion preserves is the body and the end.

func decodeFailingContext(t *testing.T) *RequestContext {
	t.Helper()
	return &RequestContext{
		Headers:      make(map[string]string),
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID: "request_1", RequestModel: "public-model",
		TraceContext: t.Context(),
	}
}

func TestFullDuplexDecodeFailureEndsTheResponseInsteadOfResettingIt(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := decodeFailingContext(t)
	ctx.FullDuplexResponseBody = true
	stream := NewMockStream(nil)

	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: undecodableUpstreamBody(), EndOfStream: true},
	}, ctx))

	require.Len(t, stream.Responses, 1)
	assert.Nil(t, stream.Responses[0].GetImmediateResponse(),
		"an ImmediateResponse after the headers went downstream resets the stream")
	streamed := stream.Responses[0].GetResponseBody().GetResponse().
		GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed, "the refusal did not travel as a body")
	assert.True(t, streamed.GetEndOfStream(), "the refused response was left open")
	assert.Contains(t, string(streamed.GetBody()), "upstream usage cannot be negative",
		"the client was told nothing about why its response ended")
}

// The same refusal reached through the trailer flush, which is how a body
// Envoy never marks as ended gets there. It must not reset the stream either,
// and the reply that carries it is the one that ends the response.
func TestTrailersFlushConvertsARefusalRatherThanResetting(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := decodeFailingContext(t)
	ctx.FullDuplexResponseBody = true
	stream := NewMockStream(nil)
	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: undecodableUpstreamBody(), EndOfStream: false},
	}, ctx))
	require.Len(t, stream.Responses, 1)
	require.False(t, stream.Responses[0].GetResponseBody().GetResponse().
		GetBodyMutation().GetStreamedResponse().GetEndOfStream())

	sendTrailers(t, router, ctx, stream)

	require.Len(t, stream.Responses, 3)
	assert.Nil(t, stream.Responses[1].GetImmediateResponse(),
		"the flushed refusal reset the stream")
	flushed := stream.Responses[1].GetResponseBody().GetResponse().
		GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, flushed)
	assert.False(t, flushed.GetEndOfStream(),
		"the flushed refusal fabricated an end_of_stream before its trailers")
	assert.Contains(t, string(flushed.GetBody()), "upstream usage cannot be negative")
	assert.NotNil(t, stream.Responses[2].GetResponseTrailers())
}

// Under BUFFERED Envoy still owns the headers, so the refusal is the real 502
// it has always been.
func TestWithoutFullDuplexDecodeFailureStaysAnImmediateResponse(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := decodeFailingContext(t)
	stream := NewMockStream(nil)

	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: undecodableUpstreamBody(), EndOfStream: true},
	}, ctx))

	require.Len(t, stream.Responses, 1)
	immediate := stream.Responses[0].GetImmediateResponse()
	require.NotNil(t, immediate, "the body-phase refusal stopped being an ImmediateResponse")
	assert.Equal(t, 502, int(immediate.GetStatus().GetCode()))
	assert.Contains(t, string(immediate.GetBody()), "upstream usage cannot be negative")
}
