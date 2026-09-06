package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Trailers are a phase, not an unknown message.
//
// response_trailer_mode: SEND is required beside response_body_mode:
// FULL_DUPLEX_STREAMED, so turning full duplex on at the cell makes Envoy
// send an HttpTrailers message the Router has never seen. The ext_proc
// contract is that the server answers the message it was sent: a
// TrailersResponse for HttpTrailers. Answering with a RequestBody CONTINUE
// replies in the wrong phase, and Envoy closes the stream on it.
func TestResponseTrailersAreAnsweredInTheirOwnPhase(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string)}
	stream := NewMockStream(nil)

	req := &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_ResponseTrailers{
			ResponseTrailers: &ext_proc.HttpTrailers{
				Trailers: &core.HeaderMap{Headers: []*core.HeaderValue{
					{Key: "grpc-status", RawValue: []byte("0")},
				}},
			},
		},
	}

	require.NoError(t, router.handleProcessRequest(stream, req, ctx))
	require.Len(t, stream.Responses, 1)
	assert.NotNil(t, stream.Responses[0].GetResponseTrailers(),
		"a response trailer was answered outside the response-trailer phase")
	assert.Nil(t, stream.Responses[0].GetRequestBody(),
		"the trailer fell through to the unknown-request body reply")
}

func TestRequestTrailersAreAnsweredInTheirOwnPhase(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string)}
	stream := NewMockStream(nil)

	req := &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_RequestTrailers{
			RequestTrailers: &ext_proc.HttpTrailers{},
		},
	}

	require.NoError(t, router.handleProcessRequest(stream, req, ctx))
	require.Len(t, stream.Responses, 1)
	assert.NotNil(t, stream.Responses[0].GetRequestTrailers(),
		"a request trailer was answered outside the request-trailer phase")
}

// The Router adds nothing to a trailer. It says so explicitly rather than by
// omission, so a later reader can tell the empty mutation from a forgotten one.
func TestTrailerReplyMutatesNothing(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string)}
	stream := NewMockStream(nil)

	req := &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_ResponseTrailers{
			ResponseTrailers: &ext_proc.HttpTrailers{},
		},
	}

	require.NoError(t, router.handleProcessRequest(stream, req, ctx))
	require.Len(t, stream.Responses, 1)
	assert.Nil(t, stream.Responses[0].GetResponseTrailers().GetHeaderMutation())
}
