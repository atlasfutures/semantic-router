package extproc

import (
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Nothing else bounds the response accumulator.
//
// Under FULL_DUPLEX_STREAMED Envoy drains each chunk once it has handed it
// over, so no data-plane buffer limit applies and the Router is the only
// holder of the body. failure_mode_allow is forced false for a full-duplex
// stream, so running out of memory is a hard 5xx for every request in flight,
// not just the one that did it. The request-side accumulator has had both
// guards since it was written; these are the same two knobs.

func guardedRouter(maxBytes int64, timeoutSec int) *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{
		RouterOptions: config.RouterOptions{
			MaxResponseBodyBytes:   maxBytes,
			ResponseBodyTimeoutSec: timeoutSec,
		},
	}}
}

func accumulatingContext(t *testing.T) *RequestContext {
	t.Helper()
	return &RequestContext{
		Headers:      make(map[string]string),
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID: "request_1", RequestModel: "public-model",
		FullDuplexResponseBody: true, TraceContext: t.Context(),
	}
}

func TestResponseAccumulatorRefusesABodyOverTheLimit(t *testing.T) {
	router := guardedRouter(32, 0)
	ctx := accumulatingContext(t)
	stream := NewMockStream(nil)

	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(strings.Repeat("a", 64)), EndOfStream: false},
	}, ctx))

	require.Len(t, stream.Responses, 1)
	streamed := stream.Responses[0].GetResponseBody().GetResponse().
		GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed, "the guard did not answer in the shape the mode requires")
	assert.True(t, streamed.GetEndOfStream(),
		"the response was left open after the Router refused to hold more of it")
	assert.Contains(t, string(streamed.GetBody()), "response_body_too_large",
		"the client was not told why its response ended")
	assert.Empty(t, ctx.ResponseBodyChunks, "the refused body is still being held")
}

func TestResponseAccumulatorRefusesABodyThatTakesTooLong(t *testing.T) {
	router := guardedRouter(0, 5)
	ctx := accumulatingContext(t)
	stream := NewMockStream(nil)

	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte("first"), EndOfStream: false},
	}, ctx))
	require.Len(t, stream.Responses, 1)
	require.False(t, stream.Responses[0].GetResponseBody().GetResponse().
		GetBodyMutation().GetStreamedResponse().GetEndOfStream())

	// The upstream is still dribbling out a body six seconds later.
	ctx.ResponseBodyHeldSince = time.Now().Add(-6 * time.Second)
	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte("second"), EndOfStream: false},
	}, ctx))

	require.Len(t, stream.Responses, 2)
	streamed := stream.Responses[1].GetResponseBody().GetResponse().
		GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.True(t, streamed.GetEndOfStream())
	assert.Contains(t, string(streamed.GetBody()), "response_body_timeout")
	assert.Empty(t, ctx.ResponseBodyChunks)
}

// A refused body is over. The trailer that follows must not try to flush what
// the guard already released.
func TestTrailersFlushNothingAfterAGuardRefusal(t *testing.T) {
	router := guardedRouter(32, 0)
	ctx := accumulatingContext(t)
	stream := NewMockStream(nil)
	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(strings.Repeat("a", 64)), EndOfStream: false},
	}, ctx))
	require.Len(t, stream.Responses, 1)

	sendTrailers(t, router, ctx, stream)

	require.Len(t, stream.Responses, 2, "a refused body was sent after its refusal")
	assert.NotNil(t, stream.Responses[1].GetResponseTrailers())
}

// Both knobs default to off, and off means what it did before: no cap, no
// deadline, nothing refused.
func TestResponseAccumulatorIsUnboundedByDefault(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := accumulatingContext(t)
	ctx.ResponseBodyHeldSince = time.Now().Add(-time.Hour)
	stream := NewMockStream(nil)

	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(strings.Repeat("a", 1<<20)), EndOfStream: false},
	}, ctx))

	require.Len(t, stream.Responses, 1)
	streamed := stream.Responses[0].GetResponseBody().GetResponse().
		GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.False(t, streamed.GetEndOfStream(), "an unbounded accumulator refused a body")
	assert.Len(t, ctx.ResponseBodyChunks, 1<<20)
}

// The request accumulator's limits are not the response accumulator's.
//
// The shipped config/config.yaml sets streamed_body max_bytes 1048576 and
// timeout_sec 15 for the request. A model response is routinely larger than a
// megabyte and routinely takes longer than fifteen seconds, so reading those
// two numbers on the response side would refuse ordinary turns the moment the
// cell declared full duplex.
func TestRequestBodyLimitsDoNotBoundTheResponse(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		RouterOptions: config.RouterOptions{
			StreamedBodyMode:       true,
			MaxStreamedBodyBytes:   1048576,
			StreamedBodyTimeoutSec: 15,
		},
	}}
	ctx := accumulatingContext(t)
	ctx.ResponseBodyHeldSince = time.Now().Add(-time.Hour)
	stream := NewMockStream(nil)

	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{
			Body: []byte(strings.Repeat("a", 2*1048576)), EndOfStream: false,
		},
	}, ctx))

	require.Len(t, stream.Responses, 1)
	streamed := stream.Responses[0].GetResponseBody().GetResponse().
		GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.False(t, streamed.GetEndOfStream(),
		"the request body limits refused an ordinary response")
	assert.Len(t, ctx.ResponseBodyChunks, 2*1048576)
}
