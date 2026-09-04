package extproc

import (
	"bytes"
	"testing"
	"time"

	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Writing the closing frames is not the same as ending the response.
//
// Build 16 measured the difference on the dev cell 2026-09-04. The deadline
// fired at 590 s and the client received content_block_stop and the typed
// error frame, and then the connection stayed open: curl saw 632.8 s of wall
// clock and Envoy logged duration 629,997 with flags DC. The Router had
// stopped emitting bytes and gone on swallowing upstream chunks, so the HTTP
// response was starved rather than finished, and the platform ended it 40
// seconds later.
//
// Envoy's ext_proc has exactly one way for the server to end a response body:
// StreamedBodyResponse.end_of_stream, which is only accepted when the response
// body mode is FULL_DUPLEX_STREAMED (external_processor.proto StreamedBodyResponse;
// Envoy processor_state.cc rejects a streamed_response in any other mode, and
// rejects a plain body mutation in this one). ImmediateResponse is not an
// alternative: after response headers have gone downstream Envoy resets the
// stream and drops the body, which would take the closing frames with it.
func TestRouterEndsTheResponseAtItsDeadline(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.FullDuplexResponseBody = true

	streamed := streamDeadlineStreamedResponse(t, router, ctx, endlessUpstreamChunk(1))

	require.NotNil(t, streamed, "the deadline reply did not carry a streamed body")
	if !bytes.Contains(streamed.GetBody(), []byte("content_block_stop")) ||
		!bytes.Contains(streamed.GetBody(), []byte(`"type":"error"`)) {
		t.Fatalf("the end-of-stream marker travelled without the closing frames:\n%s", streamed.GetBody())
	}
	assert.True(t, streamed.GetEndOfStream(),
		"the response was left open after the Router ended the turn")
}

// The marker travels once. After it, the response is over, and the chunks the
// upstream keeps sending carry neither bytes nor a second end.
func TestNothingFollowsTheEndedResponse(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.FullDuplexResponseBody = true
	streamDeadlineStreamedResponse(t, router, ctx, endlessUpstreamChunk(1))

	trailing := streamDeadlineStreamedResponse(t, router, ctx, endlessUpstreamChunk(2))
	require.NotNil(t, trailing, "a full-duplex response may not fall back to a plain body mutation")
	assert.Empty(t, trailing.GetBody(), "the upstream kept reaching a client whose response ended")
	assert.False(t, trailing.GetEndOfStream(), "the response was ended twice")
}

// A turn that finishes on its own has to be ended too. In this mode Envoy
// stops forwarding the response body until the server says the last chunk is
// the last one, so a normal stream that never sets the marker would hang
// exactly like the overrunning one.
func TestCompletedResponseIsEndedToo(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.FullDuplexResponseBody = true
	ctx.StartTime = time.Now()

	response := router.handleSemanticStreamingResponseBody(
		[]byte(endlessUpstreamChunk(1)), true, ctx,
	)
	streamed := response.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed)
	assert.True(t, streamed.GetEndOfStream(), "a stream that reached its end was left open")
}

// Envoy has to be configured for it. Where the response body mode is the
// plain STREAMED one, no reply can end the response, and the Router says so
// rather than reporting a turn it ended honestly: the platform will still cut
// this connection, tens of seconds later.
func TestResponseThatCannotBeEndedSaysSo(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)

	response := router.handleSemanticStreamingResponseBody(
		[]byte(endlessUpstreamChunk(1)), false, ctx,
	)
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	assert.Nil(t, mutation.GetStreamedResponse(),
		"a streamed_response outside FULL_DUPLEX_STREAMED is rejected by Envoy")
	assert.NotEmpty(t, mutation.GetBody(), "the closing frames did not travel")

	fields := findLogEvent(t, logs, "response_stream_deadline_exceeded")
	ended, stated := fields["response_ended"].(bool)
	require.True(t, stated, "the truncation line does not say whether the response ended: %v", fields)
	assert.False(t, ended, "the Router claimed to have ended a response it cannot end: %v", fields)
}

// The mode is Envoy's to declare, on the same protocol configuration the
// request side already reads.
func TestEnvoyDeclaresTheResponseBodyMode(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	request := &ext_proc.ProcessingRequest{
		ProtocolConfig: &ext_proc.ProtocolConfiguration{
			ResponseBodyMode: http_ext.ProcessingMode_FULL_DUPLEX_STREAMED,
		},
		Request: &ext_proc.ProcessingRequest_RequestBody{
			RequestBody: &ext_proc.HttpBody{Body: []byte(`{"mod`), EndOfStream: false},
		},
	}

	require.NoError(t, router.handleProcessRequest(NewMockStream(nil), request, ctx))
	assert.True(t, ctx.FullDuplexResponseBody)
}

func streamDeadlineStreamedResponse(
	t *testing.T,
	router *OpenAIRouter,
	ctx *RequestContext,
	chunk string,
) *ext_proc.StreamedBodyResponse {
	t.Helper()
	response := router.handleSemanticStreamingResponseBody([]byte(chunk), false, ctx)
	return response.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
}
