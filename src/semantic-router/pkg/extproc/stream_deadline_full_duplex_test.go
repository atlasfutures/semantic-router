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

// The whole point of CP9x, driven through the seam Envoy actually uses.
//
// The definition of done is a client-side measurement: a forced deadline cut
// ends at 590 s rather than 630 s. That needs a deployed cell, and it is still
// owed. What is proved here is the mechanism the measurement would measure --
// that with the cell declaring full duplex, a turn cut at the Router's own
// deadline leaves the ext_proc exchange in a state where the response is over,
// rather than starved and waiting on the platform.
//
// Three things have to hold at once, and each was a separate gap:
//
//   - the deadline reply ends the response, carrying end_of_stream with the
//     closing frames on it;
//   - it is a streamed_response, the only shape Envoy accepts in this mode;
//   - the trailers Envoy sends next are answered in their own phase, so the
//     exchange completes instead of being closed as a protocol violation.
//
// The ladder is the reason the numbers differ. The Router's deadline is 590 s;
// Cloud Run's request timeout is 630 s. Before this, the Router wrote the
// closing frames at 590 s and then went on swallowing upstream chunks, so the
// response was starved rather than finished and the platform ended it 40 s
// later. Measured on the dev cell 2026-09-04: curl saw 632.8 s.
func TestDeadlineCutEndsTheExchangeUnderFullDuplex(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string)}
	stream := NewMockStream(nil)

	// The Router's deadline has passed and the platform's has not. That window
	// is the whole subject.
	elapsed := 600 * time.Second
	require.Less(t, router.responseStreamDeadline(), elapsed,
		"the turn has not reached the Router's deadline")
	require.Less(t, elapsed, 630*time.Second,
		"the turn has already reached the platform cut, so nothing is being proved")
	seedOverrunningStreamContext(t, ctx, elapsed)

	// Envoy declares the mode on every message; the Router never chooses it.
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(
		endlessUpstreamChunk(1), false), ctx))
	require.True(t, ctx.FullDuplexResponseBody, "the cell's full-duplex mode was not detected")

	require.Len(t, stream.Responses, 1)
	streamed := stream.Responses[0].GetResponseBody().GetResponse().
		GetBodyMutation().GetStreamedResponse()
	require.NotNil(t, streamed, "the deadline reply was not a shape Envoy accepts in this mode")
	assert.True(t, streamed.GetEndOfStream(),
		"the response was left open, so the client waits for the platform cut")
	if !bytes.Contains(streamed.GetBody(), []byte("content_block_stop")) ||
		!bytes.Contains(streamed.GetBody(), []byte(`"type":"error"`)) {
		t.Fatalf("the end-of-stream marker travelled without the closing frames:\n%s", streamed.GetBody())
	}

	// response_trailer_mode: SEND comes with the mode, so this message follows.
	require.NoError(t, router.handleProcessRequest(stream, &ext_proc.ProcessingRequest{
		ProtocolConfig: fullDuplexProtocolConfig(),
		Request: &ext_proc.ProcessingRequest_ResponseTrailers{
			ResponseTrailers: &ext_proc.HttpTrailers{},
		},
	}, ctx))
	require.Len(t, stream.Responses, 2)
	assert.NotNil(t, stream.Responses[1].GetResponseTrailers(),
		"the trailer after the ended response was answered in the wrong phase")
}

// Build 23 runs BUFFERED. The same cut there must behave exactly as it did:
// the closing frames go out as a plain body mutation and the response is still
// the platform's to end.
func TestDeadlineCutIsUnchangedWithoutFullDuplex(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string)}
	stream := NewMockStream(nil)
	seedOverrunningStreamContext(t, ctx, 600*time.Second)

	require.NoError(t, router.handleProcessRequest(stream, &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(endlessUpstreamChunk(1))},
		},
	}, ctx))
	require.False(t, ctx.FullDuplexResponseBody)

	require.Len(t, stream.Responses, 1)
	mutation := stream.Responses[0].GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, mutation)
	assert.Nil(t, mutation.GetStreamedResponse(),
		"a streamed_response was sent in a mode that refuses it")
	assert.Contains(t, string(mutation.GetBody()), "content_block_stop",
		"the closing frames stopped travelling in the mode build 23 runs")
}

func seedOverrunningStreamContext(t *testing.T, ctx *RequestContext, elapsed time.Duration) {
	t.Helper()
	overrunning := overrunningStreamContext(t)
	ctx.RequestID = overrunning.RequestID
	ctx.RequestModel = overrunning.RequestModel
	ctx.SourceFormat = overrunning.SourceFormat
	ctx.TargetFormat = overrunning.TargetFormat
	ctx.IsStreamingResponse = true
	ctx.TraceContext = overrunning.TraceContext
	ctx.StartTime = time.Now().Add(-elapsed)
}

func fullDuplexProtocolConfig() *ext_proc.ProtocolConfiguration {
	return &ext_proc.ProtocolConfiguration{
		ResponseBodyMode: http_ext.ProcessingMode_FULL_DUPLEX_STREAMED,
	}
}

func fullDuplexResponseBodyRequest(chunk string, endOfStream bool) *ext_proc.ProcessingRequest {
	return &ext_proc.ProcessingRequest{
		ProtocolConfig: fullDuplexProtocolConfig(),
		Request: &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(chunk), EndOfStream: endOfStream},
		},
	}
}
