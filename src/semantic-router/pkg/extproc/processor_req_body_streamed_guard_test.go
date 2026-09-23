package extproc

import (
	"bytes"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// feedOversizedAccumulation pushes one byte past the handler's byte limit, the
// first of the two guards documented at processor_req_body_streamed.go:26-30.
func feedOversizedAccumulation(t *testing.T) (*ext_proc.ProcessingResponse, error) {
	t.Helper()
	ctx := &RequestContext{}
	handler := newStreamedBodyHandler(makeTestRouterWithLimits(100, 0), ctx)
	t.Cleanup(handler.Release)
	return handler.HandleChunk(&ext_proc.HttpBody{Body: bytes.Repeat([]byte("a"), 101)}, ctx)
}

// feedExpiredAccumulation delivers a chunk after the accumulation deadline has
// already passed, the second guard.
func feedExpiredAccumulation(t *testing.T) (*ext_proc.ProcessingResponse, error) {
	t.Helper()
	ctx := &RequestContext{}
	handler := newStreamedBodyHandler(makeTestRouterWithLimits(0, 1), ctx)
	t.Cleanup(handler.Release)
	handler.deadline = time.Now().Add(-time.Second)
	return handler.HandleChunk(&ext_proc.HttpBody{Body: []byte("opaque")}, ctx)
}

// TestStreamedBodyGuardAnswersOversizedAccumulationWith413 is the ask.
//
// The handler documents "rejects requests whose accumulated body exceeds the
// limit (HTTP 413)". checkGuards returns a plain Go error instead, and
// HandleChunk returns it as (nil, err), so no 413 is ever built. The error
// closes the ExtProc gRPC stream, and every shipped Envoy configuration sets
// failure_mode_allow: true, so the oversized body reaches the backend with no
// request processing at all rather than being refused.
func TestStreamedBodyGuardAnswersOversizedAccumulationWith413(t *testing.T) {
	response, err := feedOversizedAccumulation(t)
	require.NoError(t, err)
	require.NotNil(t, response.GetImmediateResponse())
	assert.Equal(t, typev3.StatusCode_PayloadTooLarge,
		response.GetImmediateResponse().GetStatus().GetCode())
}

// TestStreamedBodyGuardAnswersExpiredAccumulationWith408 is the same ask for
// the deadline guard, documented as "(HTTP 408)" on the same comment block.
func TestStreamedBodyGuardAnswersExpiredAccumulationWith408(t *testing.T) {
	response, err := feedExpiredAccumulation(t)
	require.NoError(t, err)
	require.NotNil(t, response.GetImmediateResponse())
	assert.Equal(t, typev3.StatusCode_RequestTimeout,
		response.GetImmediateResponse().GetStatus().GetCode())
}

// TestStreamedBodyGuardClosesTheStreamToday pins the present behaviour so a
// change to it shows up in the diff rather than only in the two tests above.
func TestStreamedBodyGuardClosesTheStreamToday(t *testing.T) {
	response, err := feedOversizedAccumulation(t)
	require.Error(t, err)
	assert.Nil(t, response)

	response, err = feedExpiredAccumulation(t)
	require.Error(t, err)
	assert.Nil(t, response)
}
