package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Joining a response body that arrives in pieces.
//
// Under FULL_DUPLEX_STREAMED Envoy delivers the response body as a series of
// HttpBody messages and marks the last one. Nothing bounds how it splits them.
// The Router's non-streaming pipeline decodes what it is handed as a complete
// wire response, so a body split in two answered the turn with a decode
// failure on the first half.
//
// It also read the end of the body from the response Content-Type: only an SSE
// turn had an end, because only the streaming path was passed
// HttpBody.end_of_stream. A non-SSE response therefore never ended, whatever
// Envoy said on the chunk.
//
// Both are the same fix. Chunks are joined here until Envoy marks the end, and
// the pipeline downstream sees exactly what it saw under BUFFERED: one whole
// body, marked as the end. That is also why buffering a whole non-SSE response
// costs nothing new -- BUFFERED is what the cell runs today.
//
// A streamed turn is exempt. Holding its deltas to the end of the turn would
// be the opposite of streaming, and the semantic streaming path already
// consumes chunks incrementally and ends the response itself.

// completeResponseBody returns the response-body message the pipeline should
// read, or nil while the body is still arriving. A non-nil error is a breached
// guard, and the turn is over.
func (r *OpenAIRouter) completeResponseBody(
	v *ext_proc.ProcessingRequest_ResponseBody,
	ctx *RequestContext,
) (*ext_proc.ProcessingRequest_ResponseBody, *llmprotocol.ProtocolError) {
	if ctx == nil || !ctx.FullDuplexResponseBody || ctx.IsStreamingResponse {
		return v, nil
	}
	body := v.ResponseBody
	if body.GetEndOfStream() && len(ctx.ResponseBodyChunks) == 0 {
		// One chunk carried the whole body. Nothing to join, and no copy.
		ctx.ResponseBodyEnded = true
		return v, nil
	}
	if ctx.ResponseBodyHeldSince.IsZero() {
		ctx.ResponseBodyHeldSince = time.Now()
	}
	ctx.ResponseBodyChunks = append(ctx.ResponseBodyChunks, body.GetBody()...)
	if breach := r.responseBodyGuardBreach(ctx); breach != nil {
		// The turn is over, so what is held is released here rather than at
		// the trailer that will not be flushing it.
		ctx.ResponseBodyChunks = nil
		ctx.ResponseBodyEnded = true
		return nil, breach
	}
	if !body.GetEndOfStream() {
		return nil, nil
	}
	ctx.ResponseBodyEnded = true
	return &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{
			Body:        ctx.ResponseBodyChunks,
			EndOfStream: true,
		},
	}, nil
}

// What bounds the accumulator.
//
// Under FULL_DUPLEX_STREAMED Envoy drains each chunk once it has handed it
// over, so no data-plane buffer limit applies and the Router is the only
// holder of the body. failure_mode_allow is forced false for a full-duplex
// stream, so exhausting memory here is a hard 5xx for every request in flight
// rather than only the one that did it.
//
// The two knobs are the response stream's own, not the request accumulator's.
// The shipped config sets streamed_body max_bytes to 1 MiB and timeout_sec to
// 15 for the request, and a model response is routinely larger and slower than
// both, so reading those here would refuse ordinary turns the moment the cell
// declared full duplex. Both default to off, and off means unbounded, which is
// what this did before there was a guard.
func (r *OpenAIRouter) responseBodyGuardBreach(ctx *RequestContext) *llmprotocol.ProtocolError {
	if r == nil || r.Config == nil {
		return nil
	}
	if maxBytes := r.Config.MaxResponseBodyBytes; maxBytes > 0 &&
		int64(len(ctx.ResponseBodyChunks)) > maxBytes {
		logging.ComponentWarnEvent("extproc", "response_body_too_large", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      ctx.RequestModel,
			"held_bytes": len(ctx.ResponseBodyChunks),
			"max_bytes":  maxBytes,
		})
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"response_body_too_large",
			"the model service sent more response body than the router will hold",
			nil,
		)
	}
	timeout := time.Duration(r.Config.ResponseBodyTimeoutSec) * time.Second
	if timeout > 0 && time.Since(ctx.ResponseBodyHeldSince) > timeout {
		logging.ComponentWarnEvent("extproc", "response_body_timeout", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      ctx.RequestModel,
			"held_bytes": len(ctx.ResponseBodyChunks),
			"timeout_ms": timeout.Milliseconds(),
		})
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamTimeout,
			"response_body_timeout",
			"the model service took too long to finish its response body",
			nil,
		)
	}
	return nil
}

// responseBodyGuardResponse ends the response at a breached guard.
//
// The response headers have already gone downstream, so an ImmediateResponse
// would reset the stream and the client would be told nothing. The refusal
// travels as the last body instead, encoded in the client's own wire format,
// on a reply that ends the response.
func (r *OpenAIRouter) responseBodyGuardResponse(
	ctx *RequestContext,
	protocolError *llmprotocol.ProtocolError,
) *ext_proc.ProcessingResponse {
	var encoded []byte
	engine, err := r.protocolEngine()
	if err == nil {
		_, target := responseWireFormats(ctx)
		encoded, err = engine.EncodeError(target, protocolError)
	}
	if err != nil {
		// Ending the response still matters more than saying why: a client
		// told nothing waits for the platform cut.
		encoded = nil
		logging.ComponentErrorEvent("extproc", "response_body_guard_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      err.Error(),
		})
	}
	metrics.RecordRequestError(ctx.RequestModel, string(protocolError.Category))
	return buildResponseBodyContinueResponse(responseStreamBodyMutation(ctx, encoded, true), nil)
}

// heldResponseBodyChunk is the reply for a chunk the Router kept. It forwards
// no bytes and ends nothing; the whole body travels on the reply to the chunk
// Envoy marked as the end.
func heldResponseBodyChunk() *ext_proc.ProcessingResponse {
	return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_StreamedResponse{
			StreamedResponse: &ext_proc.StreamedBodyResponse{},
		},
	}, nil)
}

// endResponseBodyAtTrailers is the other way a full-duplex response body ends.
//
// Envoy marks a body chunk as the end only when no trailers will follow. When
// they do, this is where the response body ends, and the reply built here is
// the one that ends it. Nothing is returned when the body already ended on a
// chunk, when nothing is held, or outside full duplex, where the trailer mode
// is SKIP and this message does not arrive at all.
//
// A streamed turn ends here too, and for the same reason. The semantic
// streaming path finalizes on end_of_stream, so trailers left it unfinalized:
// no usage, no cache or replay write, and no end on the response. The turn was
// finalized much later by handleProcessReceiveError, which marks it aborted --
// a turn the provider completed, recorded as one that failed.
func (r *OpenAIRouter) endResponseBodyAtTrailers(
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if ctx == nil || !ctx.FullDuplexResponseBody {
		return nil, nil
	}
	if ctx.IsStreamingResponse {
		if ctx.StreamingComplete {
			return nil, nil
		}
		// The streaming path builds its own end. Telling it the stream is over
		// is the same thing Envoy's flag would have told it.
		return endedByTrailers(r.handleSemanticStreamingResponseBody(nil, true, ctx)), nil
	}
	if ctx.ResponseBodyEnded || len(ctx.ResponseBodyChunks) == 0 {
		return nil, nil
	}
	joined := &ext_proc.HttpBody{Body: ctx.ResponseBodyChunks, EndOfStream: true}
	ctx.ResponseBodyEnded = true
	response, err := r.handleResponseBody(
		&ext_proc.ProcessingRequest_ResponseBody{ResponseBody: joined}, ctx)
	if err != nil {
		return nil, err
	}
	return endedByTrailers(normalizeFullDuplexResponseBody(response, ctx, joined)), nil
}

// endedByTrailers takes the end off a body reply that the trailers will end.
//
// StreamedBodyResponse.end_of_stream is documented as the server's echo of a
// body request: set it "if it has received a body request with end_of_stream
// set to true, and this is the last chunk of body responses". When trailers
// follow, that flag is exactly what Envoy withholds on every body message, so
// a body reply claiming the end here claims something no request carried, and
// ending the body stream before the trailers would take them with it. Envoy
// completes the exchange on the TrailersResponse instead.
//
// This is the opposite of the deadline cut, which sets the same flag on
// purpose because there the Router is ending a response the upstream has not
// finished, and nothing else in the protocol can say so.
func endedByTrailers(response *ext_proc.ProcessingResponse) *ext_proc.ProcessingResponse {
	streamed := response.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	if streamed != nil {
		streamed.EndOfStream = false
	}
	return response
}
