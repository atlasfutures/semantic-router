package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
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
// read, or nil while the body is still arriving.
func completeResponseBody(
	v *ext_proc.ProcessingRequest_ResponseBody,
	ctx *RequestContext,
) *ext_proc.ProcessingRequest_ResponseBody {
	if ctx == nil || !ctx.FullDuplexResponseBody || ctx.IsStreamingResponse {
		return v
	}
	body := v.ResponseBody
	if body.GetEndOfStream() && len(ctx.ResponseBodyChunks) == 0 {
		// One chunk carried the whole body. Nothing to join, and no copy.
		ctx.ResponseBodyEnded = true
		return v
	}
	ctx.ResponseBodyChunks = append(ctx.ResponseBodyChunks, body.GetBody()...)
	if !body.GetEndOfStream() {
		return nil
	}
	ctx.ResponseBodyEnded = true
	return &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{
			Body:        ctx.ResponseBodyChunks,
			EndOfStream: true,
		},
	}
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
// they do, this is where the held body travels, and the reply that carries it
// is the one that ends the response. Nothing is returned when the body already
// ended on a chunk, when nothing is held, or outside full duplex, where the
// trailer mode is SKIP and this message does not arrive at all.
func (r *OpenAIRouter) endResponseBodyAtTrailers(
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if ctx == nil || !ctx.FullDuplexResponseBody || ctx.IsStreamingResponse {
		return nil, nil
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
	return normalizeFullDuplexResponseBody(response, ctx, joined), nil
}
