package extproc

import (
	"bytes"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The one shape a response-body reply may take under FULL_DUPLEX_STREAMED.
//
// Envoy's processor_state.cc refuses a plain body mutation in that mode and
// refuses a streamed_response in every other one, so the two are not
// interchangeable and the Router has to send whichever the data plane
// declared. A CommonResponse carrying no mutation is worse than either: in
// full duplex the reply *is* the response body rather than a mutation of it,
// so a reply that mutates nothing forwards nothing, and the chunk is dropped.
//
// Four response-body paths sent one of those illegal shapes:
//
//   - req_filter_skip_processing.go, a skipped turn: bare CONTINUE
//   - processor_res_body.go, the looper capture: bare CONTINUE
//   - processor_res_transport_error.go, a translated upstream failure: plain
//     body mutation, via setResponseBodyMutation
//   - processor_res_body_pipeline.go, every rewritten non-streaming response:
//     plain body mutation, via setResponseBodyMutation
//
// Rather than teach each of them the mode, the shape is fixed once where the
// reply leaves: processResponseBody. That keeps the mode out of the paths that
// have nothing to do with it, and it holds for any reply added later.
//
// The semantic streaming path is left alone. It already builds its own
// streamed_response through responseStreamBodyMutation, and it is the only
// path that knows whether the turn it is ending is over -- an end-of-stream
// invented here would end a response mid-turn.
//
// A body-phase ImmediateResponse is the fifth shape, and it is legal only
// while Envoy still owns the response headers. Under BUFFERED it does: a
// decode failure, a jailbreak refusal, a failed context recovery and the rest
// reach the client as the 502 or 403 they say they are. Under full duplex the
// headers have already gone downstream, where sendLocalReply on a started
// response resets the stream -- so the refusal would become a truncated 200
// carrying nothing.
func normalizeFullDuplexResponseBody(
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
	chunk *ext_proc.HttpBody,
) *ext_proc.ProcessingResponse {
	if ctx == nil || !ctx.FullDuplexResponseBody || response == nil {
		return response
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		return endResponseWithImmediateBody(ctx, immediate)
	}
	bodyResponse := response.GetResponseBody()
	if bodyResponse == nil {
		// A reply in another phase, which this does not shape.
		return response
	}
	if bodyResponse.Response == nil {
		bodyResponse.Response = &ext_proc.CommonResponse{Status: ext_proc.CommonResponse_CONTINUE}
	}
	common := bodyResponse.Response
	if common.GetBodyMutation().GetStreamedResponse() != nil {
		return response
	}

	// Nothing was rewritten, so what travels is what arrived. Outside full
	// duplex Envoy would have forwarded it without being told to.
	body := chunk.GetBody()
	if mutation := common.GetBodyMutation(); mutation != nil {
		switch mutation := mutation.GetMutation().(type) {
		case *ext_proc.BodyMutation_Body:
			body = mutation.Body
		case *ext_proc.BodyMutation_ClearBody:
			if mutation.ClearBody {
				body = nil
			}
		}
	}
	common.BodyMutation = &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_StreamedResponse{
			StreamedResponse: &ext_proc.StreamedBodyResponse{
				Body: bytes.Clone(body),
				// Envoy said whether this was the last chunk. Ending the
				// response any earlier would truncate it, and any later leaves
				// the client waiting on the platform.
				EndOfStream: chunk.GetEndOfStream(),
			},
		},
	}
	return response
}

// endResponseWithImmediateBody converts a body-phase refusal into the last
// body of the response.
//
// The status is already spent, which is what "the headers have gone
// downstream" means: the client will read 200 whatever this says. What can
// still be preserved is the refusal itself and the end, so the body travels
// unchanged and the reply ends the response. The header mutation is dropped
// because Envoy scrubbed those headers at encodeHeaders, long before this.
//
// The turn is over, so the body is marked ended and the trailer that may
// follow neither flushes nor ends anything a second time.
func endResponseWithImmediateBody(
	ctx *RequestContext,
	immediate *ext_proc.ImmediateResponse,
) *ext_proc.ProcessingResponse {
	logging.ComponentWarnEvent("extproc", "body_phase_refusal_downgraded", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      ctx.RequestModel,
		// What the client would have read under BUFFERED, and will not read
		// here. The body still says it; the status line cannot.
		"refused_status": immediate.GetStatus().GetCode(),
	})
	ctx.ResponseBodyEnded = true
	if ctx.IsStreamingResponse {
		// A streamed turn would otherwise be ended a second time by the
		// trailer. The turn is accounted by whichever refusal built this
		// reply, not by the streaming finalizer.
		ctx.StreamingComplete = true
	}
	return buildResponseBodyContinueResponse(
		responseStreamBodyMutation(ctx, bytes.Clone(immediate.GetBody()), true), nil)
}
