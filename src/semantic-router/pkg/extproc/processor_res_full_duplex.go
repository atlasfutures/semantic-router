package extproc

import (
	"bytes"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
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
func normalizeFullDuplexResponseBody(
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
	chunk *ext_proc.HttpBody,
) *ext_proc.ProcessingResponse {
	if ctx == nil || !ctx.FullDuplexResponseBody || response == nil {
		return response
	}
	bodyResponse := response.GetResponseBody()
	if bodyResponse == nil {
		// An ImmediateResponse, or a reply in another phase. Both are already
		// legal in either mode.
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
