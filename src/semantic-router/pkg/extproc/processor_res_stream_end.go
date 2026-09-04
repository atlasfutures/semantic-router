package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// Ending a streamed response, as opposed to ending the message inside it.
//
// The closing frames say the turn is over. They do not finish the HTTP
// response: the Router writes them into a body mutation and then goes on
// swallowing whatever the upstream keeps sending, so the connection stays open
// until the upstream stops or the platform cuts it. Measured on the dev cell
// 2026-09-04, that gap was about 40 s -- the deadline fired at 590 s and curl
// saw 632.8 s.
//
// Envoy's ext_proc has one signal for this and it is only available in one
// mode. StreamedBodyResponse.end_of_stream, on a response-body reply, is what
// makes Envoy inject the chunk with end_stream set and finish the response
// body downstream. It is accepted only where the response body mode is
// FULL_DUPLEX_STREAMED, and in that mode a plain body mutation is refused, so
// the two shapes are not interchangeable and the Router has to send whichever
// one the data plane declared.
//
// Nothing else in the protocol does it. CommonResponse carries no such field;
// CONTINUE_AND_REPLACE stops further ext_proc messages rather than the
// response; ImmediateResponse, once response headers have gone downstream,
// resets the stream and drops its own body, which would take the closing
// frames with it.
//
// Where Envoy declared the plain STREAMED mode, this returns the body mutation
// the Router has always sent and the response is still the platform's to end.
// The cell's ext_proc filter has to set response_body_mode:
// FULL_DUPLEX_STREAMED, with response_trailer_mode: SEND beside it, for the
// deadline to end anything. That configuration also removes the ext_proc
// message_timeout rung of the ladder and forces failure_mode_allow off for the
// stream, so it is a deployment decision rather than one this code can make.
func responseStreamBodyMutation(ctx *RequestContext, body []byte, endOfStream bool) *ext_proc.BodyMutation {
	if ctx == nil || !ctx.FullDuplexResponseBody {
		return &ext_proc.BodyMutation{Mutation: &ext_proc.BodyMutation_Body{Body: body}}
	}
	return &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_StreamedResponse{
			StreamedResponse: &ext_proc.StreamedBodyResponse{
				Body:        body,
				EndOfStream: endOfStream,
			},
		},
	}
}
