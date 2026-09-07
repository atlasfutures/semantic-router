package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The trailer phases of the ext_proc exchange.
//
// Envoy sends HttpTrailers when the filter's trailer mode is SEND and the
// message actually carries trailers. That mode is not optional under full
// duplex: Envoy 1.34 accepts response_body_mode: FULL_DUPLEX_STREAMED only
// with response_trailer_mode: SEND beside it, and refuses the pair with SKIP.
// So the cell cannot turn full duplex on without the Router being sent
// trailers it never asked for.
//
// The contract is one reply per message, in the phase the message named. A
// TrailersResponse is that reply. Before this the two trailer messages fell
// through the type switch to processUnknownRequest, which answers with a
// RequestBody CONTINUE -- a reply in the request-body phase to a message in
// the response-trailer phase. Envoy treats a reply in the wrong phase as a
// protocol violation and closes the stream, which under failure_mode_allow:
// false is a 500 for the turn.
//
// The Router has nothing to add to a trailer. It carries no routing signal it
// has not already read from the headers and the body, and rewriting one would
// change what the upstream said about a message the client has already been
// given. So the mutation is left nil on purpose: the reply exists to keep the
// phase contract, not to change anything.
//
// A response trailer is more than a phase, though. HttpBody.end_of_stream does
// not mean "the last body chunk": the ext_proc contract is that it means the
// last body message and that no trailers will follow. So an upstream response
// carrying trailers never sets it, and the HttpTrailers message is where the
// response body ends. Anything the Router is still holding has to travel
// before the trailer is answered, because Envoy completes the exchange on that
// reply.

func processRequestTrailers(
	stream ext_proc.ExternalProcessor_ProcessServer,
	_ *ext_proc.ProcessingRequest_RequestTrailers,
) error {
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestTrailers{
			RequestTrailers: &ext_proc.TrailersResponse{},
		},
	}
	return sendResponse(stream, response, "request trailers")
}

func (r *OpenAIRouter) processResponseTrailers(
	stream ext_proc.ExternalProcessor_ProcessServer,
	_ *ext_proc.ProcessingRequest_ResponseTrailers,
	ctx *RequestContext,
) error {
	final, err := r.endResponseBodyAtTrailers(ctx)
	if err != nil {
		logging.Errorf("ending the response body at its trailers failed: %v", err)
		return err
	}
	if final != nil {
		if err := sendResponse(stream, final, "response body"); err != nil {
			return err
		}
	}
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ResponseTrailers{
			ResponseTrailers: &ext_proc.TrailersResponse{},
		},
	}
	return sendResponse(stream, response, "response trailers")
}
