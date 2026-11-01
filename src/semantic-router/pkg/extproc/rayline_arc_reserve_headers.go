package extproc

import (
	"bytes"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// A streamed ARC turn held for a refusal re-serve may end up served by a
// different arm than the one whose upstream headers arrived. The headers
// reply names the selected model (x-vsr-selected-model and the served
// identity), so while the hold is armed that reply is deferred and built
// again from the turn's state just before the client is sent its first
// bytes: on a re-serve it names the arm that served, on a release the arm
// that was held.
//
// While the hold is armed, its placeholder frames and keepalives are not
// sent: Envoy takes no body reply before the headers reply, and they carry
// nothing the client needs. Once the hold is decided (a re-serve started, the
// arm released, or the hold lapsed) the arm that serves is known, so the next
// reply of any kind, a keepalive included, sends the headers first; with no
// reply to carry them, the loop sends them alone at the hold's end. The
// client's wait for headers is thus at most the hold plus one keepalive
// interval. An immediate response replaces the deferred headers altogether.

// servedHeadersStream is the ext_proc stream of one request. It holds a
// deferred headers reply and sends it ahead of the first reply that carries
// something to the client.
type servedHeadersStream struct {
	ext_proc.ExternalProcessor_ProcessServer
	router *OpenAIRouter
	ctx    *RequestContext
	// pending is the outcome the deferred reply is built from; nil when no
	// reply is deferred.
	pending *responseHeaderOutcome
}

func (r *OpenAIRouter) newServedHeadersStream(
	stream ext_proc.ExternalProcessor_ProcessServer,
	ctx *RequestContext,
) ext_proc.ExternalProcessor_ProcessServer {
	if _, ok := stream.(*servedHeadersStream); ok {
		return stream
	}
	served := &servedHeadersStream{ExternalProcessor_ProcessServer: stream, router: r, ctx: ctx}
	if ctx != nil {
		ctx.servedHeaders = served
	}
	return served
}

// servedHeadersDue reports deferred headers whose hold is decided: they are
// sent now, even with no reply to carry them, so a turn whose upstream sends
// no body still has its headers when the hold ends.
func (r *OpenAIRouter) servedHeadersDue(ctx *RequestContext) bool {
	served := ctx.servedHeaders
	return served != nil && served.pending != nil && !r.holdingForRefusalReserve(ctx)
}

// servedHeadersWait is how long the loop may wait before deferred headers
// fall due at the hold's end; zero when none are deferred.
func servedHeadersWait(ctx *RequestContext) time.Duration {
	served := ctx.servedHeaders
	plan := ctx.RefusalReserve
	if served == nil || served.pending == nil || plan == nil || ctx.StartTime.IsZero() {
		return 0
	}
	if remaining := plan.hold - time.Since(ctx.StartTime); remaining > 0 {
		return remaining + time.Millisecond
	}
	return time.Millisecond
}

// flushPending sends the deferred headers reply, if one is deferred.
func (s *servedHeadersStream) flushPending() error {
	if s.pending == nil {
		return nil
	}
	reply := buildResponseHeadersReply(s.ctx, *s.pending)
	s.pending = nil
	return s.ExternalProcessor_ProcessServer.Send(reply)
}

// deferServedHeaders holds the headers reply of a turn held for a refusal
// re-serve, and reports whether it did. Only a successful upstream's
// CONTINUE reply is deferred; any other reply goes out as built.
func (r *OpenAIRouter) deferServedHeaders(
	stream ext_proc.ExternalProcessor_ProcessServer,
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
) bool {
	served, ok := stream.(*servedHeadersStream)
	if !ok || ctx == nil || ctx.servedHeadersOutcome == nil || !ctx.servedHeadersOutcome.isSuccessful {
		return false
	}
	if _, isHeaders := response.GetResponse().(*ext_proc.ProcessingResponse_ResponseHeaders); !isHeaders {
		return false
	}
	if !r.holdingForRefusalReserve(ctx) {
		return false
	}
	outcome := *ctx.servedHeadersOutcome
	served.pending = &outcome
	return true
}

func (s *servedHeadersStream) Send(response *ext_proc.ProcessingResponse) error {
	if s.pending == nil {
		return s.ExternalProcessor_ProcessServer.Send(response)
	}
	switch response.GetResponse().(type) {
	case *ext_proc.ProcessingResponse_ImmediateResponse:
		// It answers the request in place of the upstream's headers.
		s.pending = nil
		return s.ExternalProcessor_ProcessServer.Send(response)
	case *ext_proc.ProcessingResponse_ResponseBody:
		if heldPlaceholderReply(response) && s.router.holdingForRefusalReserve(s.ctx) {
			return nil
		}
	}
	if err := s.flushPending(); err != nil {
		return err
	}
	return s.ExternalProcessor_ProcessServer.Send(response)
}

// heldPlaceholderReply reports a body reply that carries nothing of the
// turn: an empty answer to a held chunk, a held frame's placeholder or a
// keepalive. An end of stream is not one.
func heldPlaceholderReply(response *ext_proc.ProcessingResponse) bool {
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		return true
	}
	var body []byte
	switch m := mutation.GetMutation().(type) {
	case *ext_proc.BodyMutation_StreamedResponse:
		if m.StreamedResponse.GetEndOfStream() {
			return false
		}
		body = m.StreamedResponse.GetBody()
	case *ext_proc.BodyMutation_Body:
		body = m.Body
	default:
		return false
	}
	return len(body) == 0 || bytes.Equal(body, hostRetryHoldFrame)
}
