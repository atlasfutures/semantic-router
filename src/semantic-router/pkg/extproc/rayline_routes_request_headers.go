package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// answerRaylineRoutesRequestHeaders owns the header phase of a route lookup.
// It runs before the generic header validation so the endpoint's refusals
// carry its own envelope, which needs the request context the generic
// validator does not take. handled is false for every other path.
func (r *OpenAIRouter) answerRaylineRoutesRequestHeaders(
	method string,
	path string,
	v *ext_proc.ProcessingRequest_RequestHeaders,
	ctx *RequestContext,
) (response *ext_proc.ProcessingResponse, handled bool) {
	if normalizeRequestPath(path) != raylineRoutesAPIPath {
		return nil, false
	}
	if refusal := r.validateRaylineRoutesMethod(method, ctx); refusal != nil {
		return refusal, true
	}
	// A route lookup is answered entirely from its body, and Envoy sends no
	// body callback for a header message that already ended the stream. A
	// bodyless POST therefore passes method validation, finds no body phase
	// to answer it, and is continued upstream -- forwarding a request to an
	// endpoint that exists only here and executes nothing. Refusing it now is
	// the only place left that still can.
	if v.RequestHeaders.GetEndOfStream() {
		return r.createRaylineRoutesError(
			ctx,
			400,
			"invalid_request_error",
			"request body is required",
		), true
	}
	return newContinueRequestHeadersResponse(buildIdentityEncodingRequestMutation()), true
}

// validateRaylineRoutesMethod keeps the endpoint invisible where it is not
// configured: a disabled cell answers 404 for every method, so probing it
// cannot distinguish "off here" from "never existed".
func (r *OpenAIRouter) validateRaylineRoutesMethod(
	method string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	// Both refusals use this endpoint's own envelope. They are produced in
	// the header phase, before the body-phase producer runs, so without this
	// they are rewritten into the source format's error shape -- which for
	// this path resolves to Chat -- and a caller sees a different contract
	// for a 404 or 405 than for every other status.
	if !r.raylineRoutesAPIEnabled() {
		return r.createRaylineRoutesError(ctx, 404, "not_found_error", "endpoint not found")
	}
	if method != "POST" {
		return r.createRaylineRoutesError(ctx, 405, "invalid_request_error", "method not allowed")
	}
	return nil
}
