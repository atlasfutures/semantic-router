package extproc

import (
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// commitSelectionOnResponseHeaders is the authoritative selector's
// response-header seam. A non-2xx status aborts the turn. A 2xx status does
// not commit it -- the turn commits only once the client has the whole
// response (finalizeSelectionCompletion) -- but a lost lease must surface
// here, while the client can still be told.
//
// It returns a response only when the request must fail: a turn that can no
// longer commit cannot be reported to the client as a provider success.
func (r *OpenAIRouter) commitSelectionOnResponseHeaders(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
	outcome responseHeaderOutcome,
) *ext_proc.ProcessingResponse {
	if v != nil {
		captureRaylineARCProviderAttempts(
			v.ResponseHeaders.GetHeaders(),
			ctx,
			outcome.statusCode,
		)
	}
	err := finalizeSelectionResponseHeaders(ctx, outcome.isSuccessful)
	if err == nil {
		return nil
	}
	recordSelectionLifecycleFailure(ctx, "response_headers", err)
	return r.createErrorResponse(
		http.StatusServiceUnavailable,
		selectionUnavailableMessage(ctx),
	)
}
