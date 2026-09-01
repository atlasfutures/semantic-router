package extproc

import (
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// commitSelectionOnResponseHeaders is the authoritative selector's terminal
// seam on the response path. The first upstream header frame is the earliest
// point at which the provider has accepted the dispatch, so it is where the
// episode commits and where a lost lease must surface.
//
// It returns a response only when the request must fail: a commit that did not
// happen cannot be reported to the client as a provider success.
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
