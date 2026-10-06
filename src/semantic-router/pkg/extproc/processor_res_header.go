package extproc

import (
	"net/http"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// handleResponseHeaders processes the response headers.
func (r *OpenAIRouter) handleResponseHeaders(v *ext_proc.ProcessingRequest_ResponseHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	if skipResp := r.handleSkipProcessingResponseHeaders(v, ctx); skipResp != nil {
		return skipResp, nil
	}
	if looperResp := r.handleLooperResponseHeaders(v, ctx); looperResp != nil {
		return looperResp, nil
	}

	outcome := evaluateResponseHeaderOutcome(v, ctx)
	if ctx != nil {
		// Persist the upstream status so the later response-body cache-write
		// path can avoid caching non-2xx error bodies (cache poisoning).
		ctx.UpstreamStatusCode = outcome.statusCode
		if ctx.IsStreamingResponse {
			// The silence clock starts when the arm accepts the turn.
			ctx.StreamContentSince = time.Now()
		}
	}
	if empty := r.emptySuccessResponse(v, ctx, outcome); empty != nil {
		return empty, nil
	}
	if unavailable := r.commitSelectionOnResponseHeaders(v, ctx, outcome); unavailable != nil {
		return unavailable, nil
	}
	finishUpstreamResponseSpan(ctx, outcome)
	maybeRecordResponseHeaderTTFT(ctx)
	r.updateRouterReplayStatus(ctx, outcome.statusCode, ctx != nil && ctx.IsStreamingResponse)
	r.observeRouterLearningProviderStatus(ctx, outcome.statusCode)
	if ctx != nil && !outcome.isSuccessful && v != nil && v.ResponseHeaders.GetEndOfStream() {
		// A provider error with no body ends here: no body callback will
		// class it or settle its usage, so both happen now, from the status.
		r.recordUpstreamErrorTurn(ctx, nil)
	}

	headerMutation := buildResponseHeaderMutation(ctx, outcome.isSuccessful)
	headerMutation = mergeHeaderMutations(headerMutation, buildResponseStreamingMutation(ctx, outcome))
	return buildResponseHeadersContinueResponse(headerMutation, ctx != nil && ctx.IsStreamingResponse), nil
}

// emptySuccessResponse refuses a 2xx that ends at its headers on an inference
// request: no body callback will validate it, settle its usage or let a turn
// commit, and an empty reply is not a served one. It is the arm's
// upstream_error, refused while the headers can still be replaced and before
// any turn commits on it.
func (r *OpenAIRouter) emptySuccessResponse(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
	outcome responseHeaderOutcome,
) *ext_proc.ProcessingResponse {
	if ctx == nil || ctx.SemanticRequest == nil || !outcome.isSuccessful || v == nil || !v.ResponseHeaders.GetEndOfStream() {
		return nil
	}
	recordTurnFailureDetail(ctx, turnFailureUpstreamError, "empty_response", false)
	// The provider attempts are counted as for any reply, and as a failure.
	captureRaylineARCProviderAttempts(v.ResponseHeaders.GetHeaders(), ctx, http.StatusBadGateway)
	// The usual header-phase bookkeeping still runs: the span ends, and the
	// replay and learning records see the status.
	finishUpstreamResponseSpan(ctx, outcome)
	r.updateRouterReplayStatus(ctx, outcome.statusCode, ctx.IsStreamingResponse)
	r.observeRouterLearningProviderStatus(ctx, outcome.statusCode)
	r.reportFailedCallUsage(ctx)
	response := r.createErrorResponse(http.StatusBadGateway, "The selected model returned an empty response")
	// It names the arm that failed, as every upstream error does.
	appendImmediateHeader(response, headers.VSRSelectedModel, ctx.VSRSelectedModel)
	return response
}
