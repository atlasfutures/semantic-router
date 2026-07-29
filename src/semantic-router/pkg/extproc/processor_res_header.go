package extproc

import (
	"context"
	"fmt"
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
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
	}
	if err := finalizeSelectionResponseHeaders(
		ctx,
		outcome.isSuccessful,
	); err != nil {
		recordSelectionLifecycleFailure(
			ctx,
			"response_headers",
			err,
		)
		return r.createErrorResponse(
			http.StatusServiceUnavailable,
			selectionUnavailableMessage(ctx),
		), nil
	}
	finishUpstreamResponseSpan(ctx, outcome)
	maybeRecordResponseHeaderTTFT(ctx)
	r.updateRouterReplayStatus(ctx, outcome.statusCode, ctx != nil && ctx.IsStreamingResponse)
	r.observeRouterLearningProviderStatus(ctx, outcome.statusCode)

	headerMutation := buildResponseHeaderMutation(ctx, outcome.isSuccessful)
	if outcome.isSuccessful && isResponseAPIStreamRequest(ctx) {
		headerMutation = mergeHeaderMutations(headerMutation, responseAPIStreamingHeaderMutation())
	}
	return buildResponseHeadersContinueResponse(headerMutation, ctx != nil && ctx.IsStreamingResponse), nil
}

func finalizeRaylineARCResponseHeaders(
	requestContext *RequestContext,
	successful bool,
) error {
	if requestContext != nil &&
		requestContext.SelectionTransaction != nil {
		return finalizeSelectionResponseHeaders(
			requestContext,
			successful,
		)
	}
	if requestContext == nil ||
		requestContext.RaylineARCTransaction == nil {
		return nil
	}
	finalizeContext, cancel := context.WithTimeout(
		context.Background(),
		episodeFinalizeTimeout,
	)
	defer cancel()
	if !successful {
		return requestContext.RaylineARCTransaction.abort(
			finalizeContext,
			"upstream_non_2xx",
		)
	}
	if err := requestContext.RaylineARCTransaction.commit(
		finalizeContext,
		requestContext,
	); err != nil {
		return fmt.Errorf(
			"rayline ARC episode commit failed (class=%s)",
			boundedARCEpisodeFailure(err),
		)
	}
	return nil
}
