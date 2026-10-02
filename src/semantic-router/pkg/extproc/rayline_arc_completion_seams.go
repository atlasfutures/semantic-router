package extproc

import (
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// cacheResponseUnlessCommitDeferred writes the buffered reply to the response
// cache now, except on a turn that commits at completion (#137): that turn
// caches its reply only once it is recorded, because a reply cached before a
// failed commit would serve the client's retry and leave the turn unrecorded.
func (r *OpenAIRouter) cacheResponseUnlessCommitDeferred(ctx *RequestContext, body []byte) {
	if !selectionCommitsOnCompletion(ctx) {
		r.updateResponseCache(ctx, body)
	}
}

// deferSelectionCommitUntilSent arms a policy-service turn's commit for after
// its complete reply has been sent (runPendingSelectionCompletion), with the
// cache write behind it. It runs after every response check, so a blocked or
// refused reply never arms it. A turn that can no longer commit fails here,
// while the client can still be told.
func (r *OpenAIRouter) deferSelectionCommitUntilSent(ctx *RequestContext, body []byte) *ext_proc.ProcessingResponse {
	if !selectionCommitsOnCompletion(ctx) {
		return nil
	}
	if err := selectionCompletionCommittable(ctx); err != nil {
		recordSelectionLifecycleFailure(ctx, "response_complete", err)
		return r.bodyPhaseErrorResponse(ctx, http.StatusServiceUnavailable, selectionUnavailableMessage(ctx))
	}
	deferSelectionCompletion(ctx, func() { r.updateResponseCache(ctx, body) })
	return nil
}

// bindRequestCostFromUsageRecord carries a priced usage record's cost onto
// the request, where upstream's cost response headers (x-vsr-cost) read it.
// #138 replaced the inline pricing that used to set these fields; an unpriced
// record leaves them unset, as before.
func bindRequestCostFromUsageRecord(ctx *RequestContext, record llmUsageRecord) {
	if ctx == nil || record.Cost == nil {
		return
	}
	ctx.RequestCost = *record.Cost
	ctx.RequestCostCurrency = ""
	if record.Currency != nil {
		ctx.RequestCostCurrency = *record.Currency
	}
	ctx.RequestCostPriced = true
}
