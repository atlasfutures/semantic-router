package extproc

import (
	"strconv"
	"strings"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// A streamed endpoint-picked turn's in-band re-send
// (global.router.response_stream.endpoint_stream_resend).
//
// An ARC turn whose arm reaches several endpoints is pinned to one of them
// with no OpenRouter fallback (rayline_arc_endpoint_pick.go). A failed status
// is re-sent on the arm's next endpoints at the response headers. OpenRouter
// can instead answer 200 and then send an in-band error event before any
// content (a provider's 5xx): the status path never sees it, and the turn is
// lost although the next endpoint was free.
//
// With the switch on, such a turn is re-sent on the next endpoint of the
// rendezvous order, spliced into the response the client already has open,
// by the host retry's machinery (stream_host_retry.go): the plan is a
// hostRetryPlan in endpoint mode, its frames are held until the first
// content with SSE comments sent in their place, and the re-send streams
// through startHostRetrySplice. Only the bodies differ: each is the endpoint
// re-send's (buildEndpointResends), pinned to its endpoint from the turn's
// feed snapshot, with the turn's privacy preferences and, across host
// families, the pick's signed and encrypted reasoning stripped.
//
// Only an in-band failure before any content moves the turn on, and a
// re-send's own failure before any content (a failed status, a broken or
// empty stream); a first-content stall does not, and is cut as it is without
// the switch. The status re-send and this one share one bound: the endpoints
// tried (RaylineARCEndpointsTried) never pass 1+endpointResendMaxExtra, and a
// turn the status path re-sent is not re-sent in-band (its reply is one whole
// body). The re-sends stop at the turn's deadline, past the re-send budget
// (response_stream.resend_budget_sec) counted from the first of them, and
// never on a refusal re-serve's arm.
//
// For an endpoint-picked turn this plan takes the place of host retry: its
// pin names one endpoint, so host retry could only re-send on that same
// endpoint (single_host_resend), and a stall stays the first-content cut.
//
// Each failed call writes its own usage line (endpoint_resend:
// endpoint_attempt); the call that serves writes the turn's line
// (endpoint_resend: endpoint_resend).

// endpointThinkingState is the run's thinking state as it stood before a
// stripped re-send restarted it.
type endpointThinkingState struct {
	restart bool
	adopted string
}

// stageEndpointStreamResend makes the turn's host-retry plan an endpoint
// re-send when the switch is on and the dispatch picked one of several
// endpoints (stageEndpointResend staged its re-send). It runs after
// stageHostRetry and replaces what that staged.
func (r *OpenAIRouter) stageEndpointStreamResend(ctx *RequestContext, dispatch *providerDispatch, body []byte) {
	if ctx == nil || r == nil || r.Config == nil || !r.Config.ResponseStreamEndpointStreamResend ||
		dispatch == nil || ctx.RaylineARCEndpointResend == nil || ctx.LooperRequest {
		return
	}
	ctx.HostRetry = &hostRetryPlan{
		dispatch:   dispatch,
		dispatched: body,
		attempts:   []hostAttempt{{started: ctx.StartTime}},
		endpoint:   ctx.RaylineARCEndpointResend,
	}
}

// holdingForEndpointResend reports that the call in flight's frames are held
// from the client: nothing of it has been sent, and a re-send could still
// replace it. Unlike host retry it needs no first-content limit, since only
// an in-band failure triggers it.
func (r *OpenAIRouter) holdingForEndpointResend(ctx *RequestContext) bool {
	plan := ctx.HostRetry
	return !refusalReserveServing(ctx) && !plan.exposed && streamSilenceTimerArmed(ctx) && !ctx.StreamContentSeen &&
		!ctx.StreamingComplete && len(endpointResendTags(ctx)) > 0 && r.endpointStreamResendBlocked(ctx, plan) == ""
}

// endpointStreamResendBlocked names why no further endpoint re-send may be
// started, or "".
func (r *OpenAIRouter) endpointStreamResendBlocked(ctx *RequestContext, plan *hostRetryPlan) string {
	if r.responseStreamOverran(ctx) {
		return "deadline_passed"
	}
	if _, skipped := r.flexResendBudget(); skipped != "" {
		return skipped
	}
	if !plan.endpointDeadline.IsZero() && !time.Now().Before(plan.endpointDeadline) {
		return "budget_spent"
	}
	return ""
}

// endpointStreamFailure classes an in-band failure before any content as its
// usage line states it, and reports whether the next endpoint may take it:
// not the request's own error, which every endpoint refuses alike.
func endpointStreamFailure(state *semanticResponseStreamState) (string, string, bool) {
	failed := state.upstreamFailed
	if failed == nil {
		failed = state.failed
	}
	retries := failed == nil || failed.Category != llmprotocol.ErrorInvalidRequest
	return streamFailureClass(failed), providerStreamFailureDetail(failed), retries
}

// endpointStreamRetries reports whether a failure before any content moves
// an endpoint-picked turn on: never a first-content stall, a refusal (the
// refusal re-serve's), a context overflow, nor a re-send's status the status
// path would not re-send.
func endpointStreamRetries(class, detail string) bool {
	switch {
	case detail == streamFirstContentTimeoutCode:
		return false
	case class == turnFailureRefusal || class == turnFailureContextOverflow:
		return false
	case strings.HasPrefix(detail, "upstream_status_"):
		status, err := strconv.Atoi(strings.TrimPrefix(detail, "upstream_status_"))
		return err == nil && endpointResendTrigger(status) != ""
	}
	return true
}

// retryOnNextEndpoint starts the re-send of an endpoint-picked turn whose
// call in flight failed before any content on the next endpoint, or reports
// that there is none: the turn then ends as it would without it.
func (r *OpenAIRouter) retryOnNextEndpoint(ctx *RequestContext, class, detail string) bool {
	plan := ctx.HostRetry
	if refusalReserveServing(ctx) || plan.exposed || ctx.StreamContentSeen || ctx.StreamingComplete ||
		!streamSilenceTimerArmed(ctx) || !endpointStreamRetries(class, detail) {
		return false
	}
	tags := endpointResendTags(ctx)
	if len(tags) == 0 {
		return false
	}
	resend := plan.endpoint
	failedEndpoint := ""
	if tried := ctx.RaylineARCEndpointsTried; len(tried) > 0 {
		failedEndpoint = tried[len(tried)-1]
	}
	fields := map[string]interface{}{
		"request_id": ctx.RequestID, "model": resend.dispatch.logicalModel,
		"trigger": class, "failure_detail": detail, "failed_endpoint": failedEndpoint,
		"next_endpoint": tags[0], "attempt": len(plan.attempts),
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
		fields["policy_action_id"] = trace.PolicyActionID
	}
	if reason := r.endpointStreamResendBlocked(ctx, plan); reason != "" {
		fields["reason"] = reason
		addEndpointPickFields(fields, ctx)
		logging.ComponentWarnEvent("extproc", "endpoint_stream_resend_unavailable", fields)
		return false
	}
	calls, unavailable := r.buildEndpointResends(ctx, resend, tags[:1])
	if calls == nil {
		fields["reason"] = unavailable
		addEndpointPickFields(fields, ctx)
		logging.ComponentWarnEvent("extproc", "endpoint_stream_resend_unavailable", fields)
		return false
	}
	next := calls[0]
	// From here the turn's re-sends are this plan's: the status path never
	// answers a later hop of it.
	ctx.RaylineARCEndpointResend = nil
	if plan.endpointDeadline.IsZero() {
		budget, _ := r.flexResendBudget()
		plan.endpointDeadline = time.Now().Add(budget)
		fields["budget_ms"] = budget.Milliseconds()
	}

	// The failed call, settled while the turn still names its endpoint.
	r.settleStalledHost(ctx, class, detail, func(record *llmUsageRecord) {
		record.EndpointResend = nonEmpty(endpointResendAttempt)
	})
	plan.stop()
	plan.attempts = append(plan.attempts, hostAttempt{started: time.Now()})
	plan.held, plan.exposed = nil, false
	plan.dispatched = next.call.body
	// The call in flight is now the next endpoint's, read back off its body,
	// with the reasoning controls it carried.
	ctx.RaylineARCEndpointsTried = append(ctx.RaylineARCEndpointsTried, next.tag)
	ctx.RaylineARCDispatchedEndpoint = knownEndpoint(ctx.RaylineARCEndpointFeed, resend.dispatch.upstreamModel, next.tag, true)
	resetStreamForHostRetry(ctx, next.call.body, plan.dispatch)
	r.noteEndpointHistory(ctx, plan, next)
	if next.body != nil {
		fields["reasoning_stripped"] = next.stripped.stripped
	}
	addEndpointPickFields(fields, ctx)
	logging.ComponentEvent("extproc", "endpoint_stream_resend", fields)
	r.initializeSemanticResponseStream(ctx)
	plan.splice = startHostRetrySplice(ctx, next.call, 0)
	return true
}

// noteEndpointHistory records the history a re-send carries. A stripped one
// is logged as the status path logs it, is the body the episode records, and
// restarts the run at its reply; a later re-send that carries the history
// whole puts back what the strip changed.
func (r *OpenAIRouter) noteEndpointHistory(ctx *RequestContext, plan *hostRetryPlan, next endpointResendCall) {
	transaction := ctx.RaylineARCTransaction
	if next.body != nil {
		logEncryptedReasoningStrip(ctx, plan.endpoint, next)
		r.auditRaylineARCUpstream(next.body, ctx)
		if transaction != nil && transaction.thinkingWorker != "" {
			if plan.thinking == nil {
				plan.thinking = &endpointThinkingState{
					restart: transaction.thinkingRestart, adopted: transaction.thinkingAdopted,
				}
			}
			transaction.thinkingRestart, transaction.thinkingAdopted = true, ""
		}
		return
	}
	if saved := plan.thinking; saved != nil {
		r.auditRaylineARCUpstream(next.call.body, ctx)
		if transaction != nil {
			transaction.thinkingRestart, transaction.thinkingAdopted = saved.restart, saved.adopted
		}
		plan.thinking = nil
	}
}
