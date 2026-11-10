package extproc

import (
	"math/rand/v2"
	"net/http"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The re-send of a transient upstream refusal
// (global.router.response_stream.transient_503_resend).
//
// OpenRouter's edge sometimes answers a call with a 503 whose body is not
// JSON, within tens of milliseconds of the dispatch and before any provider
// was reached (prod cell 2026-11-09, 22:02-22:05Z, every model family). The
// body is no provider's error, so the turn failed as upstream_5xx with
// nothing a client could act on, where one re-send would have served it.
//
// So when an OpenRouter call's response headers state a 503 and a content
// type that is not JSON (absent, or neither application/json nor a +json
// type), the Router
// waits a jittered 250-500 ms and sends the dispatched call again, once: the
// same body member for member, to the same endpoint with the same model and
// credential. A JSON 503 is a provider stating an error and is never
// re-sent, nor is any other status. The decision is made at the response
// headers, before any content, so the re-send's response reaches the client
// as one ImmediateResponse through the flex re-send's transport and pipeline
// (flex_resend.go), within the same budget (response_stream.resend_budget_sec).
//
// The re-send never stacks on another: a turn with a flex or endpoint
// re-send staged keeps that re-send, a re-served arm gets none, and a turn
// re-sends this way at most once. Both calls are journaled: the refused call
// writes its own usage line (transient_resend: transient_503_attempt),
// settled as unknown with no cost, and the re-send the turn's ordinary line
// (transient_resend: transient_503_resend), whose outcome, served or failed,
// is the client's.
//
// Only OpenRouter calls are re-sent: OpenRouter is one endpoint, so the
// re-send reaches the same one, where a pool of replicas could put it on
// another. Like the other re-sends served whole, the re-send's reply is not
// held for a refusal re-serve (rayline_arc_refusal_reserve.go): a refusal on
// it passes through, as on a re-served arm, where without the re-send the
// turn would have failed.

const (
	transientResendAttempt = "transient_503_attempt"
	transientResendServed  = "transient_503_resend"

	transientResendMinBackoff = 250 * time.Millisecond
	transientResendMaxBackoff = 500 * time.Millisecond
)

// transientResendBackoff is the wait before the re-send. Tests replace it.
var transientResendBackoff = func() time.Duration {
	return transientResendMinBackoff +
		rand.N(transientResendMaxBackoff-transientResendMinBackoff+1)
}

// transientResendPlan is what the re-send needs: the dispatch and the body it
// was sent.
type transientResendPlan struct {
	dispatch   *providerDispatch
	dispatched []byte
}

// stageTransientResend keeps what the re-send needs for an OpenRouter
// dispatch, unless the config turns it off or the dispatch already has a
// flex or endpoint re-send, which answer a 503 their own way.
func (r *OpenAIRouter) stageTransientResend(ctx *RequestContext, dispatch *providerDispatch, body []byte) {
	if ctx == nil {
		return
	}
	ctx.TransientResend = nil
	if r == nil || r.Config == nil || !r.Config.Transient503ResendEnabled() || dispatch == nil ||
		!providerIsOpenRouter(dispatch.profile) || ctx.LooperRequest || len(body) == 0 || ctx.FlexResend != nil || ctx.RaylineARCEndpointResend != nil ||
		ctx.TransientResent {
		return
	}
	ctx.TransientResend = &transientResendPlan{dispatch: dispatch, dispatched: body}
}

// jsonContentType reports whether a content type names JSON:
// application/json or a +json type, parameters aside.
func jsonContentType(value string) bool {
	mediaType, _, _ := strings.Cut(value, ";")
	mediaType = strings.ToLower(strings.TrimSpace(mediaType))
	return mediaType == "application/json" || strings.HasSuffix(mediaType, "+json")
}

// responseContentType is the first content-type a header map states, or "".
func responseContentType(headerMap *core.HeaderMap) string {
	if headerMap == nil {
		return ""
	}
	for _, header := range headerMap.Headers {
		if strings.EqualFold(header.GetKey(), "content-type") {
			return extractHeaderValue(header)
		}
	}
	return ""
}

// transientResendOnResponseHeaders answers a non-JSON 503 at the response
// headers with the dispatched call's one re-send, or returns nil to leave the
// response to the ordinary pipeline.
func (r *OpenAIRouter) transientResendOnResponseHeaders(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if v == nil || v.ResponseHeaders == nil || ctx == nil || ctx.TransientResend == nil {
		return nil
	}
	plan := ctx.TransientResend
	headers := v.ResponseHeaders.GetHeaders()
	status := getStatusFromHeaders(headers)
	contentType := responseContentType(headers)
	if status != http.StatusServiceUnavailable || jsonContentType(contentType) {
		return nil
	}
	// One turn re-sends once, and never on top of a flex re-send.
	ctx.TransientResend = nil
	if ctx.FlexResend != nil || ctx.TransientResent {
		return nil
	}
	received := time.Now()
	backoff := transientResendBackoff()
	fields := map[string]interface{}{
		"request_id": ctx.RequestID, "model": plan.dispatch.logicalModel, "status": status,
		"content_type": contentType, "backoff_ms": backoff.Milliseconds(),
	}
	budget, skipped := r.flexResendBudget()
	fields["budget_ms"] = budget.Milliseconds()
	if skipped == "" && budget-backoff < flexResendMinBudget {
		skipped = "budget_too_short"
	}
	if skipped != "" {
		fields["reason"] = skipped
		logging.ComponentWarnEvent("extproc", "transient_503_resend_skipped", fields)
		return nil
	}
	call, unavailable := r.buildResendCall(ctx, plan.dispatch.logicalModel, plan.dispatch, plan.dispatched)
	if call == nil {
		fields["reason"] = unavailable
		logging.ComponentWarnEvent("extproc", "transient_503_resend_skipped", fields)
		return nil
	}
	// The refused call: its headers are all it said.
	outcome := evaluateResponseHeaderOutcome(v, ctx)
	ctx.UpstreamStatusCode = status
	captureRaylineARCProviderAttempts(headers, ctx, status)
	finishUpstreamResponseSpan(ctx, outcome)
	r.settleRefusedAttempt(ctx, upstreamFailureClass(status, nil), status, func(record *llmUsageRecord) {
		record.TransientResend = nonEmpty(transientResendAttempt)
	})
	// The re-send is served whole, outside full duplex: no host retry can
	// hold its frames or take it over, so the staged plan is dropped.
	ctx.HostRetry = nil
	ctx.TransientResent = true

	time.Sleep(backoff)
	result := callFlexTwin(ctx, call, r.flexResendBodyLimit(), received.Add(budget))
	fields["resend_status"] = result.status
	fields["resend_latency_ms"] = time.Since(received).Milliseconds()
	if result.err != nil {
		fields["resend_error"] = result.err.Error()
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
		fields["policy_action_id"] = trace.PolicyActionID
	}
	logging.ComponentEvent("extproc", "transient_503_resend", fields)
	return r.serveResent(ctx, plan.dispatch.logicalModel, result)
}
