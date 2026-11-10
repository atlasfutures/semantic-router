package extproc

import (
	"encoding/json"
	"math/rand/v2"
	"net/http"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The edge-503 re-send.
//
// OpenRouter's edge sometimes answers a call with a 503 whose body is not
// OpenRouter's JSON error (an empty body or an HTML page), about 30 ms after
// dispatch, before any provider was asked. Those bursts are short, and the
// same call sent again a moment later is served. So when a call's response
// headers say 503 and its body is not a JSON object, the cell sends the
// dispatched bytes again once, on the same arm, after a 250-500 ms jittered
// wait, and the client gets the re-send's answer. Nothing has reached the
// client then: like the thinking-signature guard, the decision is made
// before the refused call's headers are answered, which needs the response
// body under FULL_DUPLEX_STREAMED.
//
// A 503 carrying OpenRouter's JSON error is a provider's answer and takes
// the ordinary path, as does every other status. A flex or endpoint re-send
// staged on the call answers its 503 first (processResponseHeaders): an arm
// picked to one of several endpoints already moves to the next endpoint on
// any 503, and keeps doing so; this re-send serves the arms that have no
// other endpoint to move to.
//
// Privacy invariant: the re-send carries the exact bytes the first call
// dispatched, so it can reach no host the first call could not.
//
// The refused call writes its own usage line (edge_resend: edge_attempt);
// the re-send writes the turn's ordinary line (edge_resend: edge_resend).

const (
	edgeResendAttempt = "edge_attempt"
	edgeResendServed  = "edge_resend"

	// edgeResendBodyWait bounds the wait for a 503's body before its headers
	// are answered, and edgeResendBodyLimit its size.
	edgeResendBodyWait  = 2 * time.Second
	edgeResendBodyLimit = 64 << 10
)

// edgeResendJitter is the wait before the re-send dials: 250-500 ms. A
// variable so tests can shorten it.
var edgeResendJitter = func() time.Duration {
	return 250*time.Millisecond + rand.N(250*time.Millisecond)
}

// edgeResendPlan is what the re-send needs: the dispatch and the bytes it
// was sent.
type edgeResendPlan struct {
	dispatch   *providerDispatch
	dispatched []byte
}

// stageEdgeResend keeps what the re-send needs for a dispatch. The cell's
// provider is OpenRouter; any provider's non-JSON 503 is the same edge failure.
func (r *OpenAIRouter) stageEdgeResend(ctx *RequestContext, dispatch *providerDispatch, body []byte) {
	if ctx == nil {
		return
	}
	ctx.EdgeResend = nil
	if r == nil || r.Config == nil || dispatch == nil || ctx.LooperRequest || len(body) == 0 {
		return
	}
	ctx.EdgeResend = &edgeResendPlan{dispatch: dispatch, dispatched: body}
}

// edgeErrorBody reports whether a 503's body is not OpenRouter's JSON error:
// empty, or anything that is not a JSON object.
func edgeErrorBody(body []byte) bool {
	var object map[string]json.RawMessage
	return json.Unmarshal(body, &object) != nil || object == nil
}

// edgeResendOnResponseHeaders answers a staged call's non-JSON 503 before
// its headers are answered with one re-send of the same bytes, and returns
// the re-send's reply. Otherwise it puts every message it read back, in
// order, and returns nil, so the response takes the ordinary path untouched.
func (r *OpenAIRouter) edgeResendOnResponseHeaders(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if v == nil || v.ResponseHeaders == nil || ctx == nil || ctx.EdgeResend == nil {
		return nil
	}
	plan := ctx.EdgeResend
	status := getStatusFromHeaders(v.ResponseHeaders.GetHeaders())
	if status != http.StatusServiceUnavailable {
		return nil
	}
	ended := v.ResponseHeaders.GetEndOfStream()
	if !ended && (!ctx.FullDuplexResponseBody || ctx.streamReceiver == nil) {
		return nil
	}
	// One turn re-sends once.
	ctx.EdgeResend = nil
	received := time.Now()
	var held []receivedMessage
	var body []byte
	giveBack := func(reason string) *ext_proc.ProcessingResponse {
		if len(held) > 0 {
			ctx.streamReceiver.pushBack(held)
		}
		if reason != "" {
			logging.ComponentWarnEvent("extproc", "edge_resend_skipped", map[string]interface{}{
				"request_id": ctx.RequestID, "model": plan.dispatch.logicalModel, "reason": reason,
			})
		}
		return nil
	}
	if !ended {
		var complete bool
		held, body, complete = ctx.streamReceiver.holdBody(edgeResendBodyWait, edgeResendBodyLimit)
		if !complete {
			return giveBack("body_incomplete")
		}
	}
	if !edgeErrorBody(body) {
		return giveBack("")
	}
	budget, skipped := r.flexResendBudget()
	if skipped != "" {
		return giveBack(skipped)
	}
	wait := edgeResendJitter()
	if wait >= budget {
		return giveBack("budget_too_short")
	}
	call, unavailable := r.buildResendCall(ctx, plan.dispatch.logicalModel, plan.dispatch, plan.dispatched)
	if call == nil {
		return giveBack(unavailable)
	}
	// The refused call: its headers and its body are all it said.
	outcome := evaluateResponseHeaderOutcome(v, ctx)
	ctx.UpstreamStatusCode = status
	captureRaylineARCProviderAttempts(v.ResponseHeaders.GetHeaders(), ctx, status)
	finishUpstreamResponseSpan(ctx, outcome)
	ctx.UpstreamErrorText = readUpstreamErrorText(status, body, nil)
	r.settleRefusedAttempt(ctx, upstreamFailureClass(status, nil), status, func(record *llmUsageRecord) {
		record.EdgeResend = nonEmpty(edgeResendAttempt)
	})
	refusedText := ctx.UpstreamErrorText
	ctx.UpstreamErrorText = upstreamErrorText{}
	// The re-send is served whole, outside full duplex: no host retry can
	// hold its frames or take it over, so the staged plan is dropped.
	ctx.HostRetry = nil
	ctx.EdgeResent = true

	time.Sleep(wait)
	result := callFlexTwin(ctx, call, r.flexResendBodyLimit(), received.Add(budget))
	fields := map[string]interface{}{
		"request_id":        ctx.RequestID,
		"model":             plan.dispatch.logicalModel,
		"refused_status":    status,
		"resend_status":     result.status,
		"wait_ms":           wait.Milliseconds(),
		"resend_latency_ms": time.Since(received).Milliseconds(),
	}
	addUpstreamErrorTextFields(fields, refusedText)
	if result.err != nil {
		fields["resend_error"] = result.err.Error()
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
		fields["policy_action_id"] = trace.PolicyActionID
	}
	logging.ComponentEvent("extproc", "edge_resend", fields)
	return r.serveResent(ctx, plan.dispatch.logicalModel, result)
}
