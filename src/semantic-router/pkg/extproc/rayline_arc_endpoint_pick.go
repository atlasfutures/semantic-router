package extproc

import (
	"net/http"
	"slices"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Which OpenRouter endpoint serves an ARC turn (option C, memex-desktop
// D-optc-mixed-price). An arm whose pin reaches more than one endpoint of its
// model, as the routing-price feed records them, is sent to one of them,
// picked by the conversation's session key with the gateway's rendezvous
// hash (raylinearc.PickEndpoint), so the conversation
// keeps hitting one endpoint's prompt cache, the gateway and the cell agree
// on the endpoint whichever side picks, and x-vsr-served-endpoint can name
// the endpoint that served, the one the gateway bills.
//
// The pick is a single-endpoint pin with no OpenRouter fallbacks; a failure
// before any byte reached the client is re-sent on the next endpoints of the
// rendezvous order, at most endpointResendMaxExtra of them, through the flex
// re-send's transport (flex_resend.go). A response that has begun is never
// re-sent.

const (
	// The sources of a turn's session key, as logged (session_key_source).
	sessionKeySourceHeader   = "header"
	sessionKeySourceFallback = "fallback"
	sessionKeySourceNone     = "none"

	// The sources of an arm's endpoints, as logged (endpoint_source): the
	// routing-price feed's rows, or none when no feed is in effect and the
	// arm's pin goes out unpicked.
	endpointSourceFeed = "feed"
	endpointSourceNone = "none"

	// endpointResendMaxExtra is how many further endpoints a turn tries once
	// its pick failed before any byte.
	endpointResendMaxExtra = 2

	// endpointResendAttempt and endpointResendServed mark a re-sent turn's
	// usage lines (endpoint_resend): every refused call, and the call that
	// answered the client.
	endpointResendAttempt = "endpoint_attempt"
	endpointResendServed  = "endpoint_resend"
)

// raylineARCSessionKey is the key the turn's endpoint is picked by: the
// gateway's x-rayline-session-key, read from the trusted signal headers, or
// when it sent none a digest computed here from the episode id the gateway
// keys as user:conversation, with further components for an episode scope
// or a subagent's branch, which are dropped as the gateway's key drops them.
//
// The fallback is best-effort. It equals the gateway's key for a
// conversation id without ":"; a conversation id with one is split short,
// and a sessionless turn (keyed by the gateway on its request prefix, while
// its episode id names the route) gets a key of its own. On a cell that does
// not trust the gateway's headers the episode id is the client's to choose,
// which is harmless: the key only chooses among the arm's own endpoints. An
// episode id that is not user:conversation is hashed whole; a turn with no
// episode has no key.
func raylineARCSessionKey(signalHeaders map[string]string, rawEpisodeID string) (string, string) {
	if key := strings.TrimSpace(signalHeaders[raylineARCSessionKeyHeader]); key != "" {
		return key, sessionKeySourceHeader
	}
	if rawEpisodeID == "" {
		return "", sessionKeySourceNone
	}
	parts := strings.SplitN(rawEpisodeID, ":", 3)
	if len(parts) < 2 {
		return raylinearc.HashEpisodeID(rawEpisodeID), sessionKeySourceFallback
	}
	return raylinearc.SessionKeyDigest(parts[0], parts[1]), sessionKeySourceFallback
}

// soleEndpoint is the endpoint a pin names alone with no OpenRouter fallback
// (its only list, or its order when it sets no only list, of one tag), or ""
// when OpenRouter may choose: several tags, fallbacks allowed, no pin, or a
// slug the feed says reaches several endpoints of the model.
func soleEndpoint(
	pin *config.OpenRouterProviderPreferences,
	settlement *raylinearc.RoutingPriceSettlement,
	model string,
) string {
	if pin == nil || pin.AllowFallbacks == nil || *pin.AllowFallbacks {
		return ""
	}
	listed := pin.Only
	if len(listed) == 0 {
		listed = pin.Order
	}
	slugs := []string{}
	for _, slug := range listed {
		if slug = strings.TrimSpace(slug); slug != "" && !slices.Contains(slugs, slug) {
			slugs = append(slugs, slug)
		}
	}
	if len(slugs) != 1 ||
		(settlement != nil && len(raylinearc.PinRange(pin.Only, pin.Order, pin.Ignore, settlement.EndpointTags(model))) > 1) {
		return ""
	}
	return slugs[0]
}

// pinnedEndpointPreferences pins an arm to its endpoint tag: order and only
// are the tag and fallbacks are off, so only the tag can serve; every other
// member of the arm's pin travels as configured.
//
// TODO(endpoint tag format): pinnedProviderBlock (memex-desktop
// endpointPick.ts) also ignores the arm's tier endpoints under the tag
// (pinning xai/zdr ignores xai/zdr/us) and the gateway will price the
// pinned block with the endpoint's own max_price. Both wait on the endpoint
// tag format; until then the arm's configured ignore and max_price stand.
func pinnedEndpointPreferences(
	pin *config.OpenRouterProviderPreferences,
	tag string,
) *config.OpenRouterProviderPreferences {
	pinned := config.OpenRouterProviderPreferences{}
	if pin != nil {
		pinned = *pin
	}
	noFallbacks := false
	pinned.Order = []string{tag}
	pinned.Only = []string{tag}
	pinned.AllowFallbacks = &noFallbacks
	return &pinned
}

// privacyAllowedArmEndpoints is the endpoint tags a private turn's gateway
// allows the arm, and whether it listed the arm at all.
func privacyAllowedArmEndpoints(ctx *RequestContext, arm string) ([]string, bool) {
	endpoints := ctx.RaylineARCPrivacyAllowedEndpoints
	if ctx.RaylineARCPrivacyMode == "" || !endpoints.present || endpoints.malformed {
		return nil, false
	}
	allowed, listed := endpoints.arms[strings.TrimSpace(arm)]
	return allowed, listed
}

// pickArmEndpoint returns the pin an ARC turn's call is sent with. The arm's
// candidates are the endpoints its pin reaches among the routing-price
// feed's rows for the model (raylinearc.PinRange), kept on a private turn to
// the ones its gateway allows the arm. With more than one candidate and a
// session key, the call is pinned to the session's pick and the rendezvous
// order is recorded for the re-send; a private turn narrowed to one
// candidate is pinned to it. Otherwise, and on a cell with no feed in effect
// (endpoint_source none: the pin's strings are not endpoint records), the
// arm's pin goes out unchanged. An unpinned arm is pinned only on a private
// turn whose gateway listed its endpoints.
func pickArmEndpoint(
	ctx *RequestContext,
	dispatch *providerDispatch,
	pin *config.OpenRouterProviderPreferences,
) *config.OpenRouterProviderPreferences {
	if ctx == nil || ctx.VSRRaylineARC == nil || dispatch == nil {
		return pin
	}
	settlement := routingPriceSettlementOf(ctx.VSRRaylineARC.SettlementPrices)
	if settlement == nil {
		ctx.RaylineARCEndpointSource = endpointSourceNone
		return pin
	}
	ctx.RaylineARCEndpointSource = endpointSourceFeed
	allowed, listed := privacyAllowedArmEndpoints(ctx, dispatch.logicalModel)
	if pin == nil && !listed {
		return pin
	}
	var only, order, ignore []string
	if pin != nil {
		only, order, ignore = pin.Only, pin.Order, pin.Ignore
	}
	reach := raylinearc.PinRange(only, order, ignore, settlement.EndpointTags(dispatch.upstreamModel))
	candidates := reach
	if listed {
		candidates = raylinearc.AllowedEndpoints(reach, allowed)
	}
	narrowed := len(candidates) < len(reach)
	switch {
	case len(candidates) == 0, len(candidates) == 1 && !narrowed:
		return pin
	case len(candidates) > 1 && ctx.RaylineARCSessionKey == "":
		return pin
	}
	picked := candidates
	if len(candidates) > 1 {
		picked = raylinearc.PickEndpoint(ctx.RaylineARCSessionKey, candidates)
	}
	ctx.RaylineARCEndpointOrder = picked
	ctx.RaylineARCEndpointsTried = []string{picked[0]}
	return pinnedEndpointPreferences(pin, picked[0])
}

// dispatchedEndpoint is the endpoint an ARC call is known to be served by:
// the pick when the cell picked, else the one tag the sent pin names alone
// (soleEndpoint); "" when OpenRouter chooses.
func dispatchedEndpoint(ctx *RequestContext, sent *config.OpenRouterProviderPreferences, model string) string {
	if len(ctx.RaylineARCEndpointOrder) > 0 {
		return ctx.RaylineARCEndpointOrder[0]
	}
	var settlement *raylinearc.RoutingPriceSettlement
	if ctx.VSRRaylineARC != nil {
		settlement = routingPriceSettlementOf(ctx.VSRRaylineARC.SettlementPrices)
	}
	return soleEndpoint(sent, settlement, model)
}

// resetDispatchedEndpoint clears what an earlier rendering of the body said
// about its endpoint, before the provider pin is applied again.
func resetDispatchedEndpoint(ctx *RequestContext) {
	if ctx == nil {
		return
	}
	ctx.RaylineARCEndpointOrder, ctx.RaylineARCEndpointsTried, ctx.RaylineARCDispatchedEndpoint = nil, nil, ""
	ctx.RaylineARCEndpointSource = ""
}

// addEndpointPickFields writes the turn's endpoint facts into a log record:
// where its session key and its endpoints came from, the endpoints in the
// order it tries them, the ones tried and the one the call is known to be
// served by. Fields it does not hold are left out.
func addEndpointPickFields(fields map[string]interface{}, ctx *RequestContext) {
	if ctx == nil {
		return
	}
	if ctx.RaylineARCSessionKeySource != "" {
		fields["session_key_source"] = ctx.RaylineARCSessionKeySource
	}
	if ctx.RaylineARCEndpointSource != "" {
		fields["endpoint_source"] = ctx.RaylineARCEndpointSource
	}
	if len(ctx.RaylineARCEndpointOrder) > 0 {
		fields["provider_endpoint_order"] = ctx.RaylineARCEndpointOrder
		fields["provider_endpoints_tried"] = ctx.RaylineARCEndpointsTried
	}
	if ctx.RaylineARCDispatchedEndpoint != "" {
		fields["provider_endpoint"] = ctx.RaylineARCDispatchedEndpoint
	}
}

// endpointResendPlan is what re-sending a picked call on the arm's next
// endpoints needs: the dispatch, the arm's own pin, the turn's data mode and
// the body the pick was sent.
type endpointResendPlan struct {
	dispatch    *providerDispatch
	pin         *config.OpenRouterProviderPreferences
	privacyMode string
	dispatched  []byte
}

// stageEndpointResend keeps what the re-send needs when the dispatch picked
// one of several endpoints. A flex arm keeps its own re-send instead: its
// failure is answered on the standard twin.
func (r *OpenAIRouter) stageEndpointResend(ctx *RequestContext, dispatch *providerDispatch, body []byte) {
	if ctx == nil {
		return
	}
	ctx.RaylineARCEndpointResend = nil
	if r == nil || r.Config == nil || dispatch == nil || ctx.FlexResend != nil ||
		len(ctx.RaylineARCEndpointOrder) < 2 {
		return
	}
	// An unpinned arm's re-send is pinned as its pick was, from nothing.
	pin := r.Config.ProviderPreferencesForModel(dispatch.logicalModel)
	ctx.RaylineARCEndpointResend = &endpointResendPlan{
		dispatch:    dispatch,
		pin:         pin,
		privacyMode: ctx.RaylineARCPrivacyMode,
		dispatched:  body,
	}
}

// endpointResendTrigger is the failure class a call's status re-sends on, or
// "": every failure before the first byte, a transport failure (status 0)
// included, except the request's own 400, 413 and 422, which every endpoint
// refuses alike. It is the gateway's retriesOnNextEndpoint
// (memex-desktop firebase/router/src/providers.ts).
func endpointResendTrigger(status int) string {
	switch {
	case status >= 200 && status < 300,
		status == http.StatusBadRequest, status == http.StatusRequestEntityTooLarge,
		status == http.StatusUnprocessableEntity:
		return ""
	case status == 0:
		return turnFailureUpstreamError
	}
	return upstreamFailureClass(status, nil)
}

// endpointResendTags are the endpoints the turn may still try, in rendezvous
// order, within the re-send bound.
func endpointResendTags(ctx *RequestContext) []string {
	limit := min(len(ctx.RaylineARCEndpointOrder), 1+endpointResendMaxExtra)
	if len(ctx.RaylineARCEndpointsTried) >= limit {
		return nil
	}
	return ctx.RaylineARCEndpointOrder[len(ctx.RaylineARCEndpointsTried):limit]
}

// endpointResendCall is one endpoint's re-send.
type endpointResendCall struct {
	tag  string
	call *flexTwinCall
}

// buildEndpointResends builds every re-send the turn may make, before the
// refused call is settled, so a re-send that cannot be built leaves the
// refusal to the ordinary path.
func (r *OpenAIRouter) buildEndpointResends(
	ctx *RequestContext,
	plan *endpointResendPlan,
	tags []string,
) ([]endpointResendCall, string) {
	calls := make([]endpointResendCall, 0, len(tags))
	for _, tag := range tags {
		pin := privacyProviderPreferences(pinnedEndpointPreferences(plan.pin, tag), plan.privacyMode)
		body, ok := withProviderObject(plan.dispatched, pin)
		if !ok {
			return nil, "body_unpinned"
		}
		call, unavailable := r.buildResendCall(ctx, plan.dispatch.logicalModel, plan.dispatch, body)
		if call == nil {
			return nil, unavailable
		}
		calls = append(calls, endpointResendCall{tag: tag, call: call})
	}
	return calls, ""
}

// endpointResendOnResponseHeaders answers a picked call's failure before any
// byte with a re-send on the arm's next endpoints, or returns nil to leave
// the response to the ordinary pipeline. Each refused call writes its own
// usage line; the last re-send's response, served or not, is the client's.
func (r *OpenAIRouter) endpointResendOnResponseHeaders(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if v == nil || v.ResponseHeaders == nil || ctx == nil || ctx.RaylineARCEndpointResend == nil {
		return nil
	}
	plan := ctx.RaylineARCEndpointResend
	status := getStatusFromHeaders(v.ResponseHeaders.GetHeaders())
	trigger := endpointResendTrigger(status)
	tags := endpointResendTags(ctx)
	if trigger == "" || len(tags) == 0 {
		return nil
	}
	// One turn re-sends once: its calls are made here, not by a later hop.
	ctx.RaylineARCEndpointResend = nil
	received := time.Now()
	budget, skipped := r.flexResendBudget()
	fields := map[string]interface{}{
		"request_id": ctx.RequestID, "model": plan.dispatch.logicalModel, "trigger": trigger,
		"endpoint_status": status, "budget_ms": budget.Milliseconds(),
	}
	if skipped != "" {
		fields["reason"] = skipped
		addEndpointPickFields(fields, ctx)
		logging.ComponentWarnEvent("extproc", "endpoint_resend_skipped", fields)
		return nil
	}
	calls, unavailable := r.buildEndpointResends(ctx, plan, tags)
	if calls == nil {
		fields["reason"] = unavailable
		addEndpointPickFields(fields, ctx)
		logging.ComponentWarnEvent("extproc", "endpoint_resend_unavailable", fields)
		return nil
	}
	// The refused call: its headers are all it said.
	outcome := evaluateResponseHeaderOutcome(v, ctx)
	ctx.UpstreamStatusCode = status
	captureRaylineARCProviderAttempts(v.ResponseHeaders.GetHeaders(), ctx, status)
	finishUpstreamResponseSpan(ctx, outcome)
	markAttempt := func(record *llmUsageRecord) { record.EndpointResend = nonEmpty(endpointResendAttempt) }
	r.settleRefusedAttempt(ctx, trigger, status, markAttempt)

	deadline := received.Add(budget)
	statuses := []int{status}
	var result flexTwinResult
	for index, resend := range calls {
		// The call in flight is now this endpoint's, read back off its body.
		ctx.RaylineARCEndpointsTried = append(ctx.RaylineARCEndpointsTried, resend.tag)
		ctx.RaylineARCDispatchedEndpoint = resend.tag
		ctx.DispatchedOutbound = readDispatchedOutboundParams(resend.call.body, plan.dispatch)
		recordDispatchedProviderPin(ctx, resend.call.body)
		result = callFlexTwin(ctx, resend.call, r.flexResendBodyLimit(), deadline)
		statuses = append(statuses, result.status)
		retry := endpointResendTrigger(result.status)
		if retry == "" || index == len(calls)-1 || !time.Now().Before(deadline) {
			break
		}
		r.settleRefusedAttempt(ctx, retry, result.status, markAttempt)
	}
	fields["endpoint_statuses"] = statuses
	fields["resend_latency_ms"] = time.Since(received).Milliseconds()
	if result.err != nil {
		fields["resend_error"] = result.err.Error()
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
		fields["policy_action_id"] = trace.PolicyActionID
	}
	addEndpointPickFields(fields, ctx)
	logging.ComponentEvent("extproc", "endpoint_resend", fields)
	return r.serveResent(ctx, plan.dispatch.logicalModel, result)
}
