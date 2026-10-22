package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// The standard re-send of a discount-endpoint call (ARC Contributor, formerly
// Flex; memex-desktop#7454, "Guidance: the Flex 429 -> standard fallback").
//
// A discount endpoint (OpenRouter's openai/flex, or a contributor route) serves
// the same model as its standard endpoint at a lower price and is throttled
// harder. When a call on an arm that names a standard twin
// (providers.models[].standard_twin) is refused with a 429, or times out
// before any response byte (a 408 or 504 at the response headers), the Router
// re-sends it once on the twin: the dispatched body member for member (same
// model, reasoning, max_tokens, tools), with only the provider object replaced
// by the twin's. There is no other retry, and never another model: the model the
// decision chose is the model served, and only the endpoint and the price
// differ.
//
// Both calls are journaled. The refused attempt writes its own llm_usage line
// (flex_resend: flex_attempt) settled from what it stated: a refusal at the
// headers states no usage, so the line is usage_source unknown, pricing
// no_usage and cost null, never a free call, because the provider may have
// completed and charged a call it timed out. The re-send then runs through the
// ordinary response pipeline, so its line (flex_resend: standard_resend), its
// failure class and its selection commit are the same as any call's, priced
// at the twin's rates.
//
// The decision is made at the response headers, the last moment the response
// can still be replaced under both BUFFERED and FULL_DUPLEX_STREAMED, so the
// re-send's response reaches the client as one ImmediateResponse. A streamed
// turn's events therefore arrive together once the re-send has finished.
//
// Envoy waits for the reply to the response-header message only as long as
// its ext_proc message_timeout, so the whole re-send has to fit inside it.
// The Router cannot read that timeout, so the budget is stated:
// global.router.response_stream.resend_budget_sec, 55 s by default (the 60 s
// the repository's Gateway API manifests ship, less a margin). A budget too
// short to be worth a call is not spent: the refusal then takes the ordinary
// path, settled as unknown like any provider error.

const (
	flexResendAttempt  = "flex_attempt"
	flexResendStandard = "standard_resend"
	// flexResendDefaultBudget is the budget when none is configured.
	flexResendDefaultBudget = 55 * time.Second
	// flexResendMinBudget is the least budget a re-send is attempted with.
	flexResendMinBudget = 5 * time.Second
	// flexResendMaxBody bounds the re-send's body when the cell sets no
	// response-body limit of its own.
	flexResendMaxBody int64 = 64 << 20
)

// flexResendHTTPClient dials the twin directly. The cell's Envoy route already
// spent its retries on the refused call; the re-send is the one extra call.
var flexResendHTTPClient = &http.Client{}

// flexResendPlan is the re-send staged when a discount-endpoint arm is
// dispatched. Staging checks only what the config decides (the twin resolves
// to the same model on the same wire); the twin's body and credential are
// built only when a refusal triggers the re-send, so a served turn pays for
// nothing but keeping the dispatched body.
type flexResendPlan struct {
	flexModel    string
	twinModel    string
	twinDispatch *providerDispatch
	twinPin      *config.OpenRouterProviderPreferences
	// twinTier and flexTier are the arms' configured service tiers, and
	// clientTier the client's own the flex arm replaced; the re-send
	// carries the twin's (withResendServiceTier). resentTier is the tier the
	// re-send body carries, for its usage line.
	twinTier   string
	flexTier   string
	clientTier []byte
	resentTier string
	// dispatched is the body the refused call was sent.
	dispatched []byte
	// trigger is the failure class of the refused call, set once the re-send
	// is under way; the response pipeline is then settling the twin's call.
	trigger string
}

// served reports that the call being settled is the standard re-send.
func (plan *flexResendPlan) served() bool {
	return plan != nil && plan.trigger != ""
}

// stageFlexResend prepares the re-send for a dispatch whose arm names a
// standard twin. Any arm without one is left exactly as it was.
func (r *OpenAIRouter) stageFlexResend(ctx *RequestContext, dispatch *providerDispatch, body []byte) {
	if ctx == nil {
		return
	}
	ctx.FlexResend = nil
	if r == nil || r.Config == nil || dispatch == nil || ctx.LooperRequest {
		return
	}
	twin := r.Config.StandardTwinForModel(dispatch.logicalModel)
	if twin == "" {
		return
	}
	plan, reason := r.planFlexResend(ctx, dispatch, twin, body)
	if plan == nil {
		// The arm keeps serving; it is only its re-send that is unavailable.
		logging.ComponentWarnEvent("extproc", "flex_resend_unavailable", map[string]interface{}{
			"request_id":     ctx.RequestID,
			"flex_model":     dispatch.logicalModel,
			"standard_model": twin,
			"reason":         reason,
		})
		return
	}
	ctx.FlexResend = plan
}

// planFlexResend stages the re-send, or names why there is none. The twin
// must be the same provider model on the same wire, both on OpenRouter, where
// the provider object is what selects the endpoint.
func (r *OpenAIRouter) planFlexResend(
	ctx *RequestContext,
	dispatch *providerDispatch,
	twin string,
	body []byte,
) (*flexResendPlan, string) {
	if !providerIsOpenRouter(dispatch.profile) {
		return nil, "not_openrouter"
	}
	twinDispatch, err := r.resolveProviderDispatch(twin, dispatch.decisionName, dispatch.useReasoning, ctx.SourceFormat)
	if err != nil {
		return nil, "twin_unresolved"
	}
	switch {
	case !providerIsOpenRouter(twinDispatch.profile):
		return nil, "twin_not_openrouter"
	case twinDispatch.upstreamModel != dispatch.upstreamModel:
		return nil, "different_model"
	case twinDispatch.targetFormat != dispatch.targetFormat:
		return nil, "different_wire_format"
	}
	pin := r.Config.ProviderPreferencesForModel(twin)
	if pin == nil {
		return nil, "twin_unpinned"
	}
	return &flexResendPlan{
		flexModel:    dispatch.logicalModel,
		twinModel:    twin,
		twinDispatch: twinDispatch,
		// The twin replaces the provider object, so a private turn's data mode
		// travels with it.
		twinPin:    privacyProviderPreferences(pin, ctx.RaylineARCPrivacyMode),
		twinTier:   r.Config.ServiceTierForModel(twin),
		flexTier:   r.Config.ServiceTierForModel(dispatch.logicalModel),
		clientTier: ctx.ClientServiceTier,
		dispatched: body,
	}, ""
}

// flexTwinCall is the request the twin is sent.
type flexTwinCall struct {
	url     string
	headers http.Header
	body    []byte
}

// buildFlexTwinRequest builds the twin's request from the refused call's
// body, or names why it cannot.
func (r *OpenAIRouter) buildFlexTwinRequest(ctx *RequestContext, plan *flexResendPlan) (*flexTwinCall, string) {
	twinBody, ok := withProviderObject(plan.dispatched, plan.twinPin)
	if !ok {
		// A body without the discount pin was never on the discount endpoint.
		return nil, "flex_unpinned"
	}
	twinBody, ok = withResendServiceTier(twinBody, plan.twinTier, plan.flexTier, plan.clientTier)
	if !ok {
		return nil, "twin_body_invalid"
	}
	plan.resentTier = dispatchedServiceTier(twinBody)
	call, unavailable := r.buildResendCall(ctx, plan.twinModel, plan.twinDispatch, twinBody)
	if unavailable == "address_invalid" {
		unavailable = "twin_address_invalid"
	}
	return call, unavailable
}

// buildResendCall addresses a re-send body to its dispatch's backend, with the
// credential and profile headers the dispatch would set, or names why it
// cannot. The flex twin and an arm's next endpoint (rayline_arc_endpoint_pick.go)
// are both sent this way.
func (r *OpenAIRouter) buildResendCall(
	ctx *RequestContext,
	model string,
	dispatch *providerDispatch,
	body []byte,
) (*flexTwinCall, string) {
	state := &routeHeaderState{profile: dispatch.profile}
	appendProfileHeaders(&state.setHeaders, dispatch.profile)
	if refused := r.appendProviderCredential(state, model, dispatch.backendName, ctx); refused != nil {
		return nil, "credential_unavailable"
	}
	setProviderRequestPath(&state.setHeaders, dispatch.profile, dispatch.targetFormat)
	declareAutoCacheBeta(&state.setHeaders, ctx)
	target, headers, err := flexResendRequestTarget(dispatch, state.setHeaders, ctx)
	if err != nil {
		return nil, "address_invalid"
	}
	return &flexTwinCall{url: target, headers: headers, body: body}, ""
}

// withProviderObject replaces the dispatched body's provider object with the
// twin's and keeps every other member's value as dispatched. The body is
// re-encoded through a map, so member order and escaping follow
// encoding/json, as the dispatch's own provider-pin encoding already did. It
// reports false for a body that carries no provider object.
func withProviderObject(body []byte, pin any) ([]byte, bool) {
	var members map[string]json.RawMessage
	if json.Unmarshal(body, &members) != nil {
		return nil, false
	}
	if _, pinned := members[providerPreferencesField]; !pinned {
		return nil, false
	}
	encoded, err := json.Marshal(pin)
	if err != nil {
		return nil, false
	}
	members[providerPreferencesField] = encoded
	replaced, err := json.Marshal(members)
	if err != nil {
		return nil, false
	}
	return replaced, true
}

// flexClientHeaders are the client's own headers Envoy forwards to the
// provider that change what a Messages call means. The re-send carries them
// so its controls are the refused call's.
var flexClientHeaders = []string{"anthropic-version", "anthropic-beta"}

// flexResendRequestTarget turns the dispatch headers into the twin's URL and
// HTTP headers: the backend's scheme and address, the dispatch path, and the
// credential and profile headers the dispatch would set.
func flexResendRequestTarget(
	dispatch *providerDispatch,
	setHeaders []*core.HeaderValueOption,
	ctx *RequestContext,
) (string, http.Header, error) {
	scheme := "http"
	if dispatch.profile != nil && dispatch.profile.BaseURL != "" {
		parsed, err := url.Parse(dispatch.profile.BaseURL)
		if err != nil || parsed.Scheme == "" {
			return "", nil, fmt.Errorf("invalid base_url")
		}
		scheme = parsed.Scheme
	}
	headers := http.Header{}
	for _, name := range flexClientHeaders {
		if value, ok := lookupHeaderIgnoringCase(ctx.Headers, name); ok && value != "" {
			headers.Set(name, value)
		}
	}
	path := ""
	for _, option := range setHeaders {
		header := option.GetHeader()
		if header == nil {
			continue
		}
		value := string(header.GetRawValue())
		if value == "" {
			value = header.GetValue()
		}
		if header.GetKey() == ":path" {
			path = value
			continue
		}
		if strings.HasPrefix(header.GetKey(), ":") {
			continue
		}
		headers.Set(header.GetKey(), value)
	}
	if path == "" || dispatch.backendAddress == "" {
		return "", nil, fmt.Errorf("no path or address")
	}
	headers.Set("content-type", "application/json")
	return scheme + "://" + dispatch.backendAddress + path, headers, nil
}

// flexResendTrigger is the failure class a response-header status re-sends
// on, or "" when it does not: a 429, or a timeout before any byte.
func flexResendTrigger(ctx *RequestContext, status int) string {
	if ctx == nil || ctx.FlexResend == nil || ctx.FlexResend.served() {
		return ""
	}
	switch status {
	case http.StatusTooManyRequests:
		return turnFailureRateLimited
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		return turnFailureTimeout
	default:
		return ""
	}
}

// flexResendOnResponseHeaders answers a discount-endpoint call's refusal with
// its standard re-send's response, or returns nil to leave the response to
// the ordinary pipeline.
func (r *OpenAIRouter) flexResendOnResponseHeaders(
	v *ext_proc.ProcessingRequest_ResponseHeaders,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if v == nil || v.ResponseHeaders == nil {
		return nil
	}
	status := getStatusFromHeaders(v.ResponseHeaders.GetHeaders())
	trigger := flexResendTrigger(ctx, status)
	if trigger == "" {
		return nil
	}
	received := time.Now()
	plan := ctx.FlexResend
	budget, skipped := r.flexResendBudget()
	if skipped != "" {
		// Nothing is spent on a re-send that cannot answer in time: the
		// refusal takes the ordinary path and settles as unknown there.
		logging.ComponentWarnEvent("extproc", "flex_resend_skipped", map[string]interface{}{
			"request_id": ctx.RequestID, "flex_model": plan.flexModel,
			"standard_model": plan.twinModel, "trigger": trigger, "reason": skipped,
			"budget_ms": budget.Milliseconds(),
		})
		return nil
	}
	call, unavailable := r.buildFlexTwinRequest(ctx, plan)
	if call == nil {
		logging.ComponentWarnEvent("extproc", "flex_resend_unavailable", map[string]interface{}{
			"request_id": ctx.RequestID, "flex_model": plan.flexModel,
			"standard_model": plan.twinModel, "reason": unavailable,
		})
		return nil
	}
	// The refused call: its headers are all it said.
	outcome := evaluateResponseHeaderOutcome(v, ctx)
	ctx.UpstreamStatusCode = status
	captureRaylineARCProviderAttempts(v.ResponseHeaders.GetHeaders(), ctx, status)
	finishUpstreamResponseSpan(ctx, outcome)
	r.settleRefusedAttempt(ctx, trigger, status, func(record *llmUsageRecord) {
		record.FlexResend = nonEmpty(flexResendAttempt)
	})

	plan.trigger = trigger
	started := time.Now()
	result := callFlexTwin(ctx, call, r.flexResendBodyLimit(), received.Add(budget))
	fields := map[string]interface{}{
		"request_id":          ctx.RequestID,
		"flex_model":          plan.flexModel,
		"standard_model":      plan.twinModel,
		"trigger":             trigger,
		"flex_status":         status,
		"standard_status":     result.status,
		"standard_latency_ms": time.Since(started).Milliseconds(),
		"budget_ms":           budget.Milliseconds(),
	}
	if result.err != nil {
		fields["standard_error"] = result.err.Error()
	}
	if trace := ctx.VSRRaylineARC; trace != nil {
		fields["episode_id_hash"] = trace.EpisodeIDHash
		fields["policy_action_id"] = trace.PolicyActionID
	}
	logging.ComponentEvent("extproc", "flex_resend", fields)
	return r.serveFlexTwin(ctx, plan, result)
}

// settleRefusedAttempt writes a refused call's usage line, which mark tags as
// the re-sent kind it is. It states no usage, so its counts are null and its
// cost unknown, never zero. The turn has not failed (the re-send may yet
// serve it), so the turn's own failure state is left for the re-send to set.
func (r *OpenAIRouter) settleRefusedAttempt(
	ctx *RequestContext,
	trigger string,
	status int,
	mark func(*llmUsageRecord),
) {
	latency := time.Duration(0)
	if !ctx.StartTime.IsZero() {
		latency = time.Since(ctx.StartTime)
	}
	usage := responseUsageMetrics{}
	record := r.newLLMUsageRecord(ctx, usage)
	latencyMillis := latency.Milliseconds()
	record.CompletionLatencyMS = &latencyMillis
	record.FailureClass = nonEmpty(trigger)
	record.FailureDetail = nonEmpty("upstream_status_" + strconv.Itoa(status))
	contentSent := false
	record.ContentSentBeforeFailure = &contentSent
	mark(&record)
	r.priceUsageRecord(&record, ctx, ctx.RequestModel, usage)
	emitLLMUsageRecord(record)
	// The endpoint failed the call: it counts against the arm in the
	// load-balancing window, as any backend error does.
	metrics.RecordModelWindowedRequest(
		ctx.RequestModel, latency.Seconds(), 0, 0, true, trigger == turnFailureTimeout,
	)
}

func (r *OpenAIRouter) flexResendBodyLimit() int64 {
	if r != nil && r.Config != nil && r.Config.MaxResponseBodyBytes > 0 {
		return r.Config.MaxResponseBodyBytes
	}
	return flexResendMaxBody
}

// flexResendBudget is how long the re-send may take, measured from the
// refused call's response headers, or the reason it is not attempted: the
// operator turned it off, or the budget is too short to be worth a call.
func (r *OpenAIRouter) flexResendBudget() (time.Duration, string) {
	budget := flexResendDefaultBudget
	if r != nil && r.Config != nil {
		switch seconds := r.Config.FlexResendBudgetSec; {
		case seconds < 0:
			return 0, "disabled"
		case seconds > 0:
			budget = time.Duration(seconds) * time.Second
		}
	}
	if budget < flexResendMinBudget {
		return budget, "budget_too_short"
	}
	return budget, ""
}

// flexTwinResult is what the twin answered, or why it could not be asked.
// delivered says the twin answered with a status. err on a delivered result
// is a body read that failed after the status: the body is what arrived
// before it.
type flexTwinResult struct {
	status      int
	contentType string
	body        []byte
	delivered   bool
	err         error
}

func callFlexTwin(ctx *RequestContext, call *flexTwinCall, limit int64, deadline time.Time) flexTwinResult {
	parent := ctx.TraceContext
	if parent == nil {
		parent = context.Background()
	}
	callCtx, cancel := context.WithDeadline(parent, deadline)
	defer cancel()
	request, err := http.NewRequestWithContext(callCtx, http.MethodPost, call.url, bytes.NewReader(call.body))
	if err != nil {
		return flexTwinFailure(err)
	}
	request.Header = call.headers.Clone()
	response, err := flexResendHTTPClient.Do(request)
	if err != nil {
		return flexTwinFailure(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err == nil && int64(len(body)) > limit {
		// Past the cell's limit is a breach, as the provider hop treats it.
		return flexTwinFailure(fmt.Errorf("standard re-send body exceeds %d bytes", limit))
	}
	return flexTwinResult{
		status:      response.StatusCode,
		contentType: response.Header.Get("content-type"),
		body:        body,
		delivered:   true,
		err:         err,
	}
}

// flexTwinFailure stands for a re-send that produced no usable response: a
// 504 when it ran out of time (the twin may still have charged it), otherwise
// a 502. The pipeline then classes and settles it as the twin's failure.
func flexTwinFailure(err error) flexTwinResult {
	status := http.StatusBadGateway
	if errors.Is(err, context.DeadlineExceeded) {
		status = http.StatusGatewayTimeout
	}
	return flexTwinResult{status: status, contentType: "application/json", err: err}
}

// flexTwinCutError is the in-band error of a re-sent stream whose body read
// failed: the stream is cut there, and the turn is not committed.
func flexTwinCutError(err error) *llmprotocol.ProtocolError {
	return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "standard_resend_body_cut",
		"the standard re-send's response ended before it was complete", err)
}

// terminalFlexTwinResult settles what a read error leaves of the twin's
// answer. A stream keeps what it carried, usage frames included, and is cut
// with an error; any other body cut short is no reply, so it becomes the
// twin's failure, as does a 2xx that ended in an error before any byte.
func terminalFlexTwinResult(ctx *RequestContext, result flexTwinResult) flexTwinResult {
	if !result.delivered || result.err == nil {
		return result
	}
	success := result.status >= 200 && result.status < 300
	if !success {
		// An error body cut short is still the twin's error.
		return result
	}
	if len(result.body) > 0 && strings.Contains(strings.ToLower(result.contentType), "text/event-stream") {
		ctx.UpstreamBodyCut = flexTwinCutError(result.err)
		return result
	}
	return flexTwinFailure(result.err)
}

// serveFlexTwin answers the client with the twin's response, the dispatch
// facts now the twin's.
func (r *OpenAIRouter) serveFlexTwin(
	ctx *RequestContext,
	plan *flexResendPlan,
	result flexTwinResult,
) *ext_proc.ProcessingResponse {
	ctx.DispatchedProviderOrder = append([]string(nil), plan.twinPin.Order...)
	ctx.DispatchedProviderPrivacy = dispatchedProviderPrivacy{
		dataCollection: plan.twinPin.DataCollection,
		zdr:            plan.twinPin.ZDR,
		allowFallbacks: plan.twinPin.AllowFallbacks,
	}
	ctx.DispatchedServiceTier = plan.resentTier
	ctx.RaylineARCEndpointOrder, ctx.RaylineARCEndpointsTried = nil, nil
	ctx.RaylineARCDispatchedEndpoint = dispatchedEndpoint(ctx, plan.twinPin, plan.twinDispatch.upstreamModel)
	return r.serveResent(ctx, plan.twinModel, result)
}

// serveResent runs a re-send's response through the ordinary response
// pipeline as model's call, as if the cell's provider hop had returned it,
// and answers the client with the result.
func (r *OpenAIRouter) serveResent(
	ctx *RequestContext,
	model string,
	result flexTwinResult,
) *ext_proc.ProcessingResponse {
	result = terminalFlexTwinResult(ctx, result)
	// The call in flight is now the re-send's, priced at its rates and
	// counted against it. The response body phase releases the count; a
	// reply that ends before it does releases it here.
	metrics.DecrementModelActiveRequests(ctx.RequestModel)
	metrics.IncrementModelActiveRequests(model)
	bodyPhaseRan := false
	defer func() {
		if !bodyPhaseRan {
			metrics.DecrementModelActiveRequests(model)
		}
	}()
	ctx.RequestModel = model
	// Nothing the refused call's hop said describes the re-send.
	ctx.UpstreamStatusCode = 0
	ctx.UpstreamAttemptCount, ctx.UpstreamRetryCount, ctx.UpstreamRetryExhausted = 0, 0, false
	// One whole body, delivered as one reply.
	ctx.FullDuplexResponseBody = false

	headerMap := &core.HeaderMap{Headers: []*core.HeaderValue{
		{Key: ":status", RawValue: []byte(strconv.Itoa(result.status))},
		{Key: "content-type", RawValue: []byte(result.contentType)},
		// The re-send is exactly one attempt.
		{Key: upstreamAttemptsHeader, RawValue: []byte("1")},
	}}
	// A twin that answered with no body ends at its headers, as a provider
	// hop's would (an empty 2xx is refused there as upstream_error); a
	// re-send that got no answer goes on to the body phase, which encodes its
	// failure for the client.
	ended := result.delivered && len(result.body) == 0
	headersResponse, err := r.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: headerMap, EndOfStream: ended},
	}, ctx)
	if err != nil {
		return r.createErrorResponse(http.StatusBadGateway, "The re-send could not be processed")
	}
	if headersResponse.GetImmediateResponse() != nil {
		return headersResponse
	}
	setHeaders := headersResponse.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
	body := result.body
	if !ended {
		bodyPhaseRan = true
		bodyResponse, err := r.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: result.body, EndOfStream: true},
		}, ctx)
		if err != nil {
			return r.createErrorResponse(http.StatusBadGateway, "The re-send could not be processed")
		}
		if bodyResponse.GetImmediateResponse() != nil {
			return bodyResponse
		}
		common := bodyResponse.GetResponseBody().GetResponse()
		setHeaders = append(setHeaders, common.GetHeaderMutation().GetSetHeaders()...)
		if mutation := common.GetBodyMutation(); mutation != nil {
			if streamed := mutation.GetStreamedResponse(); streamed != nil {
				body = streamed.GetBody()
			} else if _, replaced := mutation.GetMutation().(*ext_proc.BodyMutation_Body); replaced {
				body = mutation.GetBody()
			}
		}
	}
	return flexTwinImmediateResponse(result, setHeaders, body)
}

// flexTwinImmediateResponse is the client's reply: the twin's status, the
// headers the pipeline set, and the body it produced.
func flexTwinImmediateResponse(
	result flexTwinResult,
	setHeaders []*core.HeaderValueOption,
	body []byte,
) *ext_proc.ProcessingResponse {
	headers := make([]*core.HeaderValueOption, 0, len(setHeaders)+1)
	hasContentType := false
	for _, option := range setHeaders {
		key := strings.ToLower(option.GetHeader().GetKey())
		if key == "" || key == "content-length" || strings.HasPrefix(key, ":") {
			continue
		}
		hasContentType = hasContentType || key == "content-type"
		headers = append(headers, option)
	}
	if !hasContentType && result.contentType != "" {
		headers = append(headers, newHeaderValueOption("content-type", result.contentType))
	}
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status:  &typev3.HttpStatus{Code: statusCodeToEnum(result.status)},
				Headers: &ext_proc.HeaderMutation{SetHeaders: headers},
				Body:    body,
			},
		},
	}
}
