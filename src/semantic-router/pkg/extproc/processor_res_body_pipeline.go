package extproc

import (
	"errors"
	"net/http"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func (r *OpenAIRouter) handleNonStreamingResponseBody(
	responseBody []byte,
	ctx *RequestContext,
	completionLatency time.Duration,
) *ext_proc.ProcessingResponse {
	usage := invalidResponseTerminalUsage("authoritative_usage_missing")
	semanticResponse, err := r.decodeClientResponse(responseBody, ctx)
	if err == nil {
		err = injectedDecodeFailure(ctx)
	}
	if err != nil {
		metrics.RecordRequestError(ctx.RequestModel, "parse_error")
		r.reportUnusableResponseUsage(ctx, completionLatency, err)
		logging.ComponentErrorEvent("extproc", "neutral_response_decode_failed", map[string]interface{}{
			"request_id":     ctx.RequestID,
			"backend_format": ctx.TargetFormat,
			"client_format":  ctx.SourceFormat,
			"error":          err.Error(),
		})
		return r.upstreamDecodeFailureResponse(ctx, err)
	}
	// A refused policy turn is declined before any response plugin can
	// return early, so a blocked refusal still clears the boundary decision
	// that chose the refusing arm.
	refused := selectionCommitsOnCompletion(ctx) && responseRefused(semanticResponse)
	if refused {
		declineRefusedTurn(ctx)
	}
	// The usage line is written once the turn's outcome is known: after the
	// commit gate below, which classes a policy turn that can no longer
	// commit, or before a response check returns early.
	usageReported := false
	reportUsage := func() {
		if usageReported {
			return
		}
		usageReported = true
		usage = r.takeNeutralResponseUsage(ctx)
		r.reportNonStreamingUsage(ctx, completionLatency, usage)
		r.calibrateTokenEstimator(ctx, usage.promptTokens)
	}
	clientBody := responseBody
	// A same-format body whose own stop field misstates its decoded stop (a
	// Chat cut at the output limit labelled tool_calls) is re-encoded too, so
	// the client is told the length stop rather than handed a call to run.
	rewriteClientBody := requiresClientResponseRewrite(ctx) ||
		protocolcodec.UpstreamBodyMisstatesStop(ctx.TargetFormat, *semanticResponse)
	if rewriteClientBody {
		clientBody, err = r.encodeClientResponse(*semanticResponse, ctx)
		if err != nil {
			// The arm's reply cannot be put in the client's format: an
			// unusable reply, classed and settled before it is refused.
			recordTurnFailureDetail(ctx, turnFailureUpstreamError, responseFailureClass(err), false)
			reportUsage()
			return r.bodyPhaseErrorResponse(ctx, 502, "The selected model returned an incompatible response")
		}
	}

	// A turn that commits at completion caches its reply only once the turn
	// is recorded: a reply cached before a failed commit would serve the
	// client's retry from the cache, and that turn would never be recorded.
	cacheAfterCommit := selectionCommitsOnCompletion(ctx)
	if !cacheAfterCommit {
		r.updateResponseCache(ctx, clientBody)
	}

	// The response-stage signal is scored from the declared rules before any
	// plugin runs, so the observation exists whether or not the selected
	// decision carries a plugin; the plugins below then consume it.
	r.evaluateResponseJailbreakSignal(ctx, semanticAssistantContent(semanticResponse))

	jailbreakResponse := r.performSemanticResponseJailbreakDetection(ctx, semanticResponse)
	// Recorded before a block returns, so a blocked response leaves the same
	// evidence in Router Replay as a delivered one.
	r.recordRouterReplayResponseJailbreak(ctx)
	if jailbreakResponse != nil {
		recordTurnFailureDetail(ctx, turnFailureResponseBlocked, "response_jailbreak", false)
		reportUsage()
		return jailbreakResponse
	}
	if hallucinationResponse := r.performSemanticHallucinationDetection(ctx, semanticResponse); hallucinationResponse != nil {
		recordTurnFailureDetail(ctx, turnFailureResponseBlocked, "response_hallucination", false)
		reportUsage()
		return hallucinationResponse
	}

	// The full body was read, decoded and passed every response check. A
	// policy-service turn commits once this reply has been sent, and caches
	// it only then; a turn that can no longer commit fails now, while the
	// client can still be told.
	// A refused turn (declined above) is delivered but never recorded, and
	// never cached.
	if cacheAfterCommit && !refused {
		if err := selectionCompletionCommittable(ctx); err != nil {
			recordSelectionLifecycleFailure(ctx, "response_complete", err)
			// The cell failed the turn, not the arm: the reply arrived but its
			// turn can no longer be recorded. Checked after every response
			// check, so a lease lost while they ran is still caught.
			recordTurnFailure(ctx, selectionFailureUnavailable, false)
			reportUsage()
			response := r.bodyPhaseErrorResponse(ctx, http.StatusServiceUnavailable, selectionUnavailableMessage(ctx))
			// The class header survives the cell's response scrub, so a
			// gateway can tell this cell failure from an arm's.
			appendImmediateHeader(response, selectionFailureHeader, selectionFailureUnavailable)
			return response
		}
		deferSelectionCompletion(ctx, func() { r.updateResponseCache(ctx, clientBody) })
	}
	// A refusal is classed only once no response check blocked it, so one
	// turn is one failure.
	if responseRefused(semanticResponse) && ctx.ResponseFailureClass == "" {
		recordTurnFailure(ctx, turnFailureRefusal, contentBeforeRefusal(semanticResponse))
	}
	reportUsage()

	r.scheduleSemanticResponseMemoryStore(ctx, semanticResponse)
	r.markUnverifiedFactualResponse(ctx)

	response, finalBody := r.applySemanticResponseWarnings(ctx, semanticResponse, clientBody)
	if rewriteClientBody && response.GetResponseBody().GetResponse().GetBodyMutation() == nil {
		setResponseBodyMutation(response, clientBody)
	}
	r.persistResponseObject(ctx)
	r.updateRouterReplayHallucinationStatus(ctx)
	r.attachRouterReplayResponse(ctx, finalBody, true)
	return response
}

// injectedDecodeFailure turns a request that asked to fail into a decode
// failure, after the upstream call has happened and its body has been read.
// The point is the body phase: a fault that short-circuited earlier would
// exercise a path no real failure takes.
func injectedDecodeFailure(ctx *RequestContext) error {
	if ctx == nil || ctx.InjectedFault != headers.FaultUpstreamDecode {
		return nil
	}
	logging.ComponentWarnEvent("extproc", "fault_injected", map[string]interface{}{
		"fault":      ctx.InjectedFault,
		"request_id": ctx.RequestID,
	})
	return llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable,
		"fault_upstream_decode",
		"upstream response was refused by an injected fault",
		nil,
	)
}

// upstreamDecodeFailureResponse answers a response the codec could not
// decode. The failure already names what was wrong, and at this boundary that
// name is the whole diagnosis: the operator cannot see the body and the client
// cannot tell an unusable response from an unreachable model. So the refusal
// carries the failure's own code and message instead of one canned sentence.
//
// The status stays on the upstream rules: a body that cannot be used is 502
// and an upstream that ran out of time is 504. The message is the protocol
// error's own, never its cause, so no part of the response body travels with
// it.
func (r *OpenAIRouter) upstreamDecodeFailureResponse(
	ctx *RequestContext,
	err error,
) *ext_proc.ProcessingResponse {
	status, message := 502, "The selected model returned an invalid response"
	var protocolError *llmprotocol.ProtocolError
	if errors.As(err, &protocolError) && protocolError.Message != "" {
		message = protocolError.Message
		if protocolError.Category == llmprotocol.ErrorUpstreamTimeout {
			status = 504
		}
		if ctx != nil {
			ctx.ImmediateProtocolError = llmprotocol.NewError(
				protocolError.Category, protocolError.Code, protocolError.Message, nil,
			)
		}
	}
	return r.bodyPhaseErrorResponse(ctx, status, message)
}

// bodyPhaseErrorResponse builds a refusal that replaces a response after the
// response headers have already gone by. Envoy scrubs the keystone headers
// once, at encodeHeaders, so anything set here reaches the client verbatim
// rather than being cleaned up on the way out. The refusal therefore presents
// the published contract itself: the request id, the model that was selected,
// the client protocol, and nothing else the contract does not name.
//
// The success path builds its headers through the response-header mutation
// and is untouched.
func (r *OpenAIRouter) bodyPhaseErrorResponse(
	ctx *RequestContext,
	status int,
	message string,
) *ext_proc.ProcessingResponse {
	response := r.createErrorResponse(status, message)
	immediate := response.GetImmediateResponse()
	if immediate == nil || ctx == nil {
		return response
	}
	published := newResponseHeaderMutationBuilder()
	published.addString("content-type", "application/json")
	published.addString(headers.RequestID, ctx.RequestID)
	published.addString(headers.VSRSelectedModel, ctx.VSRSelectedModel)
	published.addString(headers.VSRClientProtocol, normalizeProtocol(string(ctx.SourceFormat)))
	immediate.Headers = &ext_proc.HeaderMutation{SetHeaders: published.setHeaders}
	return response
}

func (r *OpenAIRouter) applySemanticResponseWarnings(
	ctx *RequestContext,
	semanticResponse *llmprotocol.Response,
	originalBody []byte,
) (*ext_proc.ProcessingResponse, []byte) {
	response := buildResponseBodyContinueResponse(nil, nil)
	changed := false
	var codes []string
	var bodyChanged bool

	bodyChanged, code := r.applySemanticHallucinationWarning(ctx, semanticResponse)
	changed = changed || bodyChanged
	codes = appendNonEmpty(codes, code)
	bodyChanged, code = r.applySemanticUnverifiedFactualWarning(ctx, semanticResponse)
	changed = changed || bodyChanged
	codes = appendNonEmpty(codes, code)
	codes = appendNonEmpty(codes, r.responseJailbreakWarningCode(ctx))

	if len(codes) > 0 {
		setResponseWarningsHeader(response, codes)
	}
	addResponseStageSignalHeaders(ctx, response)
	if !changed {
		return response, originalBody
	}
	encoded, err := r.encodeClientResponse(*semanticResponse, ctx)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "neutral_response_warning_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		return response, originalBody
	}
	setResponseBodyMutation(response, encoded)
	return response, encoded
}

func (r *OpenAIRouter) markUnverifiedFactualResponse(ctx *RequestContext) {
	if ctx.VSRSelectedDecision == nil {
		return
	}

	hallucinationConfig := ctx.VSRSelectedDecision.GetHallucinationConfig()
	if hallucinationConfig != nil && hallucinationConfig.Enabled {
		r.checkUnverifiedFactualResponse(ctx)
	}
}

func appendNonEmpty(codes []string, code string) []string {
	if code == "" {
		return codes
	}
	return append(codes, code)
}

// addResponseStageSignalHeaders rewrites x-vsr-matched-jailbreak in the body
// phase once the response-direction rules have been scored. The response
// headers phase wrote the request-stage matches before the body existed, so
// the debug header would otherwise never show a response-stage match. Same
// gate as the request-stage signal headers: only when debug is requested.
func addResponseStageSignalHeaders(ctx *RequestContext, response *ext_proc.ProcessingResponse) {
	if ctx == nil || len(ctx.VSRMatchedResponseJailbreak) == 0 || !debugHeadersRequested(ctx) {
		return
	}
	matched := make([]string, 0, len(ctx.VSRMatchedJailbreak)+len(ctx.VSRMatchedResponseJailbreak))
	matched = append(matched, ctx.VSRMatchedJailbreak...)
	matched = append(matched, ctx.VSRMatchedResponseJailbreak...)
	setResponseBodyHeader(response, headers.VSRMatchedJailbreak, strings.Join(matched, ","))
}

// setResponseWarningsHeader writes the consolidated x-vsr-response-warnings header
// (comma-separated codes) onto the response, merging with any existing mutation.
func setResponseWarningsHeader(response *ext_proc.ProcessingResponse, codes []string) {
	setResponseBodyHeader(response, headers.VSRResponseWarnings, strings.Join(codes, ","))
}

// setResponseBodyHeader sets one response header from the body phase, merging
// with any header mutation the response already carries.
func setResponseBodyHeader(response *ext_proc.ProcessingResponse, key, value string) {
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return
	}
	if bodyResponse.ResponseBody.Response == nil {
		bodyResponse.ResponseBody.Response = &ext_proc.CommonResponse{}
	}
	opt := &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			RawValue: []byte(value),
		},
	}
	if hm := bodyResponse.ResponseBody.Response.HeaderMutation; hm != nil {
		hm.SetHeaders = append(hm.SetHeaders, opt)
		return
	}
	bodyResponse.ResponseBody.Response.HeaderMutation = &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{opt},
	}
}

func setResponseBodyMutation(response *ext_proc.ProcessingResponse, body []byte) {
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return
	}
	if bodyResponse.ResponseBody.Response == nil {
		bodyResponse.ResponseBody.Response = &ext_proc.CommonResponse{}
	}
	bodyResponse.ResponseBody.Response.BodyMutation = &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: body,
		},
	}
	if bodyResponse.ResponseBody.Response.HeaderMutation == nil {
		bodyResponse.ResponseBody.Response.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	// A body rewrite invalidates the upstream byte count. Let Envoy derive the
	// correct framing instead of forwarding a stale content-length.
	ensureHeaderRemoved(bodyResponse.ResponseBody.Response.HeaderMutation, "content-length")
}

func setResponseContentType(response *ext_proc.ProcessingResponse, contentType string) {
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return
	}
	if bodyResponse.ResponseBody.Response == nil {
		bodyResponse.ResponseBody.Response = &ext_proc.CommonResponse{}
	}
	if bodyResponse.ResponseBody.Response.HeaderMutation == nil {
		bodyResponse.ResponseBody.Response.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	mutation := bodyResponse.ResponseBody.Response.HeaderMutation
	for _, option := range mutation.SetHeaders {
		if option.GetHeader().GetKey() != "content-type" {
			continue
		}
		option.Header.Value = ""
		option.Header.RawValue = []byte(contentType)
		option.AppendAction = core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD
		return
	}
	mutation.SetHeaders = append(mutation.SetHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      "content-type",
			RawValue: []byte(contentType),
		},
		AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
	})
}

func isResponseAPIRequest(ctx *RequestContext) bool {
	return ctx != nil && ctx.SourceFormat == llmprotocol.OpenAIResponsesV1
}
