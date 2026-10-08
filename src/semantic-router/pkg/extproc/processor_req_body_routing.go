package extproc

import (
	"errors"
	"fmt"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

type routeHeaderState struct {
	setHeaders    []*core.HeaderValueOption
	removeHeaders []string
	profile       *config.ProviderProfile
}

type providerDispatch struct {
	logicalModel   string
	upstreamModel  string
	backendAddress string
	backendName    string
	profile        *config.ProviderProfile
	targetFormat   llmprotocol.WireFormat
	decisionName   string
	useReasoning   bool
}

// prepareProviderDispatch is the only point where a neutral request becomes a
// provider-bound request. Routing and plugins mutate semantic state first;
// the selected backend codec owns the final wire representation.
func (r *OpenAIRouter) prepareProviderDispatch(
	request *llmprotocol.Request,
	logicalModel string,
	decisionName string,
	useReasoning bool,
	ctx *RequestContext,
) (*providerDispatch, error) {
	if request == nil || ctx == nil || r == nil || r.Config == nil {
		return nil, status.Error(codes.Internal, "neutral inference request is unavailable")
	}
	dispatch, err := r.resolveProviderDispatch(logicalModel, decisionName, useReasoning, ctx.SourceFormat)
	if err != nil {
		return nil, err
	}
	changed, err := r.prepareProviderRequest(request, dispatch, ctx)
	if err != nil {
		return nil, err
	}
	// The disposition table makes reasoning carriable by this target before
	// the capability gate, which would otherwise refuse it: a block the target
	// cannot verify is dropped, a signature it cannot verify is stripped. Each
	// is logged by kind. Before it, a Claude worker loses the reasoning that
	// is not provably Claude's, while signatures still say whose it is.
	foreignDropped := r.dropReasoningForeignToClaude(request, dispatch, ctx.SourceFormat)
	if r.signClaudeThinkingForChat(request, dispatch) > 0 {
		changed = true
	}
	carry := protocolcodec.CarryReasoningTo(request, dispatch.targetFormat)
	carry.ForeignDropped = foreignDropped
	if carry.Changed() {
		changed = true
		if carry.Dropped() > 0 {
			logging.ComponentEvent("extproc", "reasoning_dropped", map[string]interface{}{
				"request_id":       ctx.RequestID,
				"model":            logicalModel,
				"wire_format":      dispatch.targetFormat,
				"dropped":          carry.Dropped(),
				"redacted_dropped": carry.RedactedDropped,
				"unsigned_dropped": carry.UnsignedDropped,
				"foreign_dropped":  carry.ForeignDropped,
			})
		}
		if carry.SignaturesStripped > 0 {
			logging.ComponentEvent("extproc", "reasoning_signature_stripped", map[string]interface{}{
				"request_id":          ctx.RequestID,
				"model":               logicalModel,
				"wire_format":         dispatch.targetFormat,
				"signatures_stripped": carry.SignaturesStripped,
			})
		}
	}
	if r.dropRaylineARCOpaqueReasoningIssuedElsewhere(request, dispatch, ctx) {
		changed = true
	}
	if changed {
		request.Generation++
	}
	if protocolErr := r.rejectDispatchCapabilityMismatch(request, dispatch.targetFormat, ctx); protocolErr != nil {
		return nil, protocolErr
	}
	ctx.TargetFormat = dispatch.targetFormat
	ctx.SemanticRequest = request
	logging.ComponentDebugEvent("extproc", "provider_dispatch_prepared", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       logicalModel,
		"backend":     dispatch.backendName,
		"wire_format": dispatch.targetFormat,
	})
	return dispatch, nil
}

func (r *OpenAIRouter) codecCapabilitiesForFormat(format llmprotocol.WireFormat) (llmprotocol.CapabilitySet, bool) {
	if r == nil || r.ProtocolCodecs == nil {
		return protocolcodec.NewBuiltinRegistry().CapabilitiesFor(format)
	}
	return r.ProtocolCodecs.CapabilitiesFor(format)
}

// rejectDispatchCapabilityMismatch applies the same wire-fidelity gate the
// codec engine enforces at encode, but at dispatch time so a request whose
// required capabilities the chosen backend wire cannot express surfaces as a
// protocol error instead of a generic 500 from encoding. This is the first
// slice of capability-driven dispatch: it turns "crash with 500" into "clean
// 400 unsupported_capability" and gives future capability-aware re-routing a
// single check point.
func (r *OpenAIRouter) rejectDispatchCapabilityMismatch(
	request *llmprotocol.Request,
	format llmprotocol.WireFormat,
	ctx *RequestContext,
) error {
	available, ok := r.codecCapabilitiesForFormat(format)
	if !ok {
		// Unknown wire formats are rejected earlier by wireFormatForModel.
		return nil
	}
	if err := llmprotocol.RequireCapabilities(format, available, llmprotocol.RequiredCapabilities(*request)); err != nil {
		var protocolError *llmprotocol.ProtocolError
		if errors.As(err, &protocolError) && ctx != nil {
			ctx.ImmediateProtocolError = protocolError
		}
		return err
	}
	return nil
}

func (r *OpenAIRouter) resolveProviderDispatch(
	logicalModel string,
	decisionName string,
	useReasoning bool,
	clientFormat llmprotocol.WireFormat,
) (*providerDispatch, error) {
	backendAddress, backendName, found, err := r.Config.ResolvePrimaryBackendForModel(logicalModel)
	if err != nil {
		return nil, fmt.Errorf("resolve backend for model %q: %w", logicalModel, err)
	}
	if !found {
		return nil, fmt.Errorf("model %q has no configured backend", logicalModel)
	}
	profile, err := r.Config.GetProviderProfileForEndpoint(backendName)
	if err != nil {
		return nil, fmt.Errorf("resolve provider profile for model %q: %w", logicalModel, err)
	}
	targetFormat, err := r.dispatchTargetFormat(logicalModel, clientFormat)
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", logicalModel, err)
	}
	return &providerDispatch{
		logicalModel: logicalModel, upstreamModel: r.Config.ResolveExternalModelID(logicalModel, backendName),
		backendAddress: backendAddress, backendName: backendName,
		profile: profile, targetFormat: targetFormat,
		decisionName: decisionName, useReasoning: useReasoning,
	}, nil
}

func (r *OpenAIRouter) prepareProviderRequest(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) (bool, error) {
	changed, err := r.materializeResponseObjectContext(request, ctx)
	if err != nil {
		return false, err
	}
	inlined, err := r.resolveImageFileReferences(request)
	if err != nil {
		return false, err
	}
	changed = inlined || changed
	changed = request.Model != dispatch.upstreamModel || request.Stream != ctx.ExpectStreamingResponse || changed
	request.Model = dispatch.upstreamModel
	request.Stream = ctx.ExpectStreamingResponse
	decisionChanged, err := r.applyDispatchDecision(request, dispatch, ctx)
	if err != nil {
		return false, err
	}
	changed = decisionChanged || changed
	paramsChanged, err := r.applyDispatchRequestParams(request, dispatch, ctx)
	if err != nil {
		return false, err
	}
	// The output limit for a request that states none is planned in
	// finalizeProviderDispatchResponse, on the request as it is encoded:
	// tool selection runs between here and there and can change the
	// prompt's size, which the limit is measured against.
	return paramsChanged || changed, nil
}

// boundDispatchRequest sets the output limit of a request that states none
// (the worker's card value, or on Messages the fallback, within the decision's
// cap and the room the prompt leaves in the model's window), then keeps a
// Messages request's limit above a planned v5 thinking budget. It runs on
// the request about to be encoded, after request_params and tool selection,
// so the prompt it measures is the prompt the provider receives. It reports
// whether the request changed, which retires the client's own bytes as the
// dispatch body.
func (r *OpenAIRouter) boundDispatchRequest(request *llmprotocol.Request, dispatch *providerDispatch, ctx *RequestContext) (bool, error) {
	changed, err := r.applyDispatchOutputBound(request, dispatch, ctx)
	if err != nil {
		return false, err
	}
	// A request with no limit of its own already has room above the budget,
	// since the output bound plans it; this raises only a limit the caller
	// stated.
	if planned := ctx.RaylineARCThinkingControl; planned != nil && dispatch.targetFormat == llmprotocol.AnthropicMessagesV1 {
		changed = raiseMessagesAllowance(request, planned.control.BudgetTokens) || changed
	}
	return changed, nil
}

func (r *OpenAIRouter) applyDispatchDecision(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) (bool, error) {
	if dispatch.decisionName == "" {
		return false, nil
	}
	// A v5 action's control is planned here, with the route: its admission
	// for this dispatch's (model, provider, format) and the episode's placer.
	// The provider boundary only renders it, and it owns every thinking
	// field, so the router's derived mode is not applied.
	planned, changed, err := planRaylineARCThinkingControl(request, dispatch, ctx, r.Config)
	if err != nil {
		return false, err
	}
	ctx.RaylineARCThinkingControl = planned
	if dispatch.targetFormat != llmprotocol.OpenAIChatV1 && planned == nil {
		changed = r.applySemanticReasoningMode(
			request, dispatch.logicalModel, dispatch.targetFormat, dispatch.useReasoning, ctx.VSRSelectedDecision,
		)
	}
	actionChanged, err := applyRaylineARCPolicyActionReasoning(request, dispatch.targetFormat, ctx)
	if err != nil {
		return false, err
	}
	changed = actionChanged || changed
	injected, err := r.addSemanticSystemPromptIfConfigured(
		request, dispatch.decisionName, dispatch.logicalModel, ctx,
	)
	if err != nil {
		return false, err
	}
	steered, err := r.applyRaylineARCThinkingLever(request, ctx)
	r.applyRaylineARCReasoningIssuer(request, dispatch, ctx)
	return changed || injected || steered, err
}

// dispatchTargetFormat is the wire format a request goes to the model in: the
// client's own when the model accepts it, otherwise the model's first
// accepted format (providers.models[].accepted_formats, or its single
// api_format). Every per-request guard reads providerDispatch.targetFormat,
// never the model's configured format.
func (r *OpenAIRouter) dispatchTargetFormat(model string, clientFormat llmprotocol.WireFormat) (llmprotocol.WireFormat, error) {
	return wireFormatForModel(r.Config.ResolveModelTargetAPIFormat(model, apiFormatForWire(clientFormat)))
}

func apiFormatForWire(format llmprotocol.WireFormat) string {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return config.APIFormatOpenAI
	case llmprotocol.OpenAIResponsesV1:
		return config.APIFormatResponses
	case llmprotocol.AnthropicMessagesV1:
		return config.APIFormatAnthropic
	default:
		return ""
	}
}

func wireFormatForModel(apiFormat string) (llmprotocol.WireFormat, error) {
	switch strings.ToLower(strings.TrimSpace(apiFormat)) {
	case "", config.APIFormatOpenAI, "openai.chat", string(llmprotocol.OpenAIChatV1):
		return llmprotocol.OpenAIChatV1, nil
	case config.APIFormatAnthropic, "anthropic.messages", string(llmprotocol.AnthropicMessagesV1):
		return llmprotocol.AnthropicMessagesV1, nil
	case config.APIFormatResponses, "openai.responses", string(llmprotocol.OpenAIResponsesV1):
		return llmprotocol.OpenAIResponsesV1, nil
	default:
		return "", fmt.Errorf("unsupported API format %q", apiFormat)
	}
}

func (r *OpenAIRouter) buildProviderDispatchResponse(
	dispatch *providerDispatch,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if dispatch == nil {
		return r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
	}
	if ctx != nil {
		ctx.DispatchedToOpenRouter = providerIsOpenRouter(dispatch.profile)
	}
	state := &routeHeaderState{
		setHeaders: r.startUpstreamSpanAndInjectHeaders(
			dispatch.logicalModel, dispatch.backendAddress, ctx,
		),
		removeHeaders: append([]string{"content-length"}, faultInjectionHeadersForRemoval()...),
		profile:       dispatch.profile,
	}
	// Provider metadata is applied before credentials so an operator-supplied
	// extra header can never replace the credential selected for this request.
	appendProfileHeaders(&state.setHeaders, dispatch.profile)
	if errorResponse := r.appendProviderCredential(
		state, dispatch.logicalModel, dispatch.backendName, ctx,
	); errorResponse != nil {
		return errorResponse
	}
	appendRoutingHeaders(&state.setHeaders, dispatch.logicalModel)
	setProviderRequestPath(&state.setHeaders, dispatch.profile, dispatch.targetFormat)
	r.applyDecisionHeaderMutations(state, ctx)
	return buildRequestBodyContinueResponse(state, nil, false)
}

// finalizeProviderDispatchResponse serializes the request only after every
// semantic plugin has run. This prevents late tool-selection mutations from
// being lost and keeps provider wire concerns at one boundary.
func (r *OpenAIRouter) finalizeProviderDispatchResponse(
	dispatch *providerDispatch,
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	// This is where the wire becomes known, so it is where the record the
	// decision staged is written. Deferred, so a turn that fails on the way
	// still leaves the line, carrying whatever controls the body had reached.
	defer r.emitRoutingDecision(ctx)
	if dispatch == nil || response == nil {
		return nil, status.Error(codes.Internal, "provider dispatch is unavailable")
	}
	if r.Config != nil {
		ctx.DispatchHostedTools = r.Config.ModelConfig[dispatch.logicalModel].HostedTools
	}
	// OpenRouter takes a name on a Chat tool message, and Kimi behind it
	// refuses a tool message it cannot match to its call. OpenAI's own Chat
	// schema has no such member, so no other backend is sent one.
	ctx.DispatchNamesToolResults = providerIsOpenRouter(dispatch.profile)
	// Marked here, where every dispatch path -- routed and external gateway
	// -- meets, so no path builds a dispatch the rule does not see.
	ctx.DispatchAutoCache = r.claudeAutoCacheDispatch(dispatch, ctx)
	bounded, err := r.boundDispatchRequest(ctx.SemanticRequest, dispatch, ctx)
	if err != nil {
		metrics.RecordRequestError(dispatch.logicalModel, "output_bound_error")
		return nil, status.Errorf(codes.Internal, "bound provider request: %v", err)
	}
	if bounded {
		// The client's bytes no longer describe the dispatch request.
		ctx.SemanticRequest.Generation++
	}
	body, err := r.encodeDispatchRequest(ctx)
	if err != nil {
		metrics.RecordRequestError(dispatch.logicalModel, "serialization_error")
		return nil, status.Errorf(codes.Internal, "encode provider request: %v", err)
	}
	body, err = r.adaptProviderRequest(body, dispatch, ctx)
	if err != nil {
		metrics.RecordRequestError(dispatch.logicalModel, "provider_adapter_error")
		return nil, status.Errorf(codes.Internal, "adapt provider request: %v", err)
	}
	r.auditRaylineARCUpstream(body, ctx)
	common := response.GetRequestBody().GetResponse()
	if common == nil {
		return nil, status.Error(codes.Internal, "provider dispatch response is unavailable")
	}
	if common.HeaderMutation == nil {
		common.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	appendContentLengthHeader(&common.HeaderMutation.SetHeaders, len(body))
	declareAutoCacheBeta(&common.HeaderMutation.SetHeaders, ctx)
	common.BodyMutation = &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: body},
	}
	logging.ComponentDebugEvent("extproc", "provider_dispatch_encoded", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       dispatch.logicalModel,
		"wire_format": dispatch.targetFormat,
		"body_bytes":  len(body),
	})
	return response, nil
}

func (r *OpenAIRouter) startUpstreamSpanAndInjectHeaders(
	model string,
	endpoint string,
	ctx *RequestContext,
) []*core.HeaderValueOption {
	spanContext, upstreamSpan := tracing.StartSpan(
		ctx.TraceContext, tracing.SpanUpstreamRequest, trace.WithSpanKind(trace.SpanKindClient),
	)
	ctx.TraceContext = spanContext
	ctx.UpstreamSpan = upstreamSpan
	tracing.SetSpanAttributes(upstreamSpan,
		attribute.String(tracing.AttrModelName, model),
		attribute.String(tracing.AttrEndpointAddress, endpoint),
	)
	traceHeaders := tracing.InjectTraceContextToSlice(spanContext)
	result := make([]*core.HeaderValueOption, 0, len(traceHeaders))
	for _, header := range traceHeaders {
		result = append(result, &core.HeaderValueOption{Header: &core.HeaderValue{
			Key: header[0], RawValue: []byte(header[1]),
		}})
	}
	return result
}

func resolveProviderAuth(profile *config.ProviderProfile) (authz.LLMProvider, modelcatalog.ProviderAuth, error) {
	if profile == nil {
		return authz.ProviderOpenAI, modelcatalog.ProviderAuth{
			Strategy: "bearer", Header: "Authorization", Prefix: "Bearer",
		}, nil
	}
	providerType, err := profile.ProviderType()
	if err != nil {
		return "", modelcatalog.ProviderAuth{}, fmt.Errorf("resolve provider auth: %w", err)
	}
	providerAuth, err := profile.ResolveAuth()
	if err != nil {
		return "", modelcatalog.ProviderAuth{}, fmt.Errorf("resolve provider auth header: %w", err)
	}
	return authz.LLMProvider(providerType), providerAuth, nil
}

func (r *OpenAIRouter) appendProviderCredential(
	state *routeHeaderState,
	model string,
	backendName string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	provider, providerAuth, err := resolveProviderAuth(state.profile)
	if err != nil {
		return r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
	}
	if providerAuth.Strategy == "none" {
		if r.CredentialResolver != nil {
			state.removeHeaders = append(state.removeHeaders, r.CredentialResolver.HeadersToStrip()...)
		}
		return nil
	}
	if r.CredentialResolver == nil {
		return r.createErrorResponse(500, "Provider credentials are unavailable.")
	}
	state.removeHeaders = append(state.removeHeaders, r.CredentialResolver.HeadersToStrip()...)
	accessKey, err := r.CredentialResolver.KeyForProvider(provider, model, ctx.Headers)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "credential_resolution_failed", map[string]interface{}{
			"request_id": ctx.RequestID, "model": model, "backend": backendName,
		})
		return r.createErrorResponse(401, "Authentication failed. Check your API key configuration.")
	}
	if accessKey == "" {
		return nil
	}
	value := accessKey
	if providerAuth.Prefix != "" {
		value = providerAuth.Prefix + " " + accessKey
	}
	state.setHeaders = append(state.setHeaders, overwriteRequestHeader(providerAuth.Header, value))
	return nil
}

func appendProfileHeaders(headersOut *[]*core.HeaderValueOption, profile *config.ProviderProfile) {
	if profile == nil {
		return
	}
	for key, value := range profile.ExtraHeaders {
		*headersOut = append(*headersOut, overwriteRequestHeader(key, value))
	}
}

func overwriteRequestHeader(key, value string) *core.HeaderValueOption {
	return &core.HeaderValueOption{
		Header:       &core.HeaderValue{Key: key, RawValue: []byte(value)},
		AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
	}
}

func setProviderRequestPath(
	headersOut *[]*core.HeaderValueOption,
	profile *config.ProviderProfile,
	format llmprotocol.WireFormat,
) {
	requestPath := requestWirePath(format)
	if profile != nil {
		if configured, err := profile.ResolveCreatePath(requestWireProtocol(format)); err == nil && configured != "" {
			requestPath = configured
		}
	}
	*headersOut = append(*headersOut, &core.HeaderValueOption{Header: &core.HeaderValue{
		Key: ":path", RawValue: []byte(requestPath),
	}})
}

func appendRoutingHeaders(headersOut *[]*core.HeaderValueOption, model string) {
	if model == "" {
		return
	}
	*headersOut = append(*headersOut, &core.HeaderValueOption{Header: &core.HeaderValue{
		Key: headers.SelectedModel, RawValue: []byte(model),
	}})
}

func appendContentLengthHeader(headersOut *[]*core.HeaderValueOption, bodyLength int) {
	*headersOut = append(*headersOut, &core.HeaderValueOption{Header: &core.HeaderValue{
		Key: "content-length", RawValue: []byte(fmt.Sprintf("%d", bodyLength)),
	}})
}

// faultInjectionHeadersForRemoval lists the headers the router reads for
// itself and the provider must never see. The fault header is stripped whether
// or not the cell reads it, so a caller cannot learn from the provider's
// behaviour whether this cell has the affordance turned on.
func faultInjectionHeadersForRemoval() []string {
	return append([]string{headers.VSRFault}, raylineARCTurnSignalHeadersForRemoval()...)
}

func (r *OpenAIRouter) applyDecisionHeaderMutations(state *routeHeaderState, ctx *RequestContext) {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return
	}
	setHeaders, removeHeaders := r.buildHeaderMutations(ctx.VSRSelectedDecision)
	state.setHeaders = append(state.setHeaders, setHeaders...)
	state.removeHeaders = append(state.removeHeaders, removeHeaders...)
}

func buildRequestBodyContinueResponse(
	state *routeHeaderState,
	bodyMutation *ext_proc.BodyMutation,
	clearRouteCache bool,
) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{
		RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{
			Status: ext_proc.CommonResponse_CONTINUE, ClearRouteCache: clearRouteCache,
			HeaderMutation: &ext_proc.HeaderMutation{
				SetHeaders: state.setHeaders, RemoveHeaders: state.removeHeaders,
			},
			BodyMutation: bodyMutation,
		}},
	}}
}

func (r *OpenAIRouter) getModelParams() map[string]config.ModelParams {
	if r == nil || r.Config == nil {
		return nil
	}
	return r.Config.ModelConfig
}
