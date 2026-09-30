package extproc

import (
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// adaptProviderRequest applies backend-dialect extensions after the standard
// wire codec has rendered the request. Official protocol semantics stay in
// llmprotocol/protocolcodec; model-server extensions such as vLLM
// chat_template_kwargs remain isolated at this final provider boundary.
func (r *OpenAIRouter) adaptProviderRequest(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
) ([]byte, error) {
	body, err := applyOpenRouterWebSearch(body, dispatch, ctx)
	if err != nil {
		return nil, err
	}
	if dispatch == nil || ctx == nil || dispatch.decisionName == "" {
		recordDispatchedReasoningControls(ctx, body)
		return body, nil
	}
	if dispatch.targetFormat == llmprotocol.AnthropicMessagesV1 {
		return r.adaptMessagesProviderRequest(body, dispatch, ctx)
	}
	if dispatch.targetFormat != llmprotocol.OpenAIChatV1 {
		family := r.getModelReasoningFamily(dispatch.logicalModel)
		transport := resolveProviderReasoningTransport(dispatch.profile)
		if dispatch.targetFormat != llmprotocol.OpenAIResponsesV1 || family == nil ||
			transport != modelcatalog.ReasoningTransportChatTemplate {
			recordDispatchedReasoningControls(ctx, body)
			return body, nil
		}
	}
	body, err = r.setReasoningModeToRequestBodyForModelAndProvider(
		body,
		dispatch.logicalModel,
		dispatch.useReasoning,
		ctx.VSRSelectedDecision,
		dispatch.profile,
		ctx,
	)
	if err != nil {
		return nil, err
	}
	body, err = applyRaylineARCWorkerThinking(body, dispatch, ctx)
	if err != nil {
		return nil, err
	}
	body, err = applyUpstreamSessionID(body, dispatch, ctx)
	if err != nil {
		return nil, err
	}
	body, err = applyProviderPreferences(body, dispatch, r.Config)
	if err != nil {
		return nil, err
	}
	// Read back rather than remember: the record then names the bytes that
	// travel, whichever mutation put them there.
	recordDispatchedReasoningControls(ctx, body)
	recordDispatchedProviderPin(ctx, body)
	return body, nil
}

// adaptMessagesProviderRequest applies the provider extensions a Messages body
// carries. The Messages codec owns the reasoning controls, so only
// OpenRouter's routing members are added: its Messages API reads the same
// provider pin and session_id as its Chat API. Both self-gate on OpenRouter,
// so a body bound for Anthropic itself is left as the codec rendered it.
func (r *OpenAIRouter) adaptMessagesProviderRequest(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
) ([]byte, error) {
	body, err := applyUpstreamSessionID(body, dispatch, ctx)
	if err != nil {
		return nil, err
	}
	body, err = applyProviderPreferences(body, dispatch, r.Config)
	if err != nil {
		return nil, err
	}
	recordDispatchedReasoningControls(ctx, body)
	recordDispatchedProviderPin(ctx, body)
	return body, nil
}

// clientOutputAllowance is the output allowance the caller stated, present only
// when a Router plugin raised the dispatched one above it. The reasoning bound
// comes from the caller's number, not the raised one.
func clientOutputAllowance(ctx *RequestContext) *int64 {
	if ctx == nil || ctx.SemanticRequest == nil {
		return nil
	}
	return ctx.SemanticRequest.ClientMaxOutputTokens
}
