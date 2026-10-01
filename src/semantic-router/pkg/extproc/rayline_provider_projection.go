package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// applyRaylineProviderProjection is the fork's provider-dialect step. It runs
// once, at the end of projectProviderRequest, so the projection stays exactly
// what dispatch sends (US-010): upstream's reasoning projection first, then
// the fork's reasoning controls, the ARC worker's base thinking level, the
// upstream session id and the OpenRouter provider pin. Observations of what
// travelled are recorded later, by the dispatch adapter, from the bytes.
//
// original is the body before upstream's reasoning projection rewrote it; the
// client's own reasoning request is read from it.
func (r *OpenAIRouter) applyRaylineProviderProjection(
	original []byte,
	projected []byte,
	mutation *reasoningRequestMutation,
	dispatch *providerDispatch,
	ctx *RequestContext,
	enabled bool,
) ([]byte, *reasoningRequestMutation, error) {
	body, mutation, err := applyRaylineReasoningControls(original, projected, mutation, dispatch, ctx, enabled)
	if err != nil {
		return nil, nil, err
	}
	if body, err = applyRaylineARCWorkerThinking(body, dispatch, ctx); err != nil {
		return nil, nil, err
	}
	if body, err = applyUpstreamSessionID(body, dispatch, ctx); err != nil {
		return nil, nil, err
	}
	if body, err = applyProviderPreferences(body, dispatch, r.Config); err != nil {
		return nil, nil, err
	}
	return body, mutation, nil
}

// applyRaylineReasoningControls applies the fork's reasoning rules on top of
// upstream's family projection:
//
//   - a reasoning turn on OpenRouter carries a reasoning.max_tokens bound
//     derived from the client's own output allowance (applyOpenRouterReasoningBound);
//   - a turn routed to a thinking-off arm loses the client's reasoning request
//     and, on OpenRouter, states reasoning.enabled false, the off-signal the
//     providers there actually read (dropReasoningRequestFromDisabledArm);
//   - a client's explicit thinking-off is honoured, never refused (US-003e):
//     on a model with no reasoning family the codec already rendered the
//     portable off-signal and OpenRouter additionally gets reasoning.enabled
//     false. The mutation then reports the control as applied, so the
//     dispatch adapter's missing-off-control refusal does not fire.
func applyRaylineReasoningControls(
	original []byte,
	projected []byte,
	mutation *reasoningRequestMutation,
	dispatch *providerDispatch,
	ctx *RequestContext,
	enabled bool,
) ([]byte, *reasoningRequestMutation, error) {
	explicitDisable := explicitAnthropicReasoningDisabled(ctx, dispatch)
	if mutation == nil {
		if !explicitDisable {
			return projected, nil, nil
		}
		// A model with no reasoning family: upstream preserved the body.
		parsed, err := parseReasoningRequestMutation(projected)
		if err != nil {
			return nil, nil, err
		}
		parsed.model = dispatch.logicalModel
		if parsed.hasOriginalEffort {
			parsed.requestMap["reasoning_effort"] = parsed.originalReasoningEffort
		}
		mutation = parsed
	} else {
		client, err := parseReasoningRequestMutation(original)
		if err != nil {
			return nil, nil, err
		}
		clientReasoning := snapshotClientReasoningRequest(client)
		transport := resolveProviderReasoningTransport(dispatch.profile)
		if enabled {
			applyOpenRouterReasoningBound(mutation, transport, clientReasoning, clientOutputAllowance(ctx))
		} else {
			dropReasoningRequestFromDisabledArm(mutation, transport, clientReasoning, ctx)
		}
	}
	if explicitDisable {
		stateExplicitReasoningOff(mutation, dispatch)
	}
	body, err := json.Marshal(mutation.requestMap)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to serialize modified request: %w", err)
	}
	return body, mutation, nil
}

// stateExplicitReasoningOff makes sure a client's explicit thinking-off leaves
// the Router as an off-signal the selected backend reads.
func stateExplicitReasoningOff(mutation *reasoningRequestMutation, dispatch *providerDispatch) {
	if usesReasoningObjectTransport(resolveProviderReasoningTransport(dispatch.profile)) {
		// OpenRouter reads a top-level effort as reasoning.effort, and its
		// providers ignore chat_template_kwargs.enable_thinking (ae50b4ceb).
		delete(mutation.requestMap, "reasoning_effort")
		mutation.appliedEffort = ""
		mutation.requestMap["reasoning"] = offSignalForDisabledArm(mutation.requestMap["reasoning"])
		mutation.reasoningApplied = true
		return
	}
	if mutation.reasoningApplied {
		return
	}
	if dispatch.targetFormat == llmprotocol.OpenAIChatV1 {
		if _, present := mutation.requestMap["reasoning_effort"]; !present {
			mutation.requestMap["reasoning_effort"] = reasoningStringValue("none")
		}
	}
	mutation.reasoningApplied = true
}

// recordDispatchedProviderControls reads back the reasoning controls and the
// provider pin the dispatched bytes carry, for the routing record.
func recordDispatchedProviderControls(ctx *RequestContext, body []byte) {
	recordDispatchedReasoningControls(ctx, body)
	recordDispatchedProviderPin(ctx, body)
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

// applyRaylineMessagesProviderRouting adds OpenRouter's routing members to a
// Messages body (#119). The Messages codec owns the reasoning controls and
// projectProviderRequest leaves a Messages body as the codec rendered it, so
// only the upstream session id and the provider pin are added here; both
// self-gate on OpenRouter, so a body bound for Anthropic itself is unchanged.
func applyRaylineMessagesProviderRouting(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
	cfg *config.RouterConfig,
) ([]byte, error) {
	// A direct-model request (no decision) is dispatched as the client sent
	// it, the same rule projectProviderRequest applies to the other wires.
	if dispatch == nil || dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 || dispatch.decisionName == "" {
		return body, nil
	}
	body, err := applyUpstreamSessionID(body, dispatch, ctx)
	if err != nil {
		return nil, err
	}
	return applyProviderPreferences(body, dispatch, cfg)
}
