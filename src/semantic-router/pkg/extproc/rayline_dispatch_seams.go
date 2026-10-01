package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// bindCandidateDispatchFacts is bindDispatchProviderFacts plus
// bindDispatchRequestFacts for a dispatch that encodes its own request, such
// as an out-of-band fallback candidate: the candidate's card decides hosted
// tools and the candidate's provider decides per-message effort updates, not
// the primary's. The primary has already been dispatched and failed, so the
// request context now describes the candidate.
func (r *OpenAIRouter) bindCandidateDispatchFacts(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) {
	r.bindDispatchProviderFacts(dispatch, ctx)
	bindDispatchRequestFacts(request, ctx)
}

// shadowProjectionContext is the read-only view of the request a shadow call
// projects its provider body from. Shadow calls run detached from the request,
// so they get a copy: nothing they record reaches the live context, and the
// primary's ARC worker (whose base thinking level belongs to the primary arm)
// is left out.
func shadowProjectionContext(ctx *RequestContext) RequestContext {
	if ctx == nil {
		return RequestContext{}
	}
	snapshot := RequestContext{
		SourceFormat:        ctx.SourceFormat,
		VSRSelectedDecision: ctx.VSRSelectedDecision,
		VSRRaylineARC:       ctx.VSRRaylineARC,
	}
	if ctx.VSRSelectedCandidate != nil {
		candidate := *ctx.VSRSelectedCandidate
		snapshot.VSRSelectedCandidate = &candidate
	}
	snapshot.VSREligibleModelRefs = append([]config.ModelRef(nil), ctx.VSREligibleModelRefs...)
	if ctx.SemanticRequest != nil {
		// Only the controls the provider step reads.
		snapshot.SemanticRequest = &llmprotocol.Request{
			ReasoningMode:         ctx.SemanticRequest.ReasoningMode,
			ClientMaxOutputTokens: ctx.SemanticRequest.ClientMaxOutputTokens,
		}
	}
	return snapshot
}

// applyShadowProviderProjection runs the same provider step as primary
// dispatch on a shadow target's encoded body: upstream's reasoning projection
// with the fork's reasoning controls, session id and provider pin, and on a
// Messages target the OpenRouter routing members.
func (r *OpenAIRouter) applyShadowProviderProjection(
	body []byte,
	target *shadowTarget,
	decisionName string,
	useReasoning bool,
	snapshot RequestContext,
) ([]byte, error) {
	ctx := snapshot
	dispatch := &providerDispatch{
		logicalModel: target.logicalModel, upstreamModel: target.upstreamModel,
		profile: target.profile, targetFormat: target.format,
		decisionName: decisionName, useReasoning: useReasoning,
	}
	body, mutation, err := r.projectProviderRequest(body, dispatch, &ctx)
	if err != nil {
		return nil, err
	}
	if mutation != nil {
		r.observeReasoningMutation(mutation, useReasoning && !explicitAnthropicReasoningDisabled(&ctx, dispatch))
	}
	return applyRaylineMessagesProviderRouting(body, dispatch, &ctx, r.Config)
}

// adaptPlannedThinkingControl is the provider boundary for a turn whose v5
// policy action planned a thinking control (#124). The control owns every
// thinking field, so it is rendered from the registry and the router's own
// reasoning projection, reasoning controls and worker thinking are skipped;
// OpenRouter's routing members (session id, provider pin) still apply, and
// the record reads back what travels. It runs once per dispatch: render
// stages the placer for the turn's commit, which is why it sits in the
// dispatch adapter and not in projectProviderRequest (automatic output
// projects without dispatching). handled is false on every other turn.
func (r *OpenAIRouter) adaptPlannedThinkingControl(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
) (rendered []byte, handled bool, err error) {
	if dispatch == nil || ctx == nil || ctx.RaylineARCThinkingControl == nil {
		return nil, false, nil
	}
	if body, err = ctx.RaylineARCThinkingControl.render(body, ctx); err != nil {
		return nil, true, err
	}
	if body, err = applyUpstreamSessionID(body, dispatch, ctx); err != nil {
		return nil, true, err
	}
	if body, err = applyProviderPreferences(body, dispatch, r.Config); err != nil {
		return nil, true, err
	}
	recordDispatchedProviderControls(ctx, body)
	return body, true, nil
}

// raiseRaylineARCControlAllowance keeps Messages output room above a planned
// v5 control's thinking budget. It runs after request_params and the
// completion floor, which may cap or raise max_tokens.
func raiseRaylineARCControlAllowance(request *llmprotocol.Request, dispatch *providerDispatch, ctx *RequestContext) bool {
	if ctx == nil || dispatch == nil || ctx.RaylineARCThinkingControl == nil ||
		dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 {
		return false
	}
	return raiseMessagesAllowance(request, ctx.RaylineARCThinkingControl.control.BudgetTokens)
}
