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
