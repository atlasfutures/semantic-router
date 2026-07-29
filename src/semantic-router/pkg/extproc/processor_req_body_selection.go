package extproc

import (
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// modifyRequestBodyForSelectedRoute keeps provider body shaping outside the
// request-body orchestrator. ARC owns an artifact-exact body contract; every
// other selector, including rayline_remote, uses the normal selected-model
// mutation path.
func (r *OpenAIRouter) modifyRequestBodyForSelectedRoute(
	openAIRequest *openai.ChatCompletionNewParams,
	upstreamModel string,
	decisionName string,
	useReasoning bool,
	profile *config.ProviderProfile,
	ctx *RequestContext,
) ([]byte, error) {
	if ctx.RaylineARCDispatch != nil {
		return r.modifyRequestBodyForRaylineARC(
			openAIRequest,
			decisionName,
			profile,
			ctx,
		)
	}
	return r.modifyRequestBodyForAutoRouting(
		openAIRequest,
		upstreamModel,
		decisionName,
		useReasoning,
		profile,
		ctx,
	)
}
