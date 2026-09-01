package extproc

import (
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// modifyRequestBodyForSelectedRoute keeps provider body shaping outside the
// request-body orchestrator. ARC owns an artifact-exact body contract; every
// other selector uses the normal selected-model
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
	// TODO(vsr-next Bucket B): re-seat on prepareProviderDispatch. Upstream
	// removed modifyRequestBodyForAutoRouting together with the whole
	// auto-routing body pipeline; this seam is unreachable until Bucket B
	// decides whether ARC keeps a body rewrite at all.
	_, _, _, _ = openAIRequest, upstreamModel, useReasoning, profile
	return nil, errNotPortedBucketB
}
