package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// raylineReasoningProjectionForTest runs the reasoning part of the provider
// projection as dispatch does: upstream's family projection, then the fork's
// reasoning controls (bound, thinking-off arm, explicit off-signal).
func raylineReasoningProjectionForTest(
	router *OpenAIRouter,
	body []byte,
	logicalModel string,
	enabled bool,
	decision *config.Decision,
	profile *config.ProviderProfile,
	ctx *RequestContext,
) ([]byte, error) {
	projected, mutation, err := router.projectReasoningRequest(body, logicalModel, enabled, decision, profile)
	if err != nil {
		return nil, err
	}
	target := llmprotocol.OpenAIChatV1
	if ctx != nil && ctx.TargetFormat != "" {
		target = ctx.TargetFormat
	}
	dispatch := &providerDispatch{logicalModel: logicalModel, profile: profile, targetFormat: target}
	encoded, mutation, err := applyRaylineReasoningControls(body, projected, mutation, dispatch, ctx, enabled)
	if err == nil && mutation != nil {
		router.observeReasoningMutation(mutation, enabled)
	}
	return encoded, err
}
