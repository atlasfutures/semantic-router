package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// bindDispatchProviderFacts records what the chosen provider binding admits,
// once the dispatch is known, for the encoder to read.
func (r *OpenAIRouter) bindDispatchProviderFacts(dispatch *providerDispatch, ctx *RequestContext) {
	if dispatch == nil || ctx == nil {
		return
	}
	if r.Config != nil {
		ctx.DispatchHostedTools = r.Config.ModelConfig[dispatch.logicalModel].HostedTools
	}
	ctx.DispatchMessageEffortUpdates = dispatch.targetFormat == llmprotocol.OpenAIChatV1 &&
		providerIsOpenRouter(dispatch.profile)
}

// bindDispatchRequestFacts carries those facts onto the request the encoder
// renders.
//
// Only a model admitted with a provider-run tool is sent a client's
// declaration of it; the codec drops and counts it otherwise. A per-message
// reasoning effort (the thinking lever's, or Claude Code's own
// output_config.effort) reaches an OpenRouter Chat provider as
// configuration_update; every other Chat backend omits it with a diagnostic.
func bindDispatchRequestFacts(request *llmprotocol.Request, ctx *RequestContext) {
	if request == nil || ctx == nil {
		return
	}
	request.HostedTools = ctx.DispatchHostedTools
	request.MessageEffortUpdates = ctx.DispatchMessageEffortUpdates
}
