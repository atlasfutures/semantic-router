package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// clearClientMessageEffortForARC removes the client's own per-message
// reasoning effort on a rayline_arc turn, so the only per-message effort that
// travels is the one the thinking lever inserts.
//
// On an ARC turn the action (or the lever) owns the reasoning level; under
// dispatch_effort: provider_default (#117) it is deliberately absent. Since
// the upstream merge the Messages codec decodes a client message's
// output_config.effort into Message.ReasoningEffort, which would reach the
// provider as configuration_update (OpenRouter Chat) or output_config.effort
// (Messages) and override the action. A message that existed only to carry
// that effort is dropped rather than sent empty. Non-ARC decisions keep
// upstream's behaviour and forward the client's effort.
func clearClientMessageEffortForARC(request *llmprotocol.Request, ctx *RequestContext) bool {
	if request == nil || ctx == nil || ctx.VSRSelectedDecision == nil ||
		ctx.VSRSelectedDecision.Algorithm == nil ||
		ctx.VSRSelectedDecision.Algorithm.Type != config.RaylineARCAlgorithmType {
		return false
	}
	changed := false
	kept := request.Messages[:0:0]
	for _, message := range request.Messages {
		if message.ReasoningEffort != "" {
			message.ReasoningEffort = ""
			changed = true
			if len(message.Content) == 0 {
				continue
			}
		}
		kept = append(kept, message)
	}
	if changed {
		request.Messages = kept
	}
	return changed
}
