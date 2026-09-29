package extproc

import (
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// raylineARCPolicyActionLevel is the base-level label a policy action's
// reasoning wire is recorded under; the action_id names the action itself.
const raylineARCPolicyActionLevel = "policy_action"

var errPolicyActionFormat = errors.New("the policy action's reasoning cannot be carried in this provider format")

// raylineARCPolicyAction returns the binding of the action the policy service
// chose for this turn, when the bindings declare what an action dispatches.
// Decision-only bindings leave dispatch to the modelRef, as before.
func raylineARCPolicyAction(ctx *RequestContext) (config.RaylineARCPolicyBinding, bool) {
	if ctx == nil || ctx.VSRRaylineARC == nil || ctx.VSRRaylineARC.PolicyActionID == "" {
		return config.RaylineARCPolicyBinding{}, false
	}
	decision := ctx.VSRSelectedDecision
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.RaylineARC == nil {
		return config.RaylineARCPolicyBinding{}, false
	}
	binding, ok := decision.Algorithm.RaylineARC.PolicyService.PolicyBinding(ctx.VSRRaylineARC.PolicyActionID)
	return binding, ok && binding.DeclaresDispatch()
}

// policyActionWorkerThinking is the action's native reasoning as one wire:
// its effort, its budget, or -- for the package's null effort and null
// budget -- no reasoning control at all. The loader refuses an effort and a
// budget together on a Chat worker, so one control is always enough there.
func policyActionWorkerThinking(action config.RaylineARCPolicyBinding) config.RaylineARCWorkerThinkingConfig {
	switch {
	case action.Effort != nil:
		return config.RaylineARCWorkerThinkingConfig{
			Level: raylineARCPolicyActionLevel, Wire: config.RaylineARCWorkerThinkingEffort, Effort: *action.Effort,
		}
	case action.ReasoningMaxTokens != nil:
		return config.RaylineARCWorkerThinkingConfig{
			Level: raylineARCPolicyActionLevel, Wire: config.RaylineARCWorkerThinkingBudget, MaxTokens: *action.ReasoningMaxTokens,
		}
	default:
		return config.RaylineARCWorkerThinkingConfig{
			Level: raylineARCPolicyActionLevel, Wire: config.RaylineARCWorkerThinkingProviderDefault,
		}
	}
}

// applyRaylineARCPolicyActionReasoning puts the chosen action's native
// reasoning on a request bound for Anthropic Messages, in place of what the
// router derived. Chat carries it on OpenRouter's reasoning object at the
// provider boundary instead (applyRaylineARCWorkerThinking), because the
// Chat encoder would drop an effort beside the derived budget.
//
// On Messages an effort travels as output_config.effort with adaptive
// thinking, a budget as enabled thinking with that budget_tokens, and the
// thinking-off action as disabled thinking. A null effort sends no effort
// and leaves the derived thinking mode alone.
func applyRaylineARCPolicyActionReasoning(
	request *llmprotocol.Request,
	targetFormat llmprotocol.WireFormat,
	ctx *RequestContext,
) (bool, error) {
	action, declared := raylineARCPolicyAction(ctx)
	if !declared || targetFormat == llmprotocol.OpenAIChatV1 {
		return false, nil
	}
	if targetFormat != llmprotocol.AnthropicMessagesV1 {
		return false, errPolicyActionFormat
	}
	before := *request
	switch {
	case action.Effort != nil && *action.Effort == "none":
		request.ReasoningMode, request.ReasoningEffort, request.ReasoningBudgetTokens = llmprotocol.ReasoningModeDisabled, "", nil
	case action.ReasoningMaxTokens != nil:
		budget := *action.ReasoningMaxTokens
		request.ReasoningMode, request.ReasoningBudgetTokens = llmprotocol.ReasoningModeEnabled, &budget
		request.ReasoningEffort = ""
		if action.Effort != nil {
			request.ReasoningEffort = *action.Effort
		}
	case action.Effort != nil:
		request.ReasoningMode, request.ReasoningEffort, request.ReasoningBudgetTokens = llmprotocol.ReasoningModeAdaptive, *action.Effort, nil
	default:
		request.ReasoningEffort = ""
	}
	base := policyActionWorkerThinking(action)
	ctx.RaylineARCWorkerThinking = &base
	return before.ReasoningMode != request.ReasoningMode || before.ReasoningEffort != request.ReasoningEffort ||
		!sameInt64Pointer(before.ReasoningBudgetTokens, request.ReasoningBudgetTokens), nil
}

func sameInt64Pointer(left, right *int64) bool {
	if left == nil || right == nil {
		return left == right
	}
	return *left == *right
}
