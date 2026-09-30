package extproc

import (
	"errors"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// raylineARCPolicyActionLevel is the base-level label a policy action's
// reasoning wire is recorded under; the action_id names the action itself.
const raylineARCPolicyActionLevel = "policy_action"

var errPolicyActionFormat = errors.New("the policy action's reasoning cannot be carried in this provider format")

// raylineARCDefaultMessagesMaxTokens is the Messages encoder's output limit
// for a request that states none.
const raylineARCDefaultMessagesMaxTokens int64 = 4096

// raylineARCWireTopLevelEffort sends the effort as Chat's top-level
// reasoning_effort, for a provider whose transport reads that field (an
// OpenAI-compatible binding, including one that reaches OpenRouter).
const raylineARCWireTopLevelEffort = "top_level_effort"

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

// raylineARCPolicyDispatchAction is the action as it is dispatched. Under
// dispatch_effort: provider_default its effort and budget are withheld, so
// the provider's default applies; a thinking-off action (effort none) keeps
// its effort, since off is not an effort level. The level, and so the
// steering suffix, is untouched, as is what the decision records.
func raylineARCPolicyDispatchAction(ctx *RequestContext) (config.RaylineARCPolicyBinding, bool) {
	action, declared := raylineARCPolicyAction(ctx)
	if !declared || !ctx.VSRSelectedDecision.Algorithm.RaylineARC.PolicyService.DispatchesProviderDefaultEffort() {
		return action, declared
	}
	if action.Effort == nil || *action.Effort != raylineARCPolicyActionEffortOff {
		action.Effort, action.ReasoningMaxTokens = nil, nil
	}
	return action, true
}

// addPolicyDispatchEffortFields records the action's declared effort and
// budget beside the dispatch mode, so a provider_default turn still names what
// the package chose.
func addPolicyDispatchEffortFields(fields map[string]interface{}, ctx *RequestContext) {
	action, declared := raylineARCPolicyAction(ctx)
	if !declared {
		return
	}
	mode := config.RaylineARCPolicyDispatchEffortDeclared
	if ctx.VSRSelectedDecision.Algorithm.RaylineARC.PolicyService.DispatchesProviderDefaultEffort() {
		mode = config.RaylineARCPolicyDispatchEffortProviderDefault
	}
	fields["policy_dispatch_effort"] = mode
	if action.Effort != nil {
		fields["policy_declared_effort"] = *action.Effort
	}
	if action.ReasoningMaxTokens != nil {
		fields["policy_declared_reasoning_max_tokens"] = *action.ReasoningMaxTokens
	}
}

// raylineARCPolicyActionEffortOff is the package's thinking-off effort.
const raylineARCPolicyActionEffortOff = "none"

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

// policyActionChatWire is the action's reasoning in the shape the provider's
// Chat transport reads: OpenRouter's reasoning object carries an effort, a
// budget or nothing; a top-level transport carries an effort as
// reasoning_effort, or nothing. A budget on a top-level transport, or any
// other transport, has no faithful shape, and the turn fails.
func policyActionChatWire(
	action config.RaylineARCPolicyBinding,
	transport modelcatalog.ReasoningTransport,
) (config.RaylineARCWorkerThinkingConfig, bool, error) {
	base := policyActionWorkerThinking(action)
	if base.Wire == config.RaylineARCWorkerThinkingEffort && base.Effort == "none" {
		// A thinking-off action runs on a use_reasoning:false worker, whose
		// off signal the router already wrote exactly as the artifact mode
		// writes it. Replacing it would send an off signal prod never sends.
		return config.RaylineARCWorkerThinkingConfig{}, false, nil
	}
	switch {
	case usesReasoningObjectTransport(transport):
		return base, true, nil
	case usesTopLevelReasoningEffort(transport) && base.Wire != config.RaylineARCWorkerThinkingBudget:
		if base.Wire == config.RaylineARCWorkerThinkingEffort {
			base.Wire = raylineARCWireTopLevelEffort
		}
		return base, true, nil
	default:
		return config.RaylineARCWorkerThinkingConfig{}, false, errPolicyActionFormat
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
	action, declared := raylineARCPolicyDispatchAction(ctx)
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
		// Messages counts thinking inside max_tokens and refuses a budget
		// that does not leave room below it, so the caller's allowance is
		// kept on top of the action's budget.
		allowance := raylineARCDefaultMessagesMaxTokens
		if request.Sampling.MaxOutputTokens != nil {
			allowance = *request.Sampling.MaxOutputTokens
		}
		if allowance <= budget {
			raised := budget + allowance
			request.Sampling.MaxOutputTokens = &raised
		}
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

// raylineARCPolicyActionsCarriable reports whether every declared action can
// travel to its worker's provider, so an action no provider can carry stops
// the selector arming instead of failing each turn that picks it. Messages
// carries effort and budget itself; Chat needs a reasoning transport that
// reads them (policyActionChatWire); Responses is not served.
//
// Serving Responses needs more than a reasoning transport. A Responses client
// such as Codex resends each reasoning item's encrypted_content, which only
// the provider account and model that issued it can read; the codec drops
// those items on every target today. Forwarding them to their own issuer
// needs the episode to record which workers issued the blobs it has seen, and
// to drop the whole item -- not only the blob -- on a turn bound for another.
// Serving Responses is item D of atlasfutures/semantic-router#108; the issuer
// tracking is #109.
func raylineARCPolicyActionsCarriable(cfg *config.RouterConfig, decision *config.Decision) bool {
	for _, binding := range decision.Algorithm.RaylineARC.PolicyService.Bindings {
		if !binding.DeclaresDispatch() {
			continue
		}
		_, backend, found, err := cfg.ResolvePrimaryBackendForModel(binding.Worker)
		if err != nil || !found {
			return false
		}
		profile, err := cfg.GetProviderProfileForEndpoint(backend)
		if err != nil {
			return false
		}
		format, err := wireFormatForModel(cfg.GetModelAPIFormat(binding.Worker))
		if err != nil {
			return false
		}
		switch format {
		case llmprotocol.AnthropicMessagesV1:
		case llmprotocol.OpenAIChatV1:
			if _, _, err := policyActionChatWire(binding, resolveProviderReasoningTransport(profile)); err != nil {
				return false
			}
		default:
			return false
		}
	}
	return true
}
