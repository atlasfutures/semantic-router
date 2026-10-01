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
// dispatch_effort: provider_default its effort is withheld, so the provider's
// default applies. Only effort: the v4 training turns lost the effort member
// (output_config on Messages, pathfinder #2655) while a reasoning budget,
// which travels in its own field, reached the provider, so a budget action
// keeps its budget. A thinking-off action (effort none) keeps its effort,
// since off is not an effort level. The level, and so the steering suffix, is
// untouched, as is what the decision records.
func raylineARCPolicyDispatchAction(ctx *RequestContext) (config.RaylineARCPolicyBinding, bool) {
	action, declared := raylineARCPolicyAction(ctx)
	if !declared || !ctx.VSRSelectedDecision.Algorithm.RaylineARC.PolicyService.DispatchesProviderDefaultEffort() {
		return action, declared
	}
	if action.Effort != nil && *action.Effort != raylineARCPolicyActionEffortOff {
		action.Effort = nil
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
// and leaves the derived thinking mode alone. An effort withheld under
// dispatch_effort: provider_default sends adaptive thinking without it, as the
// v4 training turns reached the provider once their output_config was dropped,
// and so a derived enabled mode is never left without a budget.
func applyRaylineARCPolicyActionReasoning(
	request *llmprotocol.Request,
	targetFormat llmprotocol.WireFormat,
	ctx *RequestContext,
) (bool, error) {
	action, declared := raylineARCPolicyDispatchAction(ctx)
	if !declared || targetFormat == llmprotocol.OpenAIChatV1 {
		return false, nil
	}
	if targetFormat == llmprotocol.OpenAIResponsesV1 {
		return applyPolicyActionResponsesReasoning(request, action, ctx)
	}
	if targetFormat != llmprotocol.AnthropicMessagesV1 {
		return false, errPolicyActionFormat
	}
	chosen, _ := raylineARCPolicyAction(ctx)
	effortWithheld := chosen.Effort != nil && action.Effort == nil
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
	case effortWithheld:
		request.ReasoningMode, request.ReasoningEffort, request.ReasoningBudgetTokens = llmprotocol.ReasoningModeAdaptive, "", nil
	default:
		request.ReasoningEffort = ""
	}
	base := policyActionWorkerThinking(action)
	ctx.RaylineARCWorkerThinking = &base
	return before.ReasoningMode != request.ReasoningMode || before.ReasoningEffort != request.ReasoningEffort ||
		!sameInt64Pointer(before.ReasoningBudgetTokens, request.ReasoningBudgetTokens), nil
}

// applyPolicyActionResponsesReasoning puts the action's effort on a request
// bound for Responses, as reasoning.effort in place of what the router or the
// client set. A null effort, and one withheld under dispatch_effort:
// provider_default, send no effort, so the provider's default applies. The
// thinking-off action sends effort none, the off signal the router derives
// for a use_reasoning:false worker, so a client's own effort can never turn
// the off arm back on; readiness admits it only on a worker whose reasoning
// family can say off. Responses has no reasoning budget, so a budget action
// has no faithful shape; readiness refuses one before any turn can pick it
// (policyActionResponsesCarriable).
func applyPolicyActionResponsesReasoning(
	request *llmprotocol.Request,
	action config.RaylineARCPolicyBinding,
	ctx *RequestContext,
) (bool, error) {
	if action.ReasoningMaxTokens != nil {
		return false, errPolicyActionFormat
	}
	effort := ""
	if action.Effort != nil {
		effort = *action.Effort
	}
	base := policyActionWorkerThinking(action)
	ctx.RaylineARCWorkerThinking = &base
	if request.ReasoningEffort == effort {
		return false, nil
	}
	request.ReasoningEffort = effort
	return true, nil
}

// policyActionResponsesCarriable reports whether Responses can carry the
// action to its worker: an effort or no reasoning control, and the
// thinking-off action only on a worker whose reasoning family has an off
// signal (semanticDisabledOpenAIReasoningControls). It has no budget.
func policyActionResponsesCarriable(cfg *config.RouterConfig, action config.RaylineARCPolicyBinding) bool {
	if action.ReasoningMaxTokens != nil {
		return false
	}
	if action.Effort == nil || *action.Effort != raylineARCPolicyActionEffortOff {
		return true
	}
	family := cfg.GetModelReasoningFamily(action.Worker)
	if family == nil {
		return false
	}
	off, _ := semanticDisabledOpenAIReasoningControls(family)
	return off == raylineARCPolicyActionEffortOff
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
// reads them (policyActionChatWire); Responses carries an effort but no
// budget (policyActionResponsesCarriable).
//
// A Responses client such as Codex also resends each reasoning item's
// encrypted_content, which only the target that issued it can read. The
// episode records which targets issued the blobs its client holds, and a turn
// forwards them only to that same target, dropping the whole item on any
// other (applyRaylineARCReasoningIssuer, atlasfutures/semantic-router#109).
func raylineARCPolicyActionsCarriable(cfg *config.RouterConfig, decision *config.Decision) bool {
	policy := decision.Algorithm.RaylineARC.PolicyService
	for _, binding := range policy.Bindings {
		// A v5 control is rendered from the registry in each format it
		// knows: Messages, Chat and Responses. Whether the worker's cells
		// admit it is checked at load (validateRaylineARCPolicyPackageV5Dispatch).
		if policy.IsPackageV5() {
			for _, accepted := range cfg.GetModelAcceptedFormats(binding.Worker) {
				format, err := wireFormatForModel(accepted)
				if err != nil || registryFormatOf(format) == "" {
					return false
				}
			}
			continue
		}
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
		// The format is chosen per request from the worker's accepted
		// formats, so the action must be carriable in every one of them.
		for _, accepted := range cfg.GetModelAcceptedFormats(binding.Worker) {
			if !policyActionCarriableIn(cfg, binding, accepted, profile) {
				return false
			}
		}
	}
	return true
}

func policyActionCarriableIn(
	cfg *config.RouterConfig,
	binding config.RaylineARCPolicyBinding,
	apiFormat string,
	profile *config.ProviderProfile,
) bool {
	format, err := wireFormatForModel(apiFormat)
	if err != nil {
		return false
	}
	switch format {
	case llmprotocol.AnthropicMessagesV1:
		return true
	case llmprotocol.OpenAIChatV1:
		_, _, err := policyActionChatWire(binding, resolveProviderReasoningTransport(profile))
		return err == nil
	case llmprotocol.OpenAIResponsesV1:
		return policyActionResponsesCarriable(cfg, binding)
	default:
		return false
	}
}
