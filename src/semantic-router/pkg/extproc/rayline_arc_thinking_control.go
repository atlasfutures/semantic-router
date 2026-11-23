/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/tidwall/sjson"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// thinkingControlLever names the registry renderer on the routing record and
// the lever metric.
const thinkingControlLever = "thinking_control_v1"

var (
	errThinkingControlUnbound = errors.New("the policy decision names an action the v5 package does not hold")
	errThinkingControlEpisode = errors.New("a v5 control is placed on the episode's transcript, and the turn has no episode")
)

// plannedThinkingControl is a v5 action's control as route construction
// resolved it for one dispatch: admitted on the dispatch's (model, provider,
// format) cell, with the episode's placer for the worker and control shape.
// The provider boundary only renders it.
type plannedThinkingControl struct {
	key     string
	control thinkingcontrol.Control
	cell    *thinkingcontrol.Cell
	placer  *thinkingcontrol.Placer
	// retire is the worker whose placements in other scopes this one
	// replaces ("" keeps them): set when every endpoint of the worker is in
	// this placement's scope.
	retire string
	// effortOverride is the effort dispatch_effort_overrides sends on top of
	// the control's native default base, at effortPath; empty sends the
	// base's own wire.
	effortOverride string
	effortPath     string
}

// raylineARCDispatchedControl is the control and worker this turn's v5 action
// is dispatched with: under a dispatch_model_overrides entry for its bound
// worker, the to-worker and the native default control (no steering);
// otherwise the bound worker and the action's own control.
func raylineARCDispatchedControl(
	policy *config.RaylineARCPolicyServiceConfig, actionID string, control thinkingcontrol.Control,
) (thinkingcontrol.Control, string, bool) {
	binding, ok := policy.PolicyBinding(actionID)
	if !ok {
		return control, "", false
	}
	worker, remapped := policy.DispatchWorker(binding.Worker)
	if remapped {
		return thinkingcontrol.Control{Native: "default"}, worker, true
	}
	return control, worker, false
}

// raylineARCDispatchEffortOverride is the effort dispatch_effort_overrides
// sends for this turn's v5 action: the worker it is dispatched on has an
// entry and the control it is dispatched with is the native default base
// with no budget.
func raylineARCDispatchEffortOverride(ctx *RequestContext) (string, bool) {
	policy, action, v5, err := raylineARCPolicyControlAction(ctx)
	if !v5 || err != nil || len(policy.DispatchEffortOverrides) == 0 {
		return "", false
	}
	control, worker, _ := raylineARCDispatchedControl(policy, ctx.VSRRaylineARC.PolicyActionID, action.Control)
	if worker == "" || !config.RaylineARCOverridableBase(control) {
		return "", false
	}
	return policy.DispatchEffortOverride(worker)
}

// raylineARCDispatchModelRemap is the worker a dispatch_model_overrides entry
// sends this turn's v5 action to, if any.
func raylineARCDispatchModelRemap(ctx *RequestContext) (string, bool) {
	policy, action, v5, err := raylineARCPolicyControlAction(ctx)
	if !v5 || err != nil || len(policy.DispatchModelOverrides) == 0 {
		return "", false
	}
	_, worker, remapped := raylineARCDispatchedControl(policy, ctx.VSRRaylineARC.PolicyActionID, action.Control)
	return worker, remapped
}

// dispatchEffortOverridePath is where the override effort goes on a body of
// format bound for the provider's reasoning transport: Messages as
// output_config.effort (other output_config keys stay), Responses as
// reasoning.effort, and Chat as OpenRouter's reasoning.effort or a top-level
// reasoning_effort. Any other Chat transport has no effort field, and the turn
// fails rather than report an effort it did not send.
func dispatchEffortOverridePath(format string, transport modelcatalog.ReasoningTransport) (string, error) {
	switch {
	case format == thinkingcontrol.FormatMessages:
		return "output_config.effort", nil
	case format == thinkingcontrol.FormatResponses:
		return "reasoning.effort", nil
	case usesReasoningObjectTransport(transport):
		return "reasoning.effort", nil
	case usesTopLevelReasoningEffort(transport):
		return "reasoning_effort", nil
	default:
		return "", fmt.Errorf("dispatch_effort_overrides: the worker's Chat transport %q carries no effort", transport)
	}
}

// applyDispatchEffortOverride writes the override effort at path on the
// rendered body.
func applyDispatchEffortOverride(body []byte, path, effort string) ([]byte, error) {
	out, err := sjson.SetBytes(body, path, effort)
	if err != nil {
		return nil, fmt.Errorf("failed to write the dispatch effort override: %w", err)
	}
	return out, nil
}

// raylineARCPolicyControlAction is the v5 action this turn's policy decision
// chose. v5 is false outside a v5 package; an action id the package does not
// hold is an error.
func raylineARCPolicyControlAction(
	ctx *RequestContext,
) (policy *config.RaylineARCPolicyServiceConfig, action config.RaylineARCPolicyActionV5, v5 bool, err error) {
	if ctx == nil || ctx.VSRRaylineARC == nil || ctx.VSRRaylineARC.PolicyActionID == "" {
		return nil, action, false, nil
	}
	decision := ctx.VSRSelectedDecision
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.RaylineARC == nil {
		return nil, action, false, nil
	}
	policy = decision.Algorithm.RaylineARC.PolicyService
	if !policy.IsPackageV5() {
		return nil, action, false, nil
	}
	action, ok := policy.PackageV5Action(ctx.VSRRaylineARC.PolicyActionID)
	if !ok {
		return policy, action, true, errThinkingControlUnbound
	}
	return policy, action, true, nil
}

// registryFormatOf is the registry's name for a wire format.
func registryFormatOf(format llmprotocol.WireFormat) string {
	switch format {
	case llmprotocol.AnthropicMessagesV1:
		return thinkingcontrol.FormatMessages
	case llmprotocol.OpenAIChatV1:
		return thinkingcontrol.FormatChat
	case llmprotocol.OpenAIResponsesV1:
		return thinkingcontrol.FormatResponses
	default:
		return ""
	}
}

// planRaylineARCThinkingControl resolves this dispatch's v5 control: the
// cell of the request's target format must admit it, and the episode's
// placer is resumed. It returns nil outside a v5 turn. A budgeted base's
// Messages allowance is raised after request_params
// (prepareProviderRequest).
func planRaylineARCThinkingControl(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
	cfg *config.RouterConfig,
) (*plannedThinkingControl, bool, error) {
	policy, action, v5, err := raylineARCPolicyControlAction(ctx)
	if !v5 || err != nil {
		return nil, false, err
	}
	if ctx.RaylineARCDispatch == nil {
		return nil, false, errThinkingControlUnbound
	}
	worker := ctx.RaylineARCDispatch.ID
	control, dispatched, _ := raylineARCDispatchedControl(policy, ctx.VSRRaylineARC.PolicyActionID, action.Control)
	if dispatched != "" && dispatched != worker {
		return nil, false, fmt.Errorf("the action dispatches on worker %q, and the turn is on %q", dispatched, worker)
	}
	action.Control = control
	provider, err := config.RaylineARCRegistryProvider(cfg, worker)
	if err != nil {
		return nil, false, err
	}
	served, err := config.RaylineARCRegistryModel(cfg, worker)
	if err != nil {
		return nil, false, err
	}
	format := registryFormatOf(dispatch.targetFormat)
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		return nil, false, err
	}
	cell, err := registry.Admit(served, provider, format, action.Control, policy.AllowExperimentalControls)
	if err != nil {
		return nil, false, err
	}
	key := raylineARCPlacementKey(worker, format, dispatch.profile)
	committed, found, hasEpisode := ctx.RaylineARCTransaction.committedControlPlacement(key)
	if !hasEpisode {
		return nil, false, errThinkingControlEpisode
	}
	placer, err := thinkingcontrol.NewPlacer(format)
	if found {
		placer, err = thinkingcontrol.ResumePlacer(committed)
	}
	if err != nil {
		return nil, false, err
	}
	planned := &plannedThinkingControl{key: key, control: action.Control, cell: cell, placer: placer}
	if scope := raylineARCPlacementScope(dispatch.profile); scope != "" && raylineARCWorkerPlacementScope(cfg, worker) == scope {
		planned.retire = worker
	}
	if config.RaylineARCOverridableBase(action.Control) {
		if effort, ok := policy.DispatchEffortOverride(worker); ok {
			path, err := dispatchEffortOverridePath(format, resolveProviderReasoningTransport(dispatch.profile))
			if err != nil {
				return nil, false, err
			}
			planned.effortOverride, planned.effortPath = effort, path
		}
	}
	return planned, false, nil
}

// render places the planned control on the provider-bound body and stages
// the placer for the turn's commit. A body the placer cannot govern fails
// the turn: serving it without its control would serve another policy.
func (planned *plannedThinkingControl) render(body []byte, ctx *RequestContext) ([]byte, error) {
	var head struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &head); err != nil || head.Model == "" {
		return nil, fmt.Errorf("the provider-bound body names no model")
	}
	rendered, receipt, err := planned.placer.Render(body, &planned.control, planned.cell, head.Model)
	if err != nil {
		return nil, err
	}
	if planned.effortOverride != "" {
		if rendered, err = applyDispatchEffortOverride(rendered, planned.effortPath, planned.effortOverride); err != nil {
			return nil, err
		}
	}
	ctx.RaylineARCTransaction.stageControlPlacement(raylinearc.ControlPlacement{Key: planned.key, State: planned.placer.State()}, planned.retire)
	ctx.RaylineARCThinking = thinkingControlTrace(planned.cell, planned.control, receipt)
	recordThinkingLeverTurn(ctx.RaylineARCThinking)
	return rendered, nil
}

// thinkingControlTrace attributes the call with the drawn control and what
// the model can see (ADR 0109 decision 4). The requested level is the
// policy's; the level in force is the receipt's, which a refused write
// (ADR 0129) leaves at the earlier one.
func thinkingControlTrace(
	cell *thinkingcontrol.Cell, control thinkingcontrol.Control, receipt thinkingcontrol.Receipt,
) *raylineARCThinkingTrace {
	trace := &raylineARCThinkingTrace{
		Lever:          thinkingControlLever,
		Source:         config.RaylineARCThinkingSourcePolicy,
		Admission:      cell.Instruction,
		ControlInForce: receipt.ControlID,
		Propensity:     1,
		Retry:          receipt.Retry,
		Epoch:          uint32(receipt.Epoch),
	}
	if control.Instruction != nil {
		trace.LevelRequested = control.Instruction.Level
	}
	if receipt.LevelInForce != nil {
		trace.LevelInForce = *receipt.LevelInForce
	}
	if receipt.Refused != nil {
		// A refused call stays attributed to the drawn control: the label is
		// the policy's action. What the provider saw is the level in force
		// beside the refusal (pathfinder ADR 0129's receipt).
		trace.Refused = *receipt.Refused
	}
	if receipt.InstructionState != nil {
		trace.InstructionState = *receipt.InstructionState
	}
	if receipt.Written != nil {
		trace.Written, trace.Emitted = *receipt.Written, true
	}
	if receipt.EpochResetReason != nil {
		trace.ResetReason = *receipt.EpochResetReason
	}
	return trace
}

// raylineARCPlacementKey names the worker's placer for a turn. A dispatch to
// a managed cloud (Vertex AI or Amazon Bedrock) has a placer of its own: its registry cell
// admits other controls than the endpoint the worker served before, so a
// worker moved there under the same name, format and episode store starts a
// new transcript instead of replaying instructions the cloud's cell refuses.
// Every other dispatch keeps the plain worker|format key.
func raylineARCPlacementKey(worker, format string, profile *config.ProviderProfile) string {
	if scope := raylineARCPlacementScope(profile); scope != "" {
		return thinkingcontrol.PlacementKey(worker+"|"+scope, format)
	}
	return thinkingcontrol.PlacementKey(worker, format)
}

// raylineARCPlacementScope is the scope a dispatch's placer key carries:
// "vertex-ai" on Vertex AI, "bedrock" on Amazon Bedrock, "" elsewhere.
func raylineARCPlacementScope(profile *config.ProviderProfile) string {
	switch {
	case providerIsVertexAI(profile):
		return "vertex-ai"
	case providerIsBedrock(profile):
		return "bedrock"
	}
	return ""
}

// raylineARCWorkerPlacementScope is the placement scope of a worker's
// endpoints ("" when they differ or cannot be read: such a worker is not a
// managed-cloud-only one). Config load refuses a model whose backends are of
// different providers (TestVertexBackendIsNotMixedWithOthers), so a worker's
// endpoints share one scope.
func raylineARCWorkerPlacementScope(cfg *config.RouterConfig, worker string) string {
	endpoints := cfg.GetEndpointsForModel(worker)
	scope := ""
	for index, endpoint := range endpoints {
		profile, err := cfg.GetProviderProfileForEndpoint(endpoint.Name)
		if err != nil {
			return ""
		}
		this := raylineARCPlacementScope(profile)
		if index > 0 && this != scope {
			return ""
		}
		scope = this
	}
	return scope
}

// raylineARCPlacementKeyParts splits a placer key into its worker and scope.
// Only a scope raylineARCPlacementScope names counts as one: an older key's
// worker|format|shape... segments are an unscoped placement.
func raylineARCPlacementKeyParts(key string) (worker, scope string, ok bool) {
	worker, rest, ok := strings.Cut(key, "|")
	if !ok {
		return "", "", false
	}
	if second, _, _ := strings.Cut(rest, "|"); second == "vertex-ai" || second == "bedrock" {
		return worker, second, true
	}
	return worker, "", true
}

// raylineARCRetirePlacements drops worker's placements in scopes other than
// scope: a worker whose every endpoint is now in scope never resumes them,
// and they would only push other workers' placements out of the bounded
// record.
func raylineARCRetirePlacements(placements []raylinearc.ControlPlacement, worker, scope string) []raylinearc.ControlPlacement {
	kept := placements[:0:0]
	for _, placement := range placements {
		if owner, other, ok := raylineARCPlacementKeyParts(placement.Key); ok && owner == worker && other != scope {
			continue
		}
		kept = append(kept, placement)
	}
	return kept
}
