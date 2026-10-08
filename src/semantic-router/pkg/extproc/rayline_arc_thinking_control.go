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
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkingcontrol"
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

// controlPlacementKey names a placer: one worker's transcript in one format
// under one control shape. Pathfinder fixes base, budget and lever presence
// per episode, so a shape change on a worker starts its own placer.
func controlPlacementKey(worker, format string, control thinkingcontrol.Control) string {
	budget := "-"
	if control.BudgetTokens != nil {
		budget = strconv.FormatInt(*control.BudgetTokens, 10)
	}
	return fmt.Sprintf("%s|%s|%s|%s|%t", worker, format, control.Native, budget, control.Instruction != nil)
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
	key := controlPlacementKey(worker, format, action.Control)
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
	return &plannedThinkingControl{key: key, control: action.Control, cell: cell, placer: placer}, false, nil
}

// raiseMessagesAllowance keeps the caller's output allowance on top of a
// thinking budget: Messages counts thinking inside max_tokens and refuses a
// budget that leaves no room below it.
func raiseMessagesAllowance(request *llmprotocol.Request, budget *int64) bool {
	if budget == nil {
		return false
	}
	// The dispatch output bound has already set a limit on a Messages
	// request; the fallback only keeps a direct caller valid.
	allowance := dispatchFallbackMaxOutputTokens
	if request.Sampling.MaxOutputTokens != nil {
		allowance = *request.Sampling.MaxOutputTokens
	}
	if allowance > *budget {
		return false
	}
	raised := *budget + allowance
	request.Sampling.MaxOutputTokens = &raised
	return true
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
	ctx.RaylineARCTransaction.stageControlPlacement(raylinearc.ControlPlacement{Key: planned.key, State: planned.placer.State()})
	ctx.RaylineARCThinking = thinkingControlTrace(planned.cell, planned.control, receipt)
	recordThinkingLeverTurn(ctx.RaylineARCThinking)
	return rendered, nil
}

// thinkingControlTrace attributes the call with the control in force and
// what the model can see (ADR 0109 decision 4). The requested level is the
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
		// The drawn control never reached the provider; attribute the call
		// to the one that did, or to none rather than to the drawn one.
		trace.Refused, trace.ControlInForce = *receipt.Refused, ""
		if registry, err := thinkingcontrol.Embedded(); err == nil && receipt.LevelInForce != nil {
			trace.ControlInForce, _ = registry.InForceControl(cell, control, *receipt.LevelInForce)
		}
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
