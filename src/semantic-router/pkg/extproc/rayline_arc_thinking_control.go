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

// applyRaylineARCThinkingControl renders a v5 action's control onto the
// provider-bound body from the compiled registry (ADR 0109): the cell's base
// wire replaces every thinking field, and the instruction is placed by
// turn_tail_v2, written on change and replayed from the episode's ledger.
// It returns false outside a v5 turn. A control the worker's cell does not
// admit, or a body the placer cannot govern, fails the turn: serving it
// without its control would serve another policy.
func applyRaylineARCThinkingControl(
	body []byte,
	dispatch *providerDispatch,
	ctx *RequestContext,
	cfg *config.RouterConfig,
) ([]byte, bool, error) {
	policy, action, v5, err := raylineARCPolicyControlAction(ctx)
	if !v5 || err != nil {
		return nil, v5, err
	}
	if ctx.RaylineARCDispatch == nil {
		return nil, true, errThinkingControlUnbound
	}
	worker := ctx.RaylineARCDispatch.ID
	provider, format, err := config.RaylineARCRegistryCell(cfg, worker)
	if err != nil {
		return nil, true, err
	}
	if format != registryFormatOf(dispatch.targetFormat) {
		return nil, true, fmt.Errorf("worker %q dispatches %s, and its registry cell is %s", worker, dispatch.targetFormat, format)
	}
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		return nil, true, err
	}
	cell, err := registry.Admit(action.Model, provider, format, action.Control, policy.AllowExperimentalControls)
	if err != nil {
		return nil, true, err
	}
	key := controlPlacementKey(worker, format, action.Control)
	committed, found, hasEpisode := ctx.RaylineARCTransaction.committedControlPlacement(key)
	if !hasEpisode {
		return nil, true, errThinkingControlEpisode
	}
	placer, err := thinkingcontrol.NewPlacer(format)
	if found {
		placer, err = thinkingcontrol.ResumePlacer(committed)
	}
	if err != nil {
		return nil, true, err
	}
	var head struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &head); err != nil || head.Model == "" {
		return nil, true, fmt.Errorf("the provider-bound body names no model")
	}
	rendered, receipt, err := placer.Render(body, &action.Control, cell, head.Model)
	if err != nil {
		return nil, true, err
	}
	ctx.RaylineARCTransaction.stageControlPlacement(raylinearc.ControlPlacement{Key: key, State: placer.State()})
	ctx.RaylineARCThinking = thinkingControlTrace(cell, receipt)
	recordThinkingLeverTurn(ctx.RaylineARCThinking)
	return rendered, true, nil
}

// thinkingControlTrace attributes the call with the control in force and
// what the model can see (ADR 0109 decision 4). The placer never declines a
// control it renders, so the level in force is the one the policy chose.
func thinkingControlTrace(cell *thinkingcontrol.Cell, receipt thinkingcontrol.Receipt) *raylineARCThinkingTrace {
	trace := &raylineARCThinkingTrace{
		Lever:          thinkingControlLever,
		Source:         config.RaylineARCThinkingSourcePolicy,
		Admission:      cell.Instruction,
		ControlInForce: receipt.ControlID,
		Propensity:     1,
		Retry:          receipt.Retry,
		Epoch:          uint32(receipt.Epoch),
	}
	if receipt.LevelInForce != nil {
		trace.LevelRequested, trace.LevelInForce = *receipt.LevelInForce, *receipt.LevelInForce
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
