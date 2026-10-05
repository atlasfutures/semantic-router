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
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// policyCapacityHold is the turn served when the policy service refuses a
// context its encoder cannot hold (pathfinder#3329, the serving half of
// pathfinder#3279): the episode keeps the action it last served, as
// pathfinder's in-process learned session does. No switch is made and no
// prompt cache is forfeited. The policy scored nothing, so the decision
// carries no scores, only the held arm; the turn commits like any held turn,
// and the next one asks the service again.
//
// It holds only the latest ledger action on the episode's held arm, and only
// when that action is still offered. With nothing held (an episode's first
// call, or the first after a compaction emptied the ledger) it returns nil and
// the turn fails closed as before: the package fallback is not known to this
// router for v4 packages.
func (selector *raylineARCSelector) policyCapacityHold(
	armed *raylineARCArmedComponents,
	selCtx *selection.SelectionContext,
	arcContext *selection.RaylineARCSelectionContext,
	state *raylinearc.EpisodeState,
	scorer *policyServiceScorer,
	refusal *raylinearc.PolicyServiceError,
	turn *raylinearc.PolicyEpisodeState,
	messages []json.RawMessage,
	sideCall bool,
	atBoundary bool,
	retained bool,
	available []string,
	workerIDs []string,
	latency time.Duration,
) *selection.SelectionResult {
	// The ledger is the current context's, as PolicyTurn left it: a request
	// that starts a compaction or prefix-break epoch has an empty one, so
	// nothing from the previous context is held.
	if state.PreviousArm == nil || turn == nil {
		return nil
	}
	var held *raylinearc.PolicyLedgerEntry
	for index := len(turn.Ledger) - 1; index >= 0; index-- {
		entry := turn.Ledger[index]
		if binding, ok := scorer.bindings[entry.ActionID]; ok && binding.arm == *state.PreviousArm {
			held = &entry
			break
		}
	}
	if held == nil || !slices.Contains(available, held.ActionID) {
		return nil
	}
	binding := scorer.bindings[held.ActionID]
	count := len(workerIDs)
	excludedArms := make([]bool, count)
	for arm := range excludedArms {
		excludedArms[arm] = arm != binding.arm
	}
	decision := raylinearc.Decision{
		SelectedArm:                 binding.arm,
		SelectedWorker:              workerIDs[binding.arm],
		RawScores:                   make([]float32, count),
		AdjustedScores:              make([]float32, count),
		SwitchCostUSD:               make([]float64, count),
		CacheMissTokens:             make([]int, count),
		ColdSwitchUpgradeExemptions: make([]bool, count),
		Stayed:                      true,
		ExcludedArms:                excludedArms,
	}
	if !validARCDecision(decision, workerIDs) {
		return nil
	}
	tokens, _ := refusal.Detail["token_count"].(int)
	maxTokens, _ := refusal.Detail["max_tokens"].(int)
	encoded := &raylinearc.EncoderResult{SerializedTokens: tokens, FullHistoryTokens: tokens}
	result := selector.selectionResult(armed, selCtx, arcContext, state, encoded, decision, 0)
	// The policy scored nothing: the trace and selection log carry no scores,
	// rather than all-zero ones a consumer would read as a real result.
	result.RaylineARC.RawScores, result.RaylineARC.AdjustedScores = nil, nil
	result.RaylineARC.PolicyLatency = latency
	result.RaylineARC.EncoderLatencyUnknown = true
	result.Reasoning = "policy-service ARC decision (encoder_capacity_hold)"
	result.RaylineARC.PolicyActionID = held.ActionID
	result.RaylineARC.PolicyArmID = held.ArmID
	result.RaylineARC.ThinkingLevel = binding.level
	result.RaylineARC.PolicyActionModel = binding.model
	result.RaylineARC.WorkerProviderModel = scorer.workers[binding.arm].Model
	result.RaylineARC.WorkerRoute = scorer.routes[binding.arm]
	result.RaylineARC.PolicySideCall = sideCall
	if !sideCall {
		result.RaylineARC.PolicyTurnState = turn.Clone()
		result.RaylineARC.PolicyNextState = turn.Next(messages, held.ActionID, held.ArmID)
	}
	if atBoundary && !retained {
		result.RaylineARC.PolicyBoundary = raylinearc.NewPolicyBoundaryDecision(binding.arm, state.TurnIndex, turn, messages)
	}
	logging.ComponentEvent("extproc", "rayline_arc_encoder_capacity_hold", map[string]interface{}{
		"request_id": arcContext.RequestID, "episode_id_hash": arcContext.EpisodeIDHash, "rule": "hold", "action_id": held.ActionID,
		"token_count": tokens, "max_tokens": maxTokens, "side_call": sideCall,
	})
	return result
}
