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

// policyHoldOfferKinds are the policy-service refusals a held action serves
// through, each with the offer kind its turn records: a context the encoder
// cannot hold, and media the projection cannot read (pathfinder#5281), which
// would otherwise fail every later turn of its conversation.
var policyHoldOfferKinds = map[string]string{
	raylinearc.PolicyCapacityRefusalClass: "capacity_hold",
	raylinearc.PolicyMediaRefusedClass:    "media_hold",
}

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
	held := policyHeldLedgerAction(state, turn, scorer, available)
	if held == nil {
		return nil
	}
	tokens, tokensKnown := refusal.Detail["token_count"].(int)
	maxTokens, _ := refusal.Detail["max_tokens"].(int)
	encoded := &raylinearc.EncoderResult{SerializedTokens: tokens, FullHistoryTokens: tokens}
	result := selector.policyUndecidedResult(
		armed, selCtx, arcContext, state, scorer, encoded, held.ActionID, held.ArmID, true,
		turn, messages, sideCall, atBoundary, retained, workerIDs,
	)
	if result == nil {
		return nil
	}
	result.RaylineARC.PolicyLatency = latency
	result.RaylineARC.PolicyHTTP = refusal.HTTPTiming
	// A runtime that refuses before counting reports no token count; the
	// counts are unknown, not zero.
	result.RaylineARC.TokenCountsUnknown = !tokensKnown
	media := refusal.Class == raylinearc.PolicyMediaRefusedClass
	result.Reasoning = "policy-service ARC decision (encoder_capacity_hold)"
	if media {
		result.Reasoning = "policy-service ARC decision (media_hold)"
	}
	if media {
		logging.ComponentEvent("extproc", "rayline_arc_media_hold", map[string]interface{}{
			"request_id": arcContext.RequestID, "episode_id_hash": arcContext.EpisodeIDHash, "rule": "hold", "action_id": held.ActionID,
			"sub_reason": refusal.Detail["sub_reason"], "side_call": sideCall,
		})
		return result
	}
	logging.ComponentEvent("extproc", "rayline_arc_encoder_capacity_hold", map[string]interface{}{
		"request_id": arcContext.RequestID, "episode_id_hash": arcContext.EpisodeIDHash, "rule": "hold", "action_id": held.ActionID,
		"token_count": knownCount(tokens, tokensKnown), "max_tokens": maxTokens, "side_call": sideCall,
	})
	return result
}

// policyHeldLedgerAction is the episode's held action: the latest ledger
// action on the episode's held arm, when that action is still offered. The
// ledger is the current context's, as PolicyTurn left it: a request that
// starts a compaction or prefix-break epoch has an empty one, so nothing from
// the previous context is held. Nil when nothing is.
func policyHeldLedgerAction(
	state *raylinearc.EpisodeState,
	turn *raylinearc.PolicyEpisodeState,
	scorer *policyServiceScorer,
	available []string,
) *raylinearc.PolicyLedgerEntry {
	if state.PreviousArm == nil || turn == nil {
		return nil
	}
	for index := len(turn.Ledger) - 1; index >= 0; index-- {
		entry := turn.Ledger[index]
		if binding, ok := scorer.bindings[entry.ActionID]; ok && binding.arm == *state.PreviousArm {
			if !slices.Contains(available, entry.ActionID) {
				return nil
			}
			return &entry
		}
	}
	return nil
}

// policyUndecidedResult is a turn served on actionID without a policy
// decision (a hold, or a lookup over its encode limit): the policy scored
// nothing, so the trace carries no scores, only the served arm. With ledger
// the turn commits like any held turn, recording actionID (attributed to
// armID) for its reply, while the state keeps where the service's encode
// last ended (NextUnencoded); without it the episode's policy state is left
// as it was. Nil when the action's arm is not a valid decision over
// workerIDs.
func (selector *raylineARCSelector) policyUndecidedResult(
	armed *raylineARCArmedComponents,
	selCtx *selection.SelectionContext,
	arcContext *selection.RaylineARCSelectionContext,
	state *raylinearc.EpisodeState,
	scorer *policyServiceScorer,
	encoded *raylinearc.EncoderResult,
	actionID string,
	armID string,
	ledger bool,
	turn *raylinearc.PolicyEpisodeState,
	messages []json.RawMessage,
	sideCall bool,
	atBoundary bool,
	retained bool,
	workerIDs []string,
) *selection.SelectionResult {
	binding, ok := scorer.bindings[actionID]
	if !ok {
		return nil
	}
	count := len(workerIDs)
	excludedArms := make([]bool, count)
	for arm := range excludedArms {
		excludedArms[arm] = arm != binding.arm
	}
	decision := raylinearc.Decision{
		SelectedArm:                 binding.arm,
		SelectedWorker:              "",
		RawScores:                   make([]float32, count),
		AdjustedScores:              make([]float32, count),
		SwitchCostUSD:               make([]float64, count),
		CacheMissTokens:             make([]int, count),
		ColdSwitchUpgradeExemptions: make([]bool, count),
		Stayed:                      state.PreviousArm != nil && *state.PreviousArm == binding.arm,
		ExcludedArms:                excludedArms,
	}
	if binding.arm >= 0 && binding.arm < count {
		decision.SelectedWorker = workerIDs[binding.arm]
	}
	if !validARCDecision(decision, workerIDs) {
		return nil
	}
	result := selector.selectionResult(armed, selCtx, arcContext, state, encoded, decision, 0)
	// The policy scored nothing: the trace and selection log carry no scores,
	// rather than all-zero ones a consumer would read as a real result.
	result.RaylineARC.RawScores, result.RaylineARC.AdjustedScores = nil, nil
	result.RaylineARC.EncoderLatencyUnknown = true
	result.RaylineARC.PolicyActionID = actionID
	result.RaylineARC.PolicyArmID = armID
	result.RaylineARC.ThinkingLevel = binding.level
	result.RaylineARC.PolicyActionModel = binding.model
	scorer.applyDispatchArm(result, selCtx, binding, arcContext.PrivateTurn)
	result.RaylineARC.PolicySideCall = sideCall
	if !sideCall {
		result.RaylineARC.PolicyTurnState = turn.Clone()
		if ledger {
			result.RaylineARC.PolicyNextState = turn.NextUnencoded(messages, actionID, armID)
		}
	}
	if atBoundary && !retained {
		result.RaylineARC.PolicyBoundary = raylinearc.NewPolicyBoundaryDecision(binding.arm, state.TurnIndex, turn, messages)
	}
	return result
}

// knownCount is a count for a log line: the value when it was measured, and
// null when it was not, never a zero that reads as one.
func knownCount(value int, known bool) any {
	if !known {
		return nil
	}
	return value
}
