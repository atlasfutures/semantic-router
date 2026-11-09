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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Reasons a lookup's encode is predicted full; "extends" when it is predicted
// to append. Bounded, so they can be logged and joined on.
const (
	raylineRoutesPredictAppends          = raylinearc.PolicyTransitionExtends
	raylineRoutesPredictArtifactMode     = "artifact_mode"
	raylineRoutesPredictRelaxed          = "relaxed_episodes"
	raylineRoutesPredictEphemeralTurn    = "ephemeral_turn"
	raylineRoutesPredictSideCall         = raylinearc.PolicyTransitionSideCall
	raylineRoutesPredictNoCommittedTurn  = "no_committed_turn"
	raylineRoutesPredictUnreadable       = "unreadable_request"
	raylineRoutesPredictShellChanged     = "shell_changed"
	raylineRoutesPredictNotRetained      = "not_retained"
	raylineRoutesPredictEphemeralRequest = "ephemeral"
)

// predictRaylineRoutesEncode says, from what the router holds once the
// episode is read and before it calls the policy service, whether a lookup's
// decision will append to the episode's retained encoder session or encode
// the transcript in full.
//
// It reads what the selector reads (selectViaPolicyService): the episode's
// committed policy state (PrefixLen and PrefixDigest, the request the last
// committed turn sent), the request's messages, and the turn signals. A
// request that extends the committed prefix is what the service appends to
// its retained session; anything else (no committed turn, a prefix break, a
// compaction, a side call) is a fresh or rebuilt session, which encodes
// every token.
//
// The committed prefix is in the service's session only if a strict decide
// left it there, so an append is predicted only when this process recorded
// that decide as the session's last write (policyRetainedSessions), under
// the system prompt and tools the request carries (the session covers them;
// the message digest does not). That rules out a prefix a zdr turn
// committed (decided on a throwaway service episode), a session a side call
// or another recipe's decide changed since, and a turn another router
// instance decided, which this process cannot see: each is gated.
//
// It is still a prediction. The service may have evicted the session (an
// idle TTL, a restart, a replica change) and rebuild it anyway; the
// decision's session_action says what actually happened, and is logged
// beside the admission outcome so an exempt lookup that rebuilt is counted.
func predictRaylineRoutesEncode(
	algorithm *config.AlgorithmConfig,
	arcContext *selection.RaylineARCSelectionContext,
	ephemeral bool,
) routerruntime.RouteEncodePrediction {
	full := func(reason string) routerruntime.RouteEncodePrediction {
		return routerruntime.RouteEncodePrediction{Reason: reason}
	}
	switch {
	case ephemeral:
		return full(raylineRoutesPredictEphemeralRequest)
	case algorithm == nil || algorithm.RaylineARC == nil || algorithm.RaylineARC.PolicyService == nil:
		return full(raylineRoutesPredictArtifactMode)
	case algorithm.RaylineARC.Episode.RelaxedConsistency():
		return full(raylineRoutesPredictRelaxed)
	case arcContext == nil:
		return full(raylineRoutesPredictUnreadable)
	case arcContext.EphemeralTurn:
		// A zero-retention turn is never appended (rayline_arc_policy_private.go).
		return full(raylineRoutesPredictEphemeralTurn)
	case arcContext.PolicyCallKind == raylinearc.PolicyCallSide:
		return full(raylineRoutesPredictSideCall)
	}
	state := arcContext.State
	if state == nil || state.Policy == nil || state.Policy.PrefixLen == 0 {
		return full(raylineRoutesPredictNoCommittedTurn)
	}
	clientRequest, messages, roles, err := policyClientRequestOf(arcContext)
	if err != nil {
		return full(raylineRoutesPredictUnreadable)
	}
	signals := raylineARCPolicyTurnSignals(
		arcContext.PolicySignalHeaders, arcContext.RequestFormat, clientRequest, messages, arcContext.EpisodeIDHash,
	)
	if arcContext.PolicyCallKind != "" && arcContext.PolicyCallKind != raylinearc.PolicyCallUnknown {
		signals.CallKind = arcContext.PolicyCallKind
	}
	digests := raylinearc.NewPrefixDigests(messages)
	_, _, transition := raylinearc.PolicyTurnFrom(state.Policy, digests, roles, state.TurnIndex, signals)
	if transition != raylinearc.PolicyTransitionExtends {
		return full(transition)
	}
	prediction := routerruntime.RouteEncodePrediction{
		SuffixTokens: raylineRoutesSuffixTokens(messages[state.Policy.PrefixLen:]),
		Reason:       raylineRoutesPredictShellChanged,
	}
	system := clientRequest.System
	if clientRequest.Instructions != nil {
		system, _ = json.Marshal(*clientRequest.Instructions)
	}
	current := raylinearc.NewPolicyPrefixShapeFrom(digests, system, clientRequest.Tools)
	held, shellChanged := policyRetainedSessions.holds(
		policyRetainedScope(algorithm.RaylineARC), arcContext.EpisodeIDHash, state.Policy, current,
	)
	switch {
	case shellChanged:
		return prediction
	case !held:
		prediction.Reason = raylineRoutesPredictNotRetained
		return prediction
	}
	prediction.Append, prediction.Reason = true, raylineRoutesPredictAppends
	return prediction
}

// raylineRoutesSuffixTokens estimates the tokens of the messages an append
// encodes, at the character heuristic's four bytes a token over their JSON.
// The JSON's own punctuation and any base64 media count too, so the estimate
// runs high for ordinary text, which errs toward gating.
func raylineRoutesSuffixTokens(suffix []json.RawMessage) int {
	bytes := 0
	for _, message := range suffix {
		bytes += len(message)
	}
	return (bytes + classification.CharactersPerToken - 1) / classification.CharactersPerToken
}
