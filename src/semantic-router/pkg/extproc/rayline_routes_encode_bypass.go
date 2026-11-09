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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// raylineRoutesEncodeBypassOfferKind is the offer kind of a lookup answered
// without an encode because its context is over routes_api.max_encode_tokens.
const raylineRoutesEncodeBypassOfferKind = "encode_size_bypass"

// raylineRoutesEncodeBypassWarning tells the caller its lookup was answered
// without scoring the conversation.
const raylineRoutesEncodeBypassWarning = "encode_size_bypass: the conversation is over this endpoint's encode limit, " +
	"so the policy did not score it; the route is the conversation's held model, or the policy's fallback"

// Outcomes of a lookup over the encode limit.
const (
	raylineRoutesEncodeBypassHeld     = "held"
	raylineRoutesEncodeBypassFallback = "fallback"
	// Neither held nor fallback action to serve, or routing controls whose
	// result points only the policy can answer: the lookup is encoded.
	raylineRoutesEncodeBypassEncoded = "encoded"
)

// policyEncodeSizeBypass answers a POST /v1/routes lookup whose context is
// over routes_api.max_encode_tokens without calling the policy service. A
// full encode holds a policy container for the whole transcript (about a
// second per 35k tokens), and live turns on that container wait behind it,
// so a lookup that large is answered the way a context the encoder refuses
// is (policyCapacityHold): with the episode's held action, or, with nothing
// held, the package's fallback action. Nothing is encoded and the policy
// scores nothing.
//
// The estimate is the one every limit on this path reads
// (RaylineARCSelectionContext.ContextTokens, the router's conservative byte
// heuristic of the request). A lookup carrying routing controls, or one with
// neither a held nor a fallback action to serve, is encoded as before: nil.
//
// A held answer commits like a capacity hold: the ledger records the held
// action for this turn's reply, as it would for any held turn, and the
// service's session is untouched. A fallback answer records no ledger entry
// (the router does not know the arm the service would attribute it to), so
// the episode's policy state stays as it was.
func (selector *raylineARCSelector) policyEncodeSizeBypass(
	armed *raylineARCArmedComponents,
	selCtx *selection.SelectionContext,
	arcContext *selection.RaylineARCSelectionContext,
	state *raylinearc.EpisodeState,
	scorer *policyServiceScorer,
	turn *raylinearc.PolicyEpisodeState,
	messages []json.RawMessage,
	sideCall bool,
	atBoundary bool,
	retained bool,
	available []string,
	workerIDs []string,
) *selection.SelectionResult {
	limit := arcContext.RoutesEncodeLimit
	if !arcContext.RoutesLookup || limit <= 0 || arcContext.ContextTokens <= limit {
		return nil
	}
	record := func(outcome, actionID string) {
		metrics.RecordRaylineARCRoutesEncodeBypass(outcome)
		logging.ComponentEvent("extproc", "rayline_routes_encode_bypass", map[string]interface{}{
			"request_id": arcContext.RequestID, "episode_id_hash": arcContext.EpisodeIDHash,
			"outcome": outcome, "action_id": actionID,
			"context_tokens": arcContext.ContextTokens, "max_encode_tokens": limit,
		})
	}
	if arcContext.RouteControls != nil {
		record(raylineRoutesEncodeBypassEncoded, "")
		return nil
	}
	outcome, actionID, armID, ledger := "", "", "", false
	if held := policyHeldLedgerAction(state, turn, scorer, available); held != nil {
		outcome, actionID, armID, ledger = raylineRoutesEncodeBypassHeld, held.ActionID, held.ArmID, true
	} else if fallback := scorer.fallbackActionID; fallback != "" && slices.Contains(available, fallback) {
		outcome, actionID = raylineRoutesEncodeBypassFallback, fallback
	}
	if outcome == "" {
		record(raylineRoutesEncodeBypassEncoded, "")
		return nil
	}
	encoded := &raylinearc.EncoderResult{}
	result := selector.policyUndecidedResult(
		armed, selCtx, arcContext, state, scorer, encoded, actionID, armID, ledger,
		turn, messages, sideCall, atBoundary, retained, workerIDs,
	)
	if result == nil {
		record(raylineRoutesEncodeBypassEncoded, "")
		return nil
	}
	// Nothing was measured: the counts are unknown, not zero.
	result.RaylineARC.TokenCountsUnknown = true
	result.RaylineARC.OfferKind = raylineRoutesEncodeBypassOfferKind
	result.Reasoning = "policy-service ARC decision (" + raylineRoutesEncodeBypassOfferKind + ", " + outcome + ")"
	record(outcome, actionID)
	return result
}
