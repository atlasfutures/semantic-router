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
// The bound is on what the encode would cost (policyLookupEncodeTokens): a
// tracked lookup that extends its context is encoded as a suffix on the
// service's retained session, so only the messages since the service's last
// encode count; any other lookup is encoded whole. A lookup carrying routing
// controls, or one with neither a held nor a fallback action to serve, is
// encoded as before: nil.
//
// A held answer commits like a capacity hold: the ledger records the held
// action for this turn's reply, as it would for any held turn, and the
// service's session is untouched: the state keeps where the service's encode
// ended, so the next lookup's suffix spans this one's messages too, and the
// next decide sends the whole request, never a delta on the ledger's prefix
// (policyDeltaFor needs a staged turn of the committed turn index, which a
// bypass advances past). A fallback answer records no ledger entry
// (the router does not know the arm the service would attribute it to), so
// the episode's policy state stays as it was.
func (selector *raylineARCSelector) policyEncodeSizeBypass(
	armed *raylineARCArmedComponents,
	selCtx *selection.SelectionContext,
	arcContext *selection.RaylineARCSelectionContext,
	state *raylinearc.EpisodeState,
	scorer *policyServiceScorer,
	turn *raylinearc.PolicyEpisodeState,
	transition string,
	shell string,
	messages []json.RawMessage,
	sideCall bool,
	atBoundary bool,
	retained bool,
	available []string,
	workerIDs []string,
) *selection.SelectionResult {
	limit := arcContext.RoutesEncodeLimit
	if !arcContext.RoutesLookup || limit <= 0 {
		return nil
	}
	relaxed := scorer.episodeMode == raylinearc.PolicyEpisodeModeRelaxed
	encodeTokens, span := policyLookupEncodeTokens(arcContext, turn, transition, relaxed, shell, messages)
	if encodeTokens <= limit {
		return nil
	}
	record := func(outcome, actionID string) {
		metrics.RecordRaylineARCRoutesEncodeBypass(outcome)
		logging.ComponentEvent("extproc", "rayline_routes_encode_bypass", map[string]interface{}{
			"request_id": arcContext.RequestID, "episode_id_hash": arcContext.EpisodeIDHash,
			"outcome": outcome, "action_id": actionID, "span": span, "encode_tokens": encodeTokens,
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

// Spans a lookup's encode estimate covers.
const (
	raylineRoutesEncodeSpanContext = "context"
	raylineRoutesEncodeSpanSuffix  = "suffix"
)

// policyLookupEncodeTokens estimates what this lookup's encode would cost,
// and the span that estimate covers. A tracked lookup on a strict cell that
// extends its context with the system prompt and tools the service last
// encoded (shell, against EncodedShell) appends to the service's retained
// session, so it costs the messages since the service's last encode
// (EncodedThrough, which a bypass or hold leaves behind): the request's own
// estimate, ContextTokens, scaled by those messages' share of the request's
// message bytes, rounded up. Every other lookup -- a stateless or ephemeral
// one, one on a relaxed cell (whose service keeps no session), a session's
// first, a compaction, prefix break or changed shell that rebuilds it, a
// side call -- is encoded whole and costs ContextTokens. A state no decided
// turn recorded a shell in is taken to have this one.
//
// The suffix is an estimate on the assumption that the service still holds
// the session: one it evicted is rebuilt in full.
func policyLookupEncodeTokens(
	arcContext *selection.RaylineARCSelectionContext,
	turn *raylinearc.PolicyEpisodeState,
	transition string,
	relaxed bool,
	shell string,
	messages []json.RawMessage,
) (int, string) {
	whole := arcContext.ContextTokens
	if arcContext.EphemeralTurn || relaxed || transition != raylinearc.PolicyTransitionExtends || turn == nil ||
		turn.EncodedShell != "" && turn.EncodedShell != shell {
		return whole, raylineRoutesEncodeSpanContext
	}
	from := turn.EncodedThrough()
	if from <= 0 || from > len(messages) {
		return whole, raylineRoutesEncodeSpanContext
	}
	total, suffix := 0, 0
	for index, message := range messages {
		total += len(message)
		if index >= from {
			suffix += len(message)
		}
	}
	if total == 0 {
		return whole, raylineRoutesEncodeSpanContext
	}
	return int((int64(whole)*int64(suffix) + int64(total) - 1) / int64(total)), raylineRoutesEncodeSpanSuffix
}
