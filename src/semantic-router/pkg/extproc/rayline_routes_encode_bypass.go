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
	"so the policy did not score it; the route is the policy's fallback"

// Outcomes of a lookup over the encode limit.
const (
	raylineRoutesEncodeBypassFallback = "fallback"
	// No fallback action to serve, or routing controls whose result points
	// only the policy can answer: the lookup is encoded.
	raylineRoutesEncodeBypassEncoded = "encoded"
)

// policyEncodeSizeBypass answers a throwaway POST /v1/routes lookup whose
// context is over routes_api.max_encode_tokens without calling the policy
// service. Such a lookup joins no conversation, so its encode is always a
// full one: it holds a policy container for the whole transcript (about a
// second per 35k tokens), and live turns on that container wait behind it.
// It is answered with the policy package's fallback action instead; nothing
// is encoded, the policy scores nothing and no episode state is committed.
//
// The decision service sets the limit (RoutesEncodeLimit) on throwaway
// lookups only. A tracked lookup is never bounded: it appends to its
// conversation's retained session as a routed turn does, and how much of it
// the service re-encodes is the service's call. The estimate is the one every
// limit on this path reads (ContextTokens, the router's conservative byte
// heuristic of the request). A lookup carrying routing controls, or one on a
// cell whose package names no fallback it binds, is encoded as before: nil.
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
	fallback := scorer.fallbackActionID
	if arcContext.RouteControls != nil || fallback == "" || !slices.Contains(available, fallback) {
		record(raylineRoutesEncodeBypassEncoded, "")
		return nil
	}
	result := selector.policyUndecidedResult(
		armed, selCtx, arcContext, state, scorer, &raylinearc.EncoderResult{}, fallback, "", false,
		turn, messages, sideCall, atBoundary, retained, workerIDs,
	)
	if result == nil {
		record(raylineRoutesEncodeBypassEncoded, "")
		return nil
	}
	// Nothing was measured: the counts are unknown, not zero.
	result.RaylineARC.TokenCountsUnknown = true
	result.RaylineARC.OfferKind = raylineRoutesEncodeBypassOfferKind
	result.Reasoning = "policy-service ARC decision (" + raylineRoutesEncodeBypassOfferKind + ", " + raylineRoutesEncodeBypassFallback + ")"
	record(raylineRoutesEncodeBypassFallback, fallback)
	return result
}
