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
	"errors"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The refusals of a Decisions lookup's routing.execution that need the
// serving package (pathfinder #5874, "Validation and refusals"): each is a
// selection failure class that the routes adapter answers with its public
// code, before any decision.
const (
	arcFailureExecutionUnknownAction    = "execution_unknown_action"
	arcFailureExecutionIdentityPackage  = "execution_identity_mismatch_package_sha256"
	arcFailureExecutionIdentityControls = "execution_identity_mismatch_thinking_controls_sha256"
	arcFailureExecutionIdentityOffer    = "execution_identity_mismatch_offer_sha256"
	arcFailureExecutionHoldConflict     = "execution_hold_conflict"
	arcFailureExecutionHoldUnavailable  = "execution_hold_unavailable"
)

// routeExecutionClasses are the execution refusals, by class: the public
// code, its status, and (for an identity mismatch) the expect member.
var routeExecutionClasses = map[string]struct {
	code   string
	status int
	member string
}{
	arcFailureExecutionUnknownAction:    {"execution_unknown_action", 400, ""},
	arcFailureExecutionIdentityPackage:  {"execution_identity_mismatch", 409, "package_sha256"},
	arcFailureExecutionIdentityControls: {"execution_identity_mismatch", 409, "thinking_controls_sha256"},
	arcFailureExecutionIdentityOffer:    {"execution_identity_mismatch", 409, "offer_sha256"},
	arcFailureExecutionHoldConflict:     {"execution_hold_conflict", 409, ""},
	arcFailureExecutionHoldUnavailable:  {"execution_hold_unavailable", 409, ""},
}

// routeIdentity is the identity a lookup decides under: the serving
// package, its thinking-controls registry, and the offer (the actions
// offered to this request, after binding health and exclusive).
func routeIdentity(scorer *policyServiceScorer, available []string) selection.RaylineARCRouteIdentity {
	return selection.RaylineARCRouteIdentity{
		PackageSHA256:          scorer.sha256,
		ThinkingControlsSHA256: scorer.thinkingControls,
		OfferSHA256:            routesOfferSHA256(scorer.sha256, scorer.offerOf(available)),
	}
}

// offerOf is the offer the service decides over: the available actions,
// narrowed to the operator's allow-list when the cell configures one.
func (scorer *policyServiceScorer) offerOf(available []string) []string {
	if len(scorer.offered) == 0 {
		return available
	}
	offer := make([]string, 0, len(available))
	for _, actionID := range available {
		if slices.Contains(scorer.offered, actionID) {
			offer = append(offer, actionID)
		}
	}
	return offer
}

// applyRouteExecution checks a lookup's execution context against the
// serving package and offer, in the contract's order (catalog, expect,
// hold), and writes it into the decide request: the caller's epoch and
// attribution, its hold, relaxed (no session state), and user_turn_cache
// when it carries admitted cache evidence. The cache evidence itself is
// sent by the request's cache_context hook (routeExecutionCacheContext).
// It answers a failure class, or "" once applied.
func applyRouteExecution(
	request *raylinearc.PolicyDecisionRequest,
	scorer *policyServiceScorer,
	execution *selection.RaylineARCRouteExecution,
	identity selection.RaylineARCRouteIdentity,
) string {
	if execution == nil {
		return ""
	}
	for _, actionID := range execution.ActionIDs {
		if _, bound := scorer.bindings[actionID]; !bound {
			return arcFailureExecutionUnknownAction
		}
	}
	for _, pin := range []struct {
		expect  *string
		serving string
		class   string
	}{
		{execution.ExpectPackage, identity.PackageSHA256, arcFailureExecutionIdentityPackage},
		{execution.ExpectThinkingControls, identity.ThinkingControlsSHA256, arcFailureExecutionIdentityControls},
		{execution.ExpectOffer, identity.OfferSHA256, arcFailureExecutionIdentityOffer},
	} {
		if pin.expect != nil && *pin.expect != pin.serving {
			return pin.class
		}
	}
	held := execution.HoldActionID
	switch execution.HoldScope {
	case "model":
		// The service holds the last attributed message's model; a hold
		// naming another would be silently overruled, so it is refused.
		if last := len(execution.Attribution); last > 0 &&
			scorer.actionModel(execution.Attribution[last-1].ActionID) != scorer.actionModel(held) {
			return arcFailureExecutionHoldConflict
		}
		request.Selection.AttestedHeldActionID = &held
	case "action":
		if !slices.Contains(scorer.offerOf(request.Selection.AvailableActionIDs), held) {
			return arcFailureExecutionHoldUnavailable
		}
		request.Selection.HeldActionID = &held
	}
	request.ContextEpoch = execution.ContextEpoch
	request.Attribution = append([]raylinearc.PolicyAttribution{}, execution.Attribution...)
	request.EpisodeMode = raylinearc.PolicyEpisodeModeRelaxed
	if execution.UserTurnCache != "" {
		request.Selection.UserTurnCache = execution.UserTurnCache
	}
	return ""
}

// routeExecutionCacheContext is the decide request's cache_context for a
// lookup with an execution context: the caller's admitted evidence in place
// of VSR's own warmth, which a stateless lookup does not have; nil when the
// caller sent no workers.
func routeExecutionCacheContext(execution *selection.RaylineARCRouteExecution) *raylinearc.PolicyCacheContext {
	if execution == nil || execution.CacheContext == nil {
		return nil
	}
	return &raylinearc.PolicyCacheContext{
		Actions: append([]raylinearc.PolicyActionCache{}, execution.CacheContext.Actions...),
	}
}

// routeExecutionServiceRefusal maps the policy service's refusal of an
// execution context onto its public class: an action or arm outside the
// package's catalog, and a held action the offer excludes.
func routeExecutionServiceRefusal(err error) string {
	var failure *raylinearc.PolicyServiceError
	if !errors.As(err, &failure) {
		return ""
	}
	reason, _ := failure.Detail["reason"].(string)
	switch {
	case failure.Class == "held_action_unavailable", reason == "held_action_unavailable":
		return arcFailureExecutionHoldUnavailable
	case reason == "attribution_out_of_catalog":
		return arcFailureExecutionUnknownAction
	}
	return ""
}

// routeExecutionRefusal is an execution refusal on its way to the routes
// adapter, which answers it with the contract's code and status.
type routeExecutionRefusal struct {
	class string
	err   error
}

func (refusal *routeExecutionRefusal) Error() string {
	return refusal.class + ": " + refusal.err.Error()
}
func (refusal *routeExecutionRefusal) Unwrap() error { return refusal.err }

// contract is the refusal as the public error: invalid_request_error, the
// code and its status, and for an identity mismatch the member that differs.
func (refusal *routeExecutionRefusal) contract() *routesContractError {
	entry := routeExecutionClasses[refusal.class]
	message := map[string]string{
		"execution_unknown_action":    "routing.execution names an action or arm outside the serving package's catalog.",
		"execution_identity_mismatch": "routing.execution.expect." + entry.member + " differs from the serving value.",
		"execution_hold_conflict":     "routing.execution.hold names a model other than the last attributed message's.",
		"execution_hold_unavailable":  "routing.execution.hold names an action this request's offer excludes.",
	}[entry.code]
	return &routesContractError{status: entry.status, errType: "invalid_request_error", code: entry.code, message: message}
}
