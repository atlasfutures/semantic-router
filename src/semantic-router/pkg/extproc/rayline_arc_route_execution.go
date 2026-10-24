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

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"

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
	// arcFailureExecutionNotAdmitted is the cell's action_not_admitted
	// without a hold: nothing it offers can serve this request (503).
	arcFailureExecutionNotAdmitted = "execution_offer_not_admitted"
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

// routeIdentity is the identity a lookup was decided under: the serving
// package, its thinking-controls registry, and the offer, which is the
// decide response's available actions (the cell may narrow the request's).
func routeIdentity(scorer *policyServiceScorer, response *raylinearc.PolicyDecisionResponse) selection.RaylineARCRouteIdentity {
	offered := make([]string, 0, len(response.Actions))
	for _, action := range response.Actions {
		if action.Available {
			offered = append(offered, action.ActionID)
		}
	}
	return selection.RaylineARCRouteIdentity{
		PackageSHA256:          scorer.sha256,
		ThinkingControlsSHA256: scorer.thinkingControls,
		OfferSHA256:            routesOfferSHA256(scorer.sha256, offered),
	}
}

// applyRouteExecution checks a lookup's execution context against the
// serving package and the request's offer, in the contract's order
// (arc_execution_context.bind: catalog and arms, the package and
// thinking-controls pins, the hold), and writes it into the decide request:
// the caller's epoch and attribution, its hold, relaxed (no session state),
// and response_fields cache_trace for the echo. The offer pin is checked
// against the answer (checkRouteExecutionOffer), since the cell may narrow
// the offer. It answers a failure class, or "" once applied.
func applyRouteExecution(
	request *raylinearc.PolicyDecisionRequest,
	scorer *policyServiceScorer,
	execution *selection.RaylineARCRouteExecution,
) string {
	if execution == nil {
		return ""
	}
	for _, actionID := range execution.ActionIDs {
		if _, bound := scorer.bindings[actionID]; !bound {
			return arcFailureExecutionUnknownAction
		}
	}
	for _, entry := range execution.Attribution {
		arms, known := scorer.actionArms[entry.ActionID]
		if entry.ArmID != nil && known && !slices.Contains(arms, *entry.ArmID) {
			return arcFailureExecutionUnknownAction
		}
	}
	if pin := execution.ExpectPackage; pin != nil && *pin != scorer.sha256 {
		return arcFailureExecutionIdentityPackage
	}
	if pin := execution.ExpectThinkingControls; pin != nil && *pin != scorer.thinkingControls {
		return arcFailureExecutionIdentityControls
	}
	held := execution.HoldActionID
	if execution.HoldScope == "model" && len(execution.Attribution) > 0 &&
		scorer.actionModel(execution.Attribution[len(execution.Attribution)-1].ActionID) != scorer.actionModel(held) {
		// The service holds the last attributed message's model; a hold
		// naming another would be silently overruled.
		return arcFailureExecutionHoldConflict
	}
	if execution.HoldScope != "" && !slices.Contains(request.Selection.AvailableActionIDs, held) {
		return arcFailureExecutionHoldUnavailable
	}
	switch execution.HoldScope {
	case "model":
		request.Selection.AttestedHeldActionID = &held
	case "action":
		request.Selection.HeldActionID = &held
	}
	request.ContextEpoch = execution.ContextEpoch
	request.Attribution = append([]raylinearc.PolicyAttribution{}, execution.Attribution...)
	request.EpisodeMode = raylinearc.PolicyEpisodeModeRelaxed
	// The trace the echo copies is asked only of a service that advertises
	// it; one that does not answers the echo unsupported, visibly.
	if scorer.advertises(raylinearc.PolicyCapabilityCacheTrace) &&
		!slices.Contains(request.ResponseFields, routesCacheTraceField) {
		request.ResponseFields = append(append([]string(nil), request.ResponseFields...), routesCacheTraceField)
	}
	return ""
}

// routeExecutionTraceMissing is a requested cache_trace the answer does not
// carry: the lookup fails as inconsistent (502), never answered with an
// echo that hides it.
func routeExecutionTraceMissing(request *raylinearc.PolicyDecisionRequest, response *raylinearc.PolicyDecisionResponse) bool {
	return slices.Contains(request.ResponseFields, routesCacheTraceField) && response.Decision.CacheTrace == nil
}

// routesCacheTraceField elicits decision.cache_trace.
const routesCacheTraceField = "cache_trace"

// checkRouteExecutionOffer refuses an answer decided over another offer
// than the caller pinned: it never reaches the caller.
func checkRouteExecutionOffer(execution *selection.RaylineARCRouteExecution, identity selection.RaylineARCRouteIdentity) string {
	if execution != nil && execution.ExpectOffer != nil && *execution.ExpectOffer != identity.OfferSHA256 {
		return arcFailureExecutionIdentityOffer
	}
	return ""
}

// routeExecutionCacheContext is the decide request's cache_context for a
// lookup with an execution context: the caller's admitted evidence in place
// of VSR's own warmth, which a stateless lookup does not have; nil when the
// caller sent no workers.
func routeExecutionCacheContext(execution *selection.RaylineARCRouteExecution) *cachecontext.PolicyCacheContext {
	if execution == nil || execution.CacheContext == nil {
		return nil
	}
	return &cachecontext.PolicyCacheContext{
		Actions: append([]cachecontext.PolicyActionCache{}, execution.CacheContext.Actions...),
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
	case reason == "action_not_admitted":
		// The cell's offer admits nothing for this request, or not the
		// held action: the hold's refusal when there is one, otherwise the
		// cell is unavailable for this request (503).
		return arcFailureExecutionNotAdmitted
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
