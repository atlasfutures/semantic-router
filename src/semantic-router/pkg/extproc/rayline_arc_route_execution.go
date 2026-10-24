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
	// Hybrid routing v2 (pathfinder #5904).
	arcFailureExecutionLastServedConflict = "execution_last_served_conflict"
	arcFailureExecutionHoldInfeasible     = "execution_hold_infeasible"
	arcFailureExecutionNoFeasibleTarget   = "execution_no_feasible_target"
	arcFailurePolicyPresetNotAdmitted     = "policy_preset_not_admitted"
	arcFailurePolicyScheduleNotAdmitted   = "policy_schedule_not_admitted"
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
	arcFailureExecutionUnknownAction:      {"execution_unknown_action", 400, ""},
	arcFailureExecutionIdentityPackage:    {"execution_identity_mismatch", 409, "package_sha256"},
	arcFailureExecutionIdentityControls:   {"execution_identity_mismatch", 409, "thinking_controls_sha256"},
	arcFailureExecutionIdentityOffer:      {"execution_identity_mismatch", 409, "offer_sha256"},
	arcFailureExecutionHoldConflict:       {"execution_hold_conflict", 409, ""},
	arcFailureExecutionHoldUnavailable:    {"execution_hold_unavailable", 409, ""},
	arcFailureExecutionLastServedConflict: {"execution_last_served_conflict", 400, ""},
	arcFailureExecutionHoldInfeasible:     {"execution_hold_infeasible", 409, ""},
	arcFailureExecutionNoFeasibleTarget:   {"execution_no_feasible_target", 409, ""},
	arcFailurePolicyPresetNotAdmitted:     {"policy_preset_not_admitted", 400, ""},
	arcFailurePolicyScheduleNotAdmitted:   {"policy_schedule_not_admitted", 400, ""},
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
	if execution != nil {
		return checkRouteOfferPin(execution.ExpectOffer, identity)
	}
	return ""
}

// checkRouteOfferPin refuses an answer decided over another offer than the
// caller pinned (arc_execution_context.check_offer).
func checkRouteOfferPin(expect *string, identity selection.RaylineARCRouteIdentity) string {
	if expect != nil && *expect != identity.OfferSHA256 {
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
	// v2's service reasons (pathfinder #5906, selection_refusals_v2.v1.json).
	// last_served_conflict is the binding's, never the service's.
	case reason == "no_feasible_target":
		return arcFailureExecutionNoFeasibleTarget
	case reason == "policy_preset_not_admitted":
		return arcFailurePolicyPresetNotAdmitted
	case reason == "policy_schedule_not_admitted":
		return arcFailurePolicyScheduleNotAdmitted
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
		"execution_unknown_action":       "routing.execution names an action or arm outside the serving package's catalog.",
		"execution_identity_mismatch":    "routing.execution.expect." + entry.member + " differs from the serving value.",
		"execution_hold_conflict":        "routing.execution.hold names a model other than the last attributed message's.",
		"execution_hold_unavailable":     "routing.execution.hold names an action this request's offer excludes.",
		"execution_last_served_conflict": "routing.execution.last_served contradicts the history it should agree with.",
		"execution_hold_infeasible":      "routing.execution.hold names an action the request's own capacity marks not to fit.",
		"execution_no_feasible_target":   "No offered action can serve this request: the held model cannot run and the policy allows no handoff.",
		"policy_preset_not_admitted":     "routing.policy.preset is not one this cell serves.",
		"policy_schedule_not_admitted":   "routing.policy.overrides.schedule names a class the package does not admit for that dimension.",
	}[entry.code]
	return &routesContractError{status: entry.status, errType: "invalid_request_error", code: entry.code, message: message}
}

// applyRoutePolicyAndExecutionV2 writes routing.policy and a v2
// routing.execution into the decide request after checking the envelope
// against the serving package and the request's offer, in bind_v2's order:
// every named action in the catalog; the package and controls pins;
// last_served consistent with history; the hold feasible, then offered. The
// offer pin is checked after the decide. It answers a failure class, or ""
// once applied.
func applyRoutePolicyAndExecutionV2(
	request *raylinearc.PolicyDecisionRequest,
	scorer *policyServiceScorer,
	controls *selection.RaylineARCRouteControls,
) string {
	if controls == nil {
		return ""
	}
	if policy := controls.Policy; policy != nil {
		member := *policy
		request.Policy = &member
	}
	v2 := controls.ExecutionV2
	if v2 == nil {
		return ""
	}
	for _, actionID := range v2.ActionIDs {
		if _, bound := scorer.bindings[actionID]; !bound {
			return arcFailureExecutionUnknownAction
		}
	}
	if pin := v2.ExpectPackage; pin != nil && *pin != scorer.sha256 {
		return arcFailureExecutionIdentityPackage
	}
	if pin := v2.ExpectThinkingControls; pin != nil && *pin != scorer.thinkingControls {
		return arcFailureExecutionIdentityControls
	}
	if served := v2.LastServedActionID; served != nil && len(v2.Attribution) > 0 {
		last := v2.Attribution[len(v2.Attribution)-1]
		var same *raylinearc.PolicyAttribution
		if message := v2.LastServedMessage; message != nil {
			for index := range v2.Attribution {
				if v2.Attribution[index].Message == *message {
					same = &v2.Attribution[index]
				}
			}
		}
		if same != nil && same.ActionID != *served ||
			same == nil && scorer.actionModel(last.ActionID) != scorer.actionModel(*served) {
			return arcFailureExecutionLastServedConflict
		}
	}
	if hold := v2.HoldActionID; hold != nil {
		if slices.Contains(v2.Infeasible, *hold) {
			return arcFailureExecutionHoldInfeasible
		}
		if !slices.Contains(request.Selection.AvailableActionIDs, *hold) {
			return arcFailureExecutionHoldUnavailable
		}
		held := *hold
		request.Selection.HeldActionID = &held
	}
	if served := v2.LastServedActionID; served != nil {
		value := *served
		request.Selection.LastServedActionID = &value
	}
	request.ContextEpoch = v2.ContextEpoch
	request.Attribution = append([]raylinearc.PolicyAttribution{}, v2.Attribution...)
	request.EpisodeMode = raylinearc.PolicyEpisodeModeRelaxed
	event := v2.Event
	request.Event = &event
	request.Capacity = append([]raylinearc.PolicyActionCapacity(nil), v2.Capacity...)
	request.TurnKey = v2.TurnKey
	request.Reconsult = v2.Reconsult
	return ""
}

// routeExecutionV2CacheContext is a v2 lookup's cache_context: the admitted
// evidence of its targets; nil when none carried any.
func routeExecutionV2CacheContext(v2 *selection.RaylineARCRouteExecutionV2) *raylinearc.PolicyCacheContext {
	if v2 == nil || v2.CacheContext == nil {
		return nil
	}
	return &raylinearc.PolicyCacheContext{Actions: append([]raylinearc.PolicyActionCache{}, v2.CacheContext.Actions...)}
}
