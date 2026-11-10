package extproc

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Failure classes of a Decisions lookup whose service answer the lookup
// cannot stand behind. Each is withheld as inconsistent (502), never served
// with a substituted or partial result.
const (
	// arcFailureRoutesResultMissing: a requested result point came back
	// with no pick, or with one this cell cannot dispatch or did not offer.
	arcFailureRoutesResultMissing = "routes_result_missing"
	// arcFailureRoutesPricesNotApplied: the service did not score with the
	// caller's prices (no live echo, or a priced model's leg filled from the
	// package).
	arcFailureRoutesPricesNotApplied = "routes_caller_prices_not_applied"
	// arcFailureRoutesExplorationInconsistent: an exploration record names
	// another requested temperature, applies a higher one, or is applied
	// at 0.
	arcFailureRoutesExplorationInconsistent = "routes_exploration_inconsistent"
	// arcFailureRoutesUnknownPriceModel: the service refused a priced model
	// no arm of its package dispatches (selection_refused,
	// unknown_price_model): the caller's to fix, a 400.
	arcFailureRoutesUnknownPriceModel = "routes_unknown_price_model"
	// arcFailureRoutesCallerPricesUnsupported and ...Inconsistent: the
	// service refused the caller's price world (unsupported_request,
	// caller_prices_unsupported: its package cannot score caller prices;
	// caller_prices_inconsistent: the prices price one model group more
	// than one way). Each answers its own 400 code.
	arcFailureRoutesCallerPricesUnsupported  = "routes_caller_prices_unsupported"
	arcFailureRoutesCallerPricesInconsistent = "routes_caller_prices_inconsistent"
	// arcFailureRoutesPredictionsInconsistent: the service answered a
	// predictions request without its table, or a tier pick names a row
	// that is not its model's offered candidate (or names one off a
	// boundary).
	arcFailureRoutesPredictionsInconsistent = "routes_predictions_inconsistent"
)

var routesInconsistentClasses = map[string]bool{
	arcFailureRoutesResultMissing:           true,
	arcFailureRoutesPricesNotApplied:        true,
	arcFailureRoutesExplorationInconsistent: true,
	arcFailureRoutesPredictionsInconsistent: true,
	arcFailureRoutesExecutionTraceMissing:   true,
	arcFailureRoutesSignalsMissing:          true,
}

// arcFailureRoutesExecutionTraceMissing: a routing.execution lookup asked
// for decision.cache_trace and the answer does not carry it.
const arcFailureRoutesExecutionTraceMissing = "routes_execution_trace_missing"

// routeExclusiveOffer keeps the offered actions whose arm serves a priced
// model: with routing.exclusive, the priced models are the only candidates.
// A remapped action is served by its dispatch_model_overrides to-worker.
func routeExclusiveOffer(scorer *policyServiceScorer, available []string, priced []string) []string {
	kept := make([]string, 0, len(available))
	for _, actionID := range available {
		binding, ok := scorer.bindings[actionID]
		if !ok || binding.dispatchArm < 0 || binding.dispatchArm >= len(scorer.workers) {
			continue
		}
		if slices.Contains(priced, scorer.workers[binding.dispatchArm].Model) {
			kept = append(kept, actionID)
		}
	}
	return kept
}

// routeServicePrices are a caller's prices keyed as the service reads them.
//
// The caller keys a model by the provider id the cell lists and serves it on
// (an action's dispatched worker's id). The service keys an arm's rate by the
// provider id of the arm's package model, and refuses a key no arm reads
// (unknown_price_model). The two differ under dispatch_model_overrides: an
// action bound to a from-worker is sent on the to-worker, yet its arm still
// reads the from-worker's id; so does an action bound to the to-worker itself
// when it serves the same package model (a twin checkpoint standing in for
// the from-worker). Each such arm's rate therefore goes out under its service
// id, at the legs the caller gave the worker it is dispatched to, and a
// dispatched id no arm reads is not sent. Every other key goes out as sent.
// Without an action whose two ids differ and whose dispatched id the caller
// priced, prices itself is returned.
//
// Precedence, never a refusal: when the caller prices both the dispatched id
// and the package model's own id, the dispatched id's price wins, since the
// call is billed at the model it is sent to. A caller pricing only the
// package model's id is read as before. Between two dispatched workers whose
// arms read one service id (two from-workers of one model, different ids,
// onto one to-worker), the first bound action's wins.
func routeServicePrices(scorer *policyServiceScorer, prices *raylinearc.PolicyRoutingPrices) *raylinearc.PolicyRoutingPrices {
	if scorer == nil || prices == nil {
		return prices
	}
	arms := routeServicePriceArms(scorer)
	translated := false
	for _, arm := range arms {
		if _, priced := prices.Rates[arm.served]; priced && arm.service != arm.served {
			translated = true
		}
	}
	if !translated {
		return prices
	}
	rates := make(map[string]raylinearc.PolicyRoutingRate, len(prices.Rates))
	put := func(key string, rate raylinearc.PolicyRoutingRate) {
		if _, set := rates[key]; !set {
			rates[key] = rate
		}
	}
	dispatched := map[string]bool{}
	for _, arm := range arms {
		if rate, priced := prices.Rates[arm.served]; priced && arm.service != arm.served {
			dispatched[arm.served] = true
			put(arm.service, rate)
		}
	}
	// A key some arm reads at the service is kept (behind any dispatched
	// price already set under it), as is any key no remapped arm is
	// dispatched to.
	read := map[string]bool{}
	for _, arm := range arms {
		read[arm.service] = true
	}
	for key, rate := range prices.Rates {
		if !dispatched[key] || read[key] {
			put(key, rate)
		}
	}
	sent := *prices
	sent.Rates = rates
	return &sent
}

// routeServicePriceArm is one bound action's two price ids: the provider id
// of the worker it is served on (the caller's key) and the one its arm reads
// at the service.
type routeServicePriceArm struct{ served, service string }

// routeServicePriceArms resolves each bound action's ids. An action's arm
// reads its bound worker's id, except on a dispatch_model_overrides
// to-worker: an action bound there whose package model is a from-worker's
// reads that from-worker's id, the one the package binds the model to. The
// cell does not see the package's model ids, so this takes the from-worker
// to be the package model's own worker and the to-worker its stand-in, as a
// dispatch-only remap is meant to be configured. Two
// from-workers of one model with different ids onto one to-worker leave the
// to-worker's own actions on its own id.
func routeServicePriceArms(scorer *policyServiceScorer) []routeServicePriceArm {
	valid := func(index int) bool { return index >= 0 && index < len(scorer.workers) }
	type twin struct {
		to    int
		model string
	}
	standIn := map[twin]string{}
	ambiguous := map[twin]bool{}
	for _, actionID := range scorer.actionOrder {
		binding, ok := scorer.bindings[actionID]
		if !ok || !valid(binding.arm) || !valid(binding.dispatchArm) || binding.arm == binding.dispatchArm || binding.model == "" {
			continue
		}
		key, own := twin{binding.dispatchArm, binding.model}, scorer.workers[binding.arm].Model
		if have, seen := standIn[key]; seen && have != own {
			ambiguous[key] = true
		}
		standIn[key] = own
	}
	arms := make([]routeServicePriceArm, 0, len(scorer.actionOrder))
	for _, actionID := range scorer.actionOrder {
		binding, ok := scorer.bindings[actionID]
		if !ok || !valid(binding.arm) || !valid(binding.dispatchArm) {
			continue
		}
		arm := routeServicePriceArm{served: scorer.workers[binding.dispatchArm].Model, service: scorer.workers[binding.arm].Model}
		key := twin{binding.arm, binding.model}
		if own, ok := standIn[key]; ok && binding.arm == binding.dispatchArm && !ambiguous[key] {
			arm.service = own
		}
		arms = append(arms, arm)
	}
	return arms
}

// routeRateLegs are a rate's four legs in the contract's order.
func routeRateLegs(rate raylinearc.PolicyRoutingRate) [4]float64 {
	var legs [4]float64
	for index, leg := range []*float64{rate.Input, rate.CacheRead, rate.CacheWrite, rate.Output} {
		if leg != nil {
			legs[index] = *leg
		}
	}
	return legs
}

// routeServicePriceSHA256 is the price_sha256 the service restates for the
// world it was sent: the caller's own hash when the rates went out as sent.
func routeServicePriceSHA256(controls *selection.RaylineARCRouteControls, sent *raylinearc.PolicyRoutingPrices) string {
	if sent == controls.Prices {
		return controls.PriceSHA256
	}
	legs := make(map[string][4]float64, len(sent.Rates))
	for key, rate := range sent.Rates {
		legs[key] = routeRateLegs(rate)
	}
	return routesPriceSHA256(legs, controls.Exclusive)
}

// applyRouteControls writes a lookup's controls into its decide request: its
// result points beside any the cell configures (a configured point of the
// same name is replaced), its exploration in place of the cell's, and its
// prices (as routeServicePrices keys them) in place of the live feed's.
func applyRouteControls(request *raylinearc.PolicyDecisionRequest, scorer *policyServiceScorer, controls *selection.RaylineARCRouteControls) {
	if controls == nil {
		return
	}
	points := make([]raylinearc.PolicyTierPoint, 0, len(request.Selection.TierPoints)+len(controls.Points))
	for _, point := range request.Selection.TierPoints {
		if !slices.ContainsFunc(controls.Points, func(own raylinearc.PolicyTierPoint) bool { return own.Name == point.Name }) {
			points = append(points, point)
		}
	}
	points = append(points, controls.Points...)
	request.Selection.TierPoints = points
	if controls.Exploration != nil {
		member := *controls.Exploration
		if member.Temperature != nil {
			temperature := *member.Temperature
			member.Temperature = &temperature
		}
		request.Selection.Exploration = &member
	}
	if controls.Prices != nil {
		request.RoutingPrices = routeServicePrices(scorer, controls.Prices)
	}
}

// recordRouteControls checks the service's answer to a lookup's controls and
// keeps its results on the trace. It returns a failure class, "" when the
// answer stands.
func recordRouteControls(
	trace *selection.RaylineARCTrace,
	scorer *policyServiceScorer,
	controls *selection.RaylineARCRouteControls,
	response *raylinearc.PolicyDecisionResponse,
	available []string,
	predictionsAsked bool,
) string {
	if controls == nil {
		return ""
	}
	if predictionsAsked {
		if !routePredictionsConsistent(controls, response) {
			return arcFailureRoutesPredictionsInconsistent
		}
		trace.RoutePredictions = response.Decision.Predictions
		trace.RoutePredictionsApplied = true
	}
	if controls.Prices != nil {
		sent := routeServicePrices(scorer, controls.Prices)
		if !routeCallerPricesApplied(controls, routeServicePriceSHA256(controls, sent), response.RoutingPricesUsed) {
			return arcFailureRoutesPricesNotApplied
		}
	}
	var requested *float64
	if controls.Exploration != nil && controls.Exploration.Temperature != nil {
		requested = &controls.Exploration.Temperature.Temperature
	}
	if requested != nil {
		if stageOne := response.Decision.StageOne; stageOne != nil && stageOne.Exploration != nil {
			if !routeExplorationConsistent(stageOne.Exploration, *requested) {
				return arcFailureRoutesExplorationInconsistent
			}
			record := *stageOne.Exploration
			trace.RouteExploration = &record
		}
	}
	valid := policyTierPickValid(scorer, available)
	results := make(map[string]selection.RaylineARCRouteResult, len(controls.Points))
	for _, point := range controls.Points {
		pick, ok := response.Decision.TierPicks[point.Name]
		if !ok || !valid(pick) {
			return arcFailureRoutesResultMissing
		}
		if pick.Exploration != nil && (requested == nil || !routeExplorationConsistent(pick.Exploration, *requested) ||
			pick.Exploration.Result != nil && *pick.Exploration.Result != point.Name) {
			return arcFailureRoutesExplorationInconsistent
		}
		// A result names the worker its action is sent on, as the primary
		// route does under dispatch_model_overrides.
		binding := scorer.bindings[pick.SelectedActionID]
		results[point.Name] = selection.RaylineARCRouteResult{Pick: pick, Arm: binding.dispatchArm, ThinkingLevel: binding.level}
	}
	trace.RouteResults = results
	return ""
}

// routeCallerPricesApplied reports whether the decision was scored in the
// caller's price world (pathfinder#5668): a caller-mode echo restating this
// world's price_sha256 (sha, the hash of the rates as sent) and exclusivity,
// with nothing filled or fallen back.
func routeCallerPricesApplied(controls *selection.RaylineARCRouteControls, sha string, used *raylinearc.PolicyRoutingPricesUsed) bool {
	if used == nil || used.Mode != raylinearc.RoutingPricesModeCaller || used.PricesSHA256 == nil || used.Exclusive == nil {
		return false
	}
	return *used.PricesSHA256 == sha && *used.Exclusive == controls.Exclusive &&
		len(used.FilledLegs) == 0 && len(used.FallbackArms) == 0
}

// routeExplorationConsistent holds a temperature-shape record to the
// requested T: it names that T, never applies a higher one, and is not
// applied at 0.
func routeExplorationConsistent(record *raylinearc.PolicyExplorationRecord, requested float64) bool {
	if record.RequestedTemperature == nil || *record.RequestedTemperature != requested {
		return false
	}
	if record.Applied {
		return record.Temperature != nil && *record.Temperature > 0 && *record.Temperature <= requested &&
			record.Propensity != nil && *record.Propensity > 0 && *record.Propensity <= 1 && record.Explored != nil
	}
	return record.Reason != nil
}

// routePredictionsConsistent holds a predictions answer to the request: the
// table is present, and each requested point's pick names an offered row of
// its own model on a boundary, and none off one.
func routePredictionsConsistent(controls *selection.RaylineARCRouteControls, response *raylinearc.PolicyDecisionResponse) bool {
	predictions := response.Decision.Predictions
	if predictions == nil {
		return false
	}
	for _, point := range controls.Points {
		pick, ok := response.Decision.TierPicks[point.Name]
		if !ok {
			return false
		}
		index := pick.StageOneCandidate
		if predictions.StageOne == nil {
			if index != nil {
				return false
			}
			continue
		}
		rows := predictions.StageOne.Candidates
		if index == nil || *index < 0 || *index >= len(rows) || rows[*index].Model != pick.Model || !rows[*index].Offered {
			return false
		}
	}
	return true
}

// routeCallerPricesRefusal maps the service's refusal of a caller price world
// (unsupported_request with detail.reason caller_prices_unsupported or
// caller_prices_inconsistent, pathfinder#5668) to its failure class; "" for
// any other refusal.
func routeCallerPricesRefusal(detail map[string]any) string {
	switch reason, _ := detail["reason"].(string); reason {
	case "caller_prices_unsupported":
		return arcFailureRoutesCallerPricesUnsupported
	case "caller_prices_inconsistent":
		return arcFailureRoutesCallerPricesInconsistent
	default:
		return ""
	}
}

// routeRefusedPriceModels are the models an unknown_price_model refusal
// names (detail.models), kept only where the caller priced them: no service
// text reaches the caller that the caller did not send. Empty, never nil,
// when it names none.
func routeRefusedPriceModels(detail map[string]any, priced []string) []string {
	named := map[string]bool{}
	if list, ok := detail["models"].([]any); ok {
		for _, entry := range list {
			if model, ok := entry.(string); ok {
				named[model] = true
			}
		}
	}
	models := []string{}
	for _, model := range priced {
		if named[model] {
			models = append(models, model)
		}
	}
	return models
}
