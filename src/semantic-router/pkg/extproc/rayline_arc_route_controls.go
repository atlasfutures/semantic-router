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
}

// arcFailureRoutesExecutionTraceMissing: a routing.execution lookup asked
// for decision.cache_trace and the answer does not carry it.
const arcFailureRoutesExecutionTraceMissing = "routes_execution_trace_missing"

// routeExclusiveOffer keeps the offered actions whose arm serves a priced
// model: with routing.exclusive, the priced models are the only candidates.
func routeExclusiveOffer(scorer *policyServiceScorer, available []string, priced []string) []string {
	kept := make([]string, 0, len(available))
	for _, actionID := range available {
		binding, ok := scorer.bindings[actionID]
		if !ok || binding.arm < 0 || binding.arm >= len(scorer.workers) {
			continue
		}
		if slices.Contains(priced, scorer.workers[binding.arm].Model) {
			kept = append(kept, actionID)
		}
	}
	return kept
}

// applyRouteControls writes a lookup's controls into its decide request: its
// result points beside any the cell configures (a configured point of the
// same name is replaced), its exploration in place of the cell's, and its
// prices in place of the live feed's.
func applyRouteControls(request *raylinearc.PolicyDecisionRequest, controls *selection.RaylineARCRouteControls) {
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
		request.RoutingPrices = controls.Prices
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
	if controls.Prices != nil && !routeCallerPricesApplied(controls, response.RoutingPricesUsed) {
		return arcFailureRoutesPricesNotApplied
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
		binding := scorer.bindings[pick.SelectedActionID]
		results[point.Name] = selection.RaylineARCRouteResult{Pick: pick, Arm: binding.arm, ThinkingLevel: binding.level}
	}
	trace.RouteResults = results
	return ""
}

// routeCallerPricesApplied reports whether the decision was scored in the
// caller's price world (pathfinder#5668): a caller-mode echo restating this
// world's price_sha256 and exclusivity, with nothing filled or fallen back.
func routeCallerPricesApplied(controls *selection.RaylineARCRouteControls, used *raylinearc.PolicyRoutingPricesUsed) bool {
	if used == nil || used.Mode != raylinearc.RoutingPricesModeCaller || used.PricesSHA256 == nil || used.Exclusive == nil {
		return false
	}
	return *used.PricesSHA256 == controls.PriceSHA256 && *used.Exclusive == controls.Exclusive &&
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
