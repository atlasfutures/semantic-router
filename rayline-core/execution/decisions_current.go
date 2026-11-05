package execution

import (
	"fmt"
	"math"
)

// The current public contract identifies each preset/custom result by its
// policy action, model and worker. It reports that result's V and price world,
// but does not attest a package, registry, offer or control identity. Those
// remain the configured manifest's provenance; no identity is synthesized.
func (provider *HTTPDecisions) currentTierDecision(result decisionsResponse, selected publicDecision, routingRequested bool) (Decision, error) {
	if provider.config.Decisions.OfferSHA256 != "" {
		return Decision{}, fmt.Errorf("Decisions did not attest the configured offer pin")
	}
	action, err := provider.bindCurrentAction(result.RouteID, selected)
	if err != nil {
		return Decision{}, err
	}
	var value float64
	switch provider.config.Decisions.Result {
	case "swift":
		value = 0.1
	case "sage":
		value = 4.5
	case "custom":
		value = *provider.valueUSD // required at construction, echoed before selection
	default:
		return Decision{}, fmt.Errorf("Decisions result has no current tier contract")
	}
	if selected.ValueUSD == nil || *selected.ValueUSD != value || selected.CostWeight == nil || *selected.CostWeight <= 0 || math.Abs(*selected.CostWeight*value-1) > 1e-9 {
		return Decision{}, fmt.Errorf("Decisions selected result does not match its requested v_value and cost_weight")
	}
	if (routingRequested && result.Routing.Unapplied == nil) || selected.Prices == nil {
		return Decision{}, fmt.Errorf("Decisions omitted the selected result's routing or price receipt")
	}
	// Current result prices contain only basis and digest: the price world
	// the result was decided in, attested by the routing.prices echo.
	profile, err := provider.acknowledgePrices(result, selected.Model, selected.Prices)
	if err != nil {
		return Decision{}, err
	}
	decision := Decision{RouteID: result.RouteID, ActionID: action.ID, Worker: action.Worker, Model: action.Model, ReportedModel: selected.Model, PackageSHA256: provider.config.PackageSHA256, ControlID: action.ControlID, Result: provider.config.Decisions.Result, IdentitySource: "configured_manifest", AppliedValueUSD: selected.ValueUSD, PriceProfile: profile}
	provider.logDecision(decision, selected.Prices.Basis, selected.Prices.SHA256)
	return decision, nil
}
