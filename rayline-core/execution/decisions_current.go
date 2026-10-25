package execution

import (
	"fmt"
	"math"
	"strings"
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
		return Decision{}, fmt.Errorf("Decisions selected result does not match its requested value_usd and cost_weight")
	}
	if (routingRequested && result.Routing.Unapplied == nil) || selected.Prices == nil {
		return Decision{}, fmt.Errorf("Decisions omitted the selected result's routing or price receipt")
	}
	profile, err := provider.appliedPriceProfile(result)
	if err != nil {
		return Decision{}, err
	}
	prices := selected.Prices
	if provider.config.Decisions.PriceSHA256 != "" || profile != nil {
		echo := result.Routing.Prices
		// Current result prices contain only basis and digest. Exclusivity is
		// attested by routing.prices, whose digest covers the entire price world.
		if echo == nil || echo.Status != "applied" || !validSHA256(echo.SHA256) || (provider.config.Decisions.PriceSHA256 != "" && echo.SHA256 != provider.config.Decisions.PriceSHA256) || echo.Exclusive == nil || *echo.Exclusive != provider.exclusive || prices.Basis != "caller" || prices.SHA256 == nil || *prices.SHA256 != echo.SHA256 {
			return Decision{}, fmt.Errorf("Decisions selected result did not apply the requested caller price world")
		}
	} else if prices.Basis != "catalog" || prices.SHA256 != nil || provider.exclusive || result.Routing.Prices != nil {
		return Decision{}, fmt.Errorf("Decisions selected result did not confirm the requested catalog price world")
	}
	decision := Decision{RouteID: result.RouteID, ActionID: action.ID, Worker: action.Worker, Model: action.Model, ReportedModel: selected.Model, PackageSHA256: provider.config.PackageSHA256, ControlID: action.ControlID, Result: provider.config.Decisions.Result, IdentitySource: "configured_manifest", AppliedValueUSD: selected.ValueUSD, PriceProfile: profile}
	provider.logDecision(decision, prices.Basis, prices.SHA256)
	return decision, nil
}

func (provider *HTTPDecisions) appliedPriceProfile(result decisionsResponse) (*AppliedPriceProfile, error) {
	if provider.profileID == "" {
		return nil, nil
	}
	profile := result.Routing.PriceProfile
	if profile == nil || profile.Status != "applied" || profile.ID != provider.profileID || profile.Version <= 0 || profile.Version > 9007199254740991 || (provider.profileVersion != 0 && profile.Version != provider.profileVersion) || !strings.HasPrefix(profile.ContentHash, "sha256:") || !validSHA256(strings.TrimPrefix(profile.ContentHash, "sha256:")) {
		return nil, fmt.Errorf("Decisions did not apply the requested price profile")
	}
	return &profile.AppliedPriceProfile, nil
}
