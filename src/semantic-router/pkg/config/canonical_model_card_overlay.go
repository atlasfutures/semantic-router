package config

// applyCanonicalModelCardOverlay carries the model-card fields this fork adds
// beside the compiled catalog onto the materialized ModelParams: the vision
// and disabled flags from routing.modelCards, and the OpenRouter provider pin
// from providers.models. It runs after applyEffectiveModelRegistry so every
// alias already has its params, and it touches nothing the catalog owns.
func applyCanonicalModelCardOverlay(cfg *RouterConfig, canonical *CanonicalConfig) {
	if cfg == nil || canonical == nil || cfg.ModelConfig == nil {
		return
	}
	cardsByName := make(map[string]RoutingModel, len(canonical.Routing.ModelCards))
	for _, card := range canonicalRoutingModels(canonical.Routing) {
		cardsByName[card.Name] = card
	}
	if len(canonical.Providers.Models) == 0 {
		// A routing-only file has no access bindings; its cards are
		// materialised straight from routing.modelCards, so the flags are
		// carried by card name.
		for name, card := range cardsByName {
			params, ok := cfg.ModelConfig[name]
			if !ok {
				continue
			}
			params.Vision = copyBool(card.Vision)
			params.Disabled = copyBool(card.Disabled)
			cfg.ModelConfig[name] = params
		}
		return
	}
	for _, model := range canonical.Providers.Models {
		params, ok := cfg.ModelConfig[model.Name]
		if !ok {
			continue
		}
		card, hasCard := cardsByName[effectiveCanonicalCardID(model)]
		if !hasCard {
			card = cardsByName[model.Name]
		}
		params.Vision = copyBool(card.Vision)
		params.Disabled = copyBool(card.Disabled)
		if model.ProviderPreferences != nil {
			params.ProviderPreferences = copyProviderPreferences(model.ProviderPreferences)
		}
		cfg.ModelConfig[model.Name] = params
	}
}
