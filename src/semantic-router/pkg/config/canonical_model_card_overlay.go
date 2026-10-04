package config

// applyCanonicalModelCardOverlay carries the model-card fields this fork adds
// beside the compiled catalog onto the materialized ModelParams: the vision
// and disabled flags, hosted tools and output limit from routing.modelCards,
// and the OpenRouter provider pin from providers.models. It runs after applyEffectiveModelRegistry so every
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
			params.HostedTools = append([]string(nil), card.HostedTools...)
			params.MaxOutputTokens = cardMaxOutputTokens(card)
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
		params.HostedTools = append([]string(nil), card.HostedTools...)
		params.MaxOutputTokens = cardMaxOutputTokens(card)
		if model.ProviderPreferences != nil {
			params.ProviderPreferences = copyProviderPreferences(model.ProviderPreferences)
		}
		cfg.ModelConfig[model.Name] = params
	}
}

// cardMaxOutputTokens is the output limit the operator declared on the card,
// or zero. Only the operator's own value reaches dispatch: a built-in catalog
// maximum can exceed what the deployment's stream deadline can serve, so it is
// not taken as the limit to send.
func cardMaxOutputTokens(card RoutingModel) int {
	if card.MaxOutputTokens == nil {
		return 0
	}
	return *card.MaxOutputTokens
}
