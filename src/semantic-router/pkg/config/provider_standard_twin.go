package config

import (
	"fmt"
	"slices"
	"strings"
)

// A standard twin pairs a discount endpoint with the standard endpoint of the
// same model. A discount endpoint (OpenRouter's openai/flex, or a contributor
// route) serves the same model at a lower price and is throttled harder, so a
// call it refuses with a 429, or lets time out before any response byte, is
// re-sent once on the twin: same model, same controls, standard provider
// routing. Nothing else is re-sent, and never to another model
// (memex-desktop#7454, "Guidance: the Flex 429 -> standard fallback").
//
// The pairing is declared, never inferred: only a providers.models entry that
// names its twin gets the re-send, so every other arm is unchanged.

// validateStandardTwins refuses a pairing that could re-send to another
// model, back onto the discount endpoint, or along a chain of re-sends.
func validateStandardTwins(models []CanonicalProviderModel) error {
	byName := make(map[string]CanonicalProviderModel, len(models))
	for _, model := range models {
		byName[model.Name] = model
	}
	for _, model := range models {
		if model.StandardTwin == "" {
			continue
		}
		if err := validateStandardTwin(model, byName); err != nil {
			return err
		}
	}
	return nil
}

func validateStandardTwin(discount CanonicalProviderModel, byName map[string]CanonicalProviderModel) error {
	name := discount.Name
	twinName := strings.TrimSpace(discount.StandardTwin)
	if twinName != discount.StandardTwin || twinName == name {
		return fmt.Errorf("providers.models[%s].standard_twin must name another providers.models entry", name)
	}
	twin, ok := byName[twinName]
	if !ok {
		return fmt.Errorf("providers.models[%s].standard_twin %q is not a providers.models entry", name, twinName)
	}
	if twin.StandardTwin != "" {
		// One re-send, never a chain of them.
		return fmt.Errorf("providers.models[%s].standard_twin %q declares a twin of its own", name, twinName)
	}
	if discount.ProviderModelID != "" && twin.ProviderModelID != "" &&
		discount.ProviderModelID != twin.ProviderModelID {
		return fmt.Errorf(
			"providers.models[%s].standard_twin %q serves provider model %q, not %q: a twin is the same model",
			name, twinName, twin.ProviderModelID, discount.ProviderModelID,
		)
	}
	pin := discount.ProviderPreferences
	if pin == nil || len(pin.Only) == 0 {
		return fmt.Errorf(
			"providers.models[%s].standard_twin needs provider_preferences.only naming the discount endpoint",
			name,
		)
	}
	if twin.ProviderPreferences == nil {
		return fmt.Errorf(
			"providers.models[%s] is a standard twin and must set provider_preferences that exclude %v",
			twinName, pin.Only,
		)
	}
	for _, slug := range pin.Only {
		if providerPinAdmits(twin.ProviderPreferences, slug) {
			return fmt.Errorf(
				"providers.models[%s] is the standard twin of %s and must exclude its endpoint %q (ignore it, or leave it out of only)",
				twinName, name, slug,
			)
		}
	}
	return nil
}

// providerPinAdmits reports whether a pin lets OpenRouter serve from slug.
func providerPinAdmits(pin *OpenRouterProviderPreferences, slug string) bool {
	if slices.Contains(pin.Ignore, slug) {
		return false
	}
	return len(pin.Only) == 0 || slices.Contains(pin.Only, slug)
}

// StandardTwinForModel returns the standard twin a discount endpoint's
// refused or timed-out call is re-sent on, or "" when the model declares
// none. It accepts the names the other model lookups accept.
func (c *RouterConfig) StandardTwinForModel(modelName string) string {
	if c == nil {
		return ""
	}
	params, found := c.resolveModelConfig(modelName)
	if !found {
		return ""
	}
	return params.StandardTwin
}
