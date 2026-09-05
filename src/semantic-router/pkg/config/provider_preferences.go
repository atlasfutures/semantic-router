package config

import (
	"fmt"
	"strings"
)

// OpenRouter's data_collection verdict: whether a provider that stores prompts
// may serve the arm.
const (
	ProviderDataCollectionAllow = "allow"
	ProviderDataCollectionDeny  = "deny"
)

// OpenRouterProviderPreferences pins which OpenRouter providers may serve an
// arm. OpenRouter picks a provider per request by its own price and uptime
// ranking, so two turns on the same arm can land on providers that differ in
// tokenizer, quantization and context handling; an arm measured on one of them
// is not the arm the next turn gets. This states the choice instead.
//
// It is encoded verbatim as the top-level `provider` object on the Chat
// Completions body, so the JSON tags are the wire and the YAML tags are the
// config. One struct rather than two keeps the two from drifting apart.
// Documented at https://openrouter.ai/docs/features/provider-routing, read
// 2026-09-05.
//
// Order and AllowFallbacks are the pair that pins: an order with fallbacks
// still allowed is a preference, not a pin, because OpenRouter may serve the
// turn from any other provider once the listed ones decline.
//
// AllowFallbacks and RequireParameters are pointers because absent is not
// false. OpenRouter's own defaults apply to a key the operator did not write,
// and sending false where nothing was said would change routing rather than
// leave it alone.
type OpenRouterProviderPreferences struct {
	Order             []string `yaml:"order,omitempty" json:"order,omitempty"`
	AllowFallbacks    *bool    `yaml:"allow_fallbacks,omitempty" json:"allow_fallbacks,omitempty"`
	Only              []string `yaml:"only,omitempty" json:"only,omitempty"`
	Ignore            []string `yaml:"ignore,omitempty" json:"ignore,omitempty"`
	RequireParameters *bool    `yaml:"require_parameters,omitempty" json:"require_parameters,omitempty"`
	DataCollection    string   `yaml:"data_collection,omitempty" json:"data_collection,omitempty"`
}

// validateProviderPreferences refuses a pin the upstream would refuse, at load
// rather than at dispatch. A cell that starts with an unusable pin serves every
// turn to a provider the operator ruled out, or none at all.
func validateProviderPreferences(modelName string, preferences *OpenRouterProviderPreferences) error {
	if preferences == nil {
		return nil
	}
	if len(preferences.Order) == 0 && len(preferences.Only) == 0 {
		return fmt.Errorf(
			"providers.models[%s].provider_preferences must set order or only",
			modelName,
		)
	}
	for field, slugs := range map[string][]string{
		"order":  preferences.Order,
		"only":   preferences.Only,
		"ignore": preferences.Ignore,
	} {
		if err := validateProviderSlugs(modelName, field, slugs); err != nil {
			return err
		}
	}
	switch preferences.DataCollection {
	case "", ProviderDataCollectionAllow, ProviderDataCollectionDeny:
		return nil
	default:
		return fmt.Errorf(
			"providers.models[%s].provider_preferences.data_collection must be %q or %q",
			modelName,
			ProviderDataCollectionAllow,
			ProviderDataCollectionDeny,
		)
	}
}

func validateProviderSlugs(modelName string, field string, slugs []string) error {
	for index, slug := range slugs {
		if strings.TrimSpace(slug) == "" {
			return fmt.Errorf(
				"providers.models[%s].provider_preferences.%s[%d] must be a non-empty provider slug",
				modelName,
				field,
				index,
			)
		}
	}
	return nil
}

// ProviderPreferencesForModel returns the OpenRouter provider pin for a model,
// or nil when the model does not pin. It accepts the same names the rest of the
// model lookups do, including a provider model id the gateway rewrote the
// request to, so the caller does not have to know which name it holds.
func (c *RouterConfig) ProviderPreferencesForModel(modelName string) *OpenRouterProviderPreferences {
	params, found := c.resolveModelConfig(modelName)
	if !found {
		return nil
	}
	return params.ProviderPreferences
}

// copyProviderPreferences keeps the loaded config free of shared slices, the
// same way the other normalizers copy what they carry.
func copyProviderPreferences(preferences *OpenRouterProviderPreferences) *OpenRouterProviderPreferences {
	if preferences == nil {
		return nil
	}
	copied := OpenRouterProviderPreferences{
		Order:             append([]string(nil), preferences.Order...),
		AllowFallbacks:    copyBool(preferences.AllowFallbacks),
		Only:              append([]string(nil), preferences.Only...),
		Ignore:            append([]string(nil), preferences.Ignore...),
		RequireParameters: copyBool(preferences.RequireParameters),
		DataCollection:    preferences.DataCollection,
	}
	return &copied
}
