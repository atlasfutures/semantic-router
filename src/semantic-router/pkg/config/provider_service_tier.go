package config

import (
	"fmt"
	"slices"
	"strings"
)

// providerServiceTiers are the tiers an arm may name for OpenRouter's
// top-level service_tier request member.
var providerServiceTiers = []string{"default", "flex", "priority", "fast", "ultrafast"}

// validateProviderServiceTier refuses a tier OpenRouter would not take, at
// load rather than at dispatch.
func validateProviderServiceTier(modelName string, tier string) error {
	if tier == "" || slices.Contains(providerServiceTiers, tier) {
		return nil
	}
	return fmt.Errorf(
		"providers.models[%s].service_tier must be one of %s",
		modelName,
		strings.Join(providerServiceTiers, ", "),
	)
}

// ServiceTierForModel returns the service tier an arm sends OpenRouter, or ""
// when the arm sets none. It accepts the names the other model lookups accept.
func (c *RouterConfig) ServiceTierForModel(modelName string) string {
	if c == nil {
		return ""
	}
	params, found := c.resolveModelConfig(modelName)
	if !found {
		return ""
	}
	return params.ServiceTier
}
