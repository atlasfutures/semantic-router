package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

// validateRaylineARCFallbackContract refuses upstream cross-model fallback on
// a recipe that holds a rayline_arc decision (US-005). ARC owns dispatch: it
// commits the episode on the primary's response headers and aborts on a
// non-2xx, so a fallback 200 from another arm would reach the client while the
// episode records nothing. extproc also guards this at request time
// (selectionOwnsDispatch); this is the config-time refusal. The policy checked
// is the recipe's effective one, so a recipe that omits fallback inherits a
// routing.fallback or global.router.fallback enabled: true and is refused.
func validateRaylineARCFallbackContract(cfg *RouterConfig, decision Decision) error {
	policy, recipe := raylineARCEffectiveFallback(cfg, decision)
	if policy == nil || !policy.Enabled {
		return nil
	}
	return fmt.Errorf(
		"decision '%s': algorithm.type=%s does not support upstream fallback, but recipe %q has fallback.enabled=true (set it false in that recipe's routing.fallback)",
		decision.Name,
		RaylineARCAlgorithmType,
		recipe,
	)
}

// raylineARCEffectiveFallback returns the fallback policy of the recipe that
// owns decision. A recipe-scoped config (the per-profile validation pass)
// carries its effective policy on cfg.Fallback; a whole config is searched for
// the owning recipe by decision identity.
func raylineARCEffectiveFallback(cfg *RouterConfig, decision Decision) (*fallback.FallbackPolicy, RecipeName) {
	if cfg == nil {
		return nil, ""
	}
	scope := cfg.RoutingScope
	if scope == "" {
		scope = DefaultRecipeName
	}
	for i := range cfg.Recipes {
		recipe := &cfg.Recipes[i]
		for j := range recipe.Profile.Decisions {
			owned := &recipe.Profile.Decisions[j]
			if owned.Name == decision.Name && owned.Algorithm == decision.Algorithm {
				return recipe.Profile.Fallback, recipe.Name
			}
		}
	}
	return cfg.Fallback, scope
}
