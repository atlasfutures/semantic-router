package config

import (
	"fmt"
	"slices"
	"strings"
)

// DefaultRecipeName names the routing profile normalized from the top-level
// `routing:` block. Additional named profiles come from `recipes:`.
const DefaultRecipeName = "default"

// RoutingRecipe is one normalized routing profile and the only place decisions
// live: the recipe named DefaultRecipeName owns the top-level `routing:`
// profile, named recipes own theirs. The flat Signals and Projections fields on
// RouterConfig instead hold the global registry: the union of every recipe's
// profile, so one classifier evaluates any recipe's rules (issue #2331 keeps
// the signal registry global).
type RoutingRecipe struct {
	Name        string
	Description string
	Signals     Signals
	Projections Projections
	Decisions   []Decision
}

// EntrypointMapping binds request-facing virtual model names to a named
// recipe. The virtual names never reach a backend; they only select which
// routing profile evaluates the request.
type EntrypointMapping struct {
	ModelNames []string
	Recipe     string
}

// RecipeByName returns the normalized recipe with the given name.
func (c *RouterConfig) RecipeByName(name string) (*RoutingRecipe, bool) {
	if c == nil {
		return nil, false
	}
	for i := range c.Recipes {
		if c.Recipes[i].Name == name {
			return &c.Recipes[i], true
		}
	}
	return nil, false
}

// DefaultRecipe returns the profile normalized from the top-level `routing:`
// block, or nil for configs built without one.
func (c *RouterConfig) DefaultRecipe() *RoutingRecipe {
	recipe, ok := c.RecipeByName(DefaultRecipeName)
	if !ok {
		return nil
	}
	return recipe
}

// DefaultDecisions returns the default profile's decisions, for the read sites
// whose scope really is that profile (entrypoint-less request paths, canonical
// export of the top-level routing block). Callers that reason about routing as
// a whole want AllRoutingDecisions or GetDecisionByName instead.
func (c *RouterConfig) DefaultDecisions() []Decision {
	if c == nil {
		return nil
	}
	if recipe := c.DefaultRecipe(); recipe != nil {
		return recipe.Decisions
	}
	return nil
}

// RecipeForRequestModel resolves a request model name through the entrypoint
// table. It returns false when the name matches no entrypoint; callers fall
// back to auto-model or specified-model handling.
func (c *RouterConfig) RecipeForRequestModel(modelName string) (*RoutingRecipe, bool) {
	if c == nil {
		return nil, false
	}
	trimmed := strings.TrimSpace(modelName)
	if trimmed == "" {
		return nil, false
	}
	for _, entrypoint := range c.Entrypoints {
		if slices.Contains(entrypoint.ModelNames, trimmed) {
			return c.RecipeByName(entrypoint.Recipe)
		}
	}
	return nil, false
}

// IsEntrypointModelName reports whether the name is a request-facing virtual
// model name from the entrypoint table. Such names never reach a backend; the
// router resolves them like auto-model aliases.
func (c *RouterConfig) IsEntrypointModelName(modelName string) bool {
	_, ok := c.RecipeForRequestModel(modelName)
	return ok
}

// EntrypointRecipeDescription returns the model-listing description for an
// entrypoint's recipe: the recipe's own description when set, otherwise a
// generic label naming the recipe.
func (c *RouterConfig) EntrypointRecipeDescription(recipeName string) string {
	if recipe, ok := c.RecipeByName(recipeName); ok && strings.TrimSpace(recipe.Description) != "" {
		return recipe.Description
	}
	return fmt.Sprintf("Entrypoint for the %s routing recipe", recipeName)
}

// AllRoutingDecisions returns the decisions of every routing profile, for
// callers that reason about routing as a whole (signal usage analysis,
// contract validation). The result is always a fresh slice, never an alias of
// a recipe's own decisions, so callers cannot mutate config state by accident.
func (c *RouterConfig) AllRoutingDecisions() []Decision {
	if c == nil {
		return nil
	}
	total := 0
	for i := range c.Recipes {
		total += len(c.Recipes[i].Decisions)
	}
	if total == 0 {
		return nil
	}
	all := make([]Decision, 0, total)
	for i := range c.Recipes {
		all = append(all, c.Recipes[i].Decisions...)
	}
	return all
}

// HasRoutingDecisions reports whether any routing profile declares decisions,
// without the per-request allocation of AllRoutingDecisions.
func (c *RouterConfig) HasRoutingDecisions() bool {
	if c == nil {
		return false
	}
	for i := range c.Recipes {
		if len(c.Recipes[i].Decisions) > 0 {
			return true
		}
	}
	return false
}

// RoutingProfileSignals returns the default profile's signals for canonical
// export. The flat Signals field holds the global registry (union across
// recipes); exporting it would leak other recipes' rules into the top-level
// routing block.
func (c *RouterConfig) RoutingProfileSignals() Signals {
	if c == nil {
		return Signals{}
	}
	if recipe := c.DefaultRecipe(); recipe != nil {
		return recipe.Signals
	}
	return c.Signals
}

// RoutingProfileProjections is the projections counterpart of
// RoutingProfileSignals.
func (c *RouterConfig) RoutingProfileProjections() Projections {
	if c == nil {
		return Projections{}
	}
	if recipe := c.DefaultRecipe(); recipe != nil {
		return recipe.Projections
	}
	return c.Projections
}
