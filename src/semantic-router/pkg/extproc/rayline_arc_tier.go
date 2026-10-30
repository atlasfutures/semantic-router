package extproc

import (
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// One cell serves every ARC tier (docs/architecture/rayline-arc-one-cell-all-tiers.md).
// A request names its tier in any public spelling; the cell resolves that
// spelling to the recipe that serves the tier and decides there, on every
// endpoint, or refuses. It never decides one tier's request on another
// tier's recipe.

// raylineARCCanonicalTierIDs maps every public spelling of an ARC tier, with
// any rayline/ prefix removed, to its canonical versioned id.
var raylineARCCanonicalTierIDs = map[string]string{
	"arc-1": "arc-1.0", "arc-1.0": "arc-1.0",
	"arc-swift": "arc-swift-1.0", "arc-swift-1.0": "arc-swift-1.0",
	"arc-sage": "arc-sage-1.0", "arc-sage-1.0": "arc-sage-1.0",
}

// raylineARCCanonicalTier is the canonical versioned tier id a model name
// spells, or "" for a name that spells no public tier.
func raylineARCCanonicalTier(model string) string {
	return raylineARCCanonicalTierIDs[strings.TrimPrefix(strings.TrimSpace(model), "rayline/")]
}

// raylineARCShapedModel reports whether a name is in the ARC namespace (arc,
// arc-*, arc@*, with or without rayline/): one a cell must either serve as
// the tier it names or refuse, never decide as some other tier.
func raylineARCShapedModel(model string) bool {
	bare := strings.TrimPrefix(strings.TrimSpace(model), "rayline/")
	return bare == "arc" || strings.HasPrefix(bare, "arc-") || strings.HasPrefix(bare, "arc@")
}

// raylineARCTierOfRecipe names the public tier a recipe serves on this cell:
// the first canonical tier id among the names that select it (its
// entrypoints and, for the default recipe, the auto model names). "" when no
// public tier name selects it.
func raylineARCTierOfRecipe(cfg *config.RouterConfig, recipe *config.RoutingRecipe) string {
	if cfg == nil || recipe == nil {
		return ""
	}
	var names []string
	for _, entrypoint := range cfg.Entrypoints {
		if entrypoint.Recipe == recipe.Name {
			names = append(names, entrypoint.ModelNames...)
		}
	}
	if recipe.Name == config.DefaultRecipeName {
		names = append(names, cfg.EffectiveAutoModelNames()...)
	}
	for _, name := range names {
		if tier := raylineARCCanonicalTier(name); tier != "" {
			return tier
		}
	}
	// An ARC default recipe no tier name selects is configurable ARC: the
	// one tier that is not fixed (a cell named only by an alias such as
	// rayline/arc-dev).
	if recipe.Name == config.DefaultRecipeName && recipeServesRaylineARC(recipe) {
		return "arc-1.0"
	}
	return ""
}

// raylineARCRecipeForModel resolves a request's model name to the recipe that
// decides it, the same way on every endpoint and cell:
//
//   - a name the config selects (an auto name or an entrypoint) selects its
//     recipe, as generation does;
//   - any other spelling of a public tier (arc-sage, rayline/arc-sage-1.0, ...)
//     selects the ARC recipe a configured spelling of the same tier selects;
//   - arc-1.0 on a cell whose ARC default recipe serves no other tier is that
//     default recipe (a single-recipe configurable cell);
//   - any other ARC-shaped name, and a tier this cell does not serve, is
//     refused (ok false);
//   - a name outside the ARC namespace is not a tier: the default recipe, as
//     a lookup has always read it.
func raylineARCRecipeForModel(cfg *config.RouterConfig, model string) (*config.RoutingRecipe, bool) {
	if cfg == nil {
		return nil, false
	}
	model = strings.TrimSpace(model)
	if recipe, ok := cfg.RecipeForRoutingModel(model); ok && recipe != nil {
		return recipe, true
	}
	canonical := raylineARCCanonicalTier(model)
	if canonical != "" {
		if recipe := raylineARCRecipeForTier(cfg, canonical); recipe != nil && recipeServesRaylineARC(recipe) {
			return recipe, true
		}
		if canonical == "arc-1.0" {
			recipe := cfg.DefaultRecipe()
			if tier := raylineARCTierOfRecipe(cfg, recipe); recipeServesRaylineARC(recipe) && (tier == "" || tier == canonical) {
				return recipe, true
			}
		}
		return nil, false
	}
	if raylineARCShapedModel(model) {
		return nil, false
	}
	recipe := cfg.DefaultRecipe()
	return recipe, recipe != nil
}

// raylineARCRecipeForTier is the recipe a configured spelling of the tier
// selects (an entrypoint name, or an auto name for the default recipe), or
// nil where none does.
func raylineARCRecipeForTier(cfg *config.RouterConfig, tier string) *config.RoutingRecipe {
	for _, entrypoint := range cfg.Entrypoints {
		for _, name := range entrypoint.ModelNames {
			if raylineARCCanonicalTier(name) != tier {
				continue
			}
			if entrypoint.Recipe == config.DefaultRecipeName {
				return cfg.DefaultRecipe()
			}
			if recipe, ok := cfg.RecipeByName(entrypoint.Recipe); ok {
				return recipe
			}
		}
	}
	for _, name := range cfg.EffectiveAutoModelNames() {
		if raylineARCCanonicalTier(name) == tier {
			return cfg.DefaultRecipe()
		}
	}
	return nil
}

// raylineARCIdentity is what the identity headers state about one ARC
// response. A member the cell does not know stays empty and its header is not
// sent.
type raylineARCIdentity struct {
	tier         string
	recipe       string
	arm          string
	model        string
	priceProfile string
}

// raylineARCIdentityHeaders are the identity headers in response order.
var raylineARCIdentityHeaders = []string{
	headers.RaylineTier, headers.RaylineRecipe, headers.RaylineArm, headers.RaylineModel, headers.RaylinePriceProfile,
}

func (identity raylineARCIdentity) values() []string {
	return []string{identity.tier, identity.recipe, identity.arm, identity.model, identity.priceProfile}
}

// raylineARCIdentityFor starts the identity of a request decided by recipe:
// its tier and recipe, in the catalog price world until the caller's prices
// are applied.
//
// The tier is the one the recipe's configured names spell, else the one the
// request spelled (arc-1.0 on a configurable cell named only by an alias).
func raylineARCIdentityFor(cfg *config.RouterConfig, recipe *config.RoutingRecipe, model string) *raylineARCIdentity {
	if recipe == nil {
		return nil
	}
	tier := raylineARCTierOfRecipe(cfg, recipe)
	if tier == "" {
		tier = raylineARCCanonicalTier(model)
	}
	return &raylineARCIdentity{
		tier:         tier,
		recipe:       string(recipe.Name),
		priceProfile: headers.RaylinePriceProfileCatalog,
	}
}

// recipeServesRaylineARC reports whether a recipe decides with Rayline ARC.
func recipeServesRaylineARC(recipe *config.RoutingRecipe) bool {
	if recipe == nil {
		return false
	}
	for index := range recipe.Profile.Decisions {
		algorithm := recipe.Profile.Decisions[index].Algorithm
		if raylineARCSelection(algorithm) && algorithm.RaylineARC != nil {
			return true
		}
	}
	return false
}

// raylineARCGenerationIdentity is the identity a response states: as the
// lookup recorded it, or for a generation turn the recipe its model selected,
// the arm ARC chose and the provider model the call was dispatched to.
// Generation never takes caller prices, so its price world is always the
// catalog. Nil for a request no ARC recipe decides.
func raylineARCGenerationIdentity(ctx *RequestContext) *raylineARCIdentity {
	if ctx == nil || ctx.RaylineARCIdentity == nil || ctx.LooperRequest {
		return nil
	}
	// A recipe may hold other decisions beside its ARC one: when one of
	// those decided the turn, no ARC selection stands behind the response.
	if decision := ctx.VSRSelectedDecision; decision != nil && !raylineARCSelection(decision.Algorithm) {
		return nil
	}
	// Nor when a body went out that ARC did not choose (the default-model
	// fallback when no decision matched).
	if ctx.DispatchedOutbound != nil && ctx.VSRRaylineARC == nil {
		return nil
	}
	identity := *ctx.RaylineARCIdentity
	if ctx.VSRRaylineARC != nil {
		identity.arm = strings.TrimSpace(ctx.VSRSelectedModel)
	}
	if ctx.DispatchedOutbound != nil {
		identity.model = strings.TrimSpace(ctx.DispatchedOutbound.Model)
	}
	return &identity
}

// addRaylineARCIdentity states a generation turn's identity on its upstream
// response, replacing any copy the upstream sent, and removes the headers
// from every response no ARC recipe decided.
func (builder *responseHeaderMutationBuilder) addRaylineARCIdentity(ctx *RequestContext) {
	identity := raylineARCGenerationIdentity(ctx)
	values := make([]string, len(raylineARCIdentityHeaders))
	// An upstream response is an ARC turn's only when ARC selected its arm.
	if identity != nil && ctx.VSRRaylineARC != nil {
		values = identity.values()
	}
	for index, name := range raylineARCIdentityHeaders {
		if values[index] == "" {
			builder.removeHeaders = append(builder.removeHeaders, name)
			continue
		}
		builder.addString(name, values[index])
		last := builder.setHeaders[len(builder.setHeaders)-1]
		last.AppendAction = core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD
	}
}

// stateRaylineARCIdentity puts the request's ARC identity on an immediate
// response the cell answers itself (a lookup, a refusal, a selection
// failure), so an error states what decided it as a success does. Any other
// response passes through unchanged.
func stateRaylineARCIdentity(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if response == nil || response.GetImmediateResponse() == nil {
		return
	}
	identity := raylineARCGenerationIdentity(ctx)
	if identity == nil {
		return
	}
	immediate := response.GetImmediateResponse()
	if immediate.Headers == nil {
		immediate.Headers = &ext_proc.HeaderMutation{}
	}
	for index, name := range raylineARCIdentityHeaders {
		value := identity.values()[index]
		if value == "" {
			continue
		}
		replaced := false
		for _, existing := range immediate.Headers.SetHeaders {
			if existing.GetHeader().GetKey() == name {
				existing.Header.RawValue, existing.Header.Value = []byte(value), ""
				replaced = true
			}
		}
		if !replaced {
			immediate.Headers.SetHeaders = append(immediate.Headers.SetHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{Key: name, RawValue: []byte(value)},
			})
		}
	}
}
