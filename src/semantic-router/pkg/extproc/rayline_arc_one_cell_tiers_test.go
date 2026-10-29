package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The one-cell invariant (docs/architecture/rayline-arc-one-cell-all-tiers.md):
// any ARC tier, in any public spelling, resolves on any cell to the recipe
// that serves it, on /v1/routes and on generation, priced or not, and every
// response says what decided it. A cell that does not serve the tier refuses;
// it never decides on another tier's recipe.

// tierStubSelector picks a fixed arm and answers every tier point a lookup
// asks with it, so each recipe's answer is distinguishable.
type tierStubSelector struct {
	stubARCSelector
	calls int
}

func (s *tierStubSelector) Select(ctx context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	s.calls++
	result, err := s.stubARCSelector.Select(ctx, selCtx)
	if err != nil {
		return nil, err
	}
	if controls := selCtx.RaylineARC.RouteControls; controls != nil {
		result.RaylineARC.RouteResults = map[string]selection.RaylineARCRouteResult{}
		for _, point := range controls.Points {
			result.RaylineARC.RouteResults[point.Name] = selection.RaylineARCRouteResult{Arm: s.arm}
		}
	}
	return result, nil
}

// oneCellArms: each recipe of the fixture picks its own arm.
var oneCellArms = map[config.RecipeName]int{config.DefaultRecipeName: 0, "arc-swift": 1, "arc-sage": 2}

func oneCellWorkers() []raylinearc.WorkerManifest {
	return []raylinearc.WorkerManifest{
		{ID: "w-default", Model: "vendor/default"},
		{ID: "w-swift", Model: "vendor/swift"},
		{ID: "w-sage", Model: "vendor/sage"},
	}
}

func oneCellDecision() config.Decision {
	return config.Decision{
		Name:      "rayline-arc",
		ModelRefs: []config.ModelRef{{Model: "w-default"}, {Model: "w-swift"}, {Model: "w-sage"}},
		Algorithm: &config.AlgorithmConfig{
			Type:    config.RaylineARCAlgorithmType,
			OnError: "fail_closed",
			RaylineARC: &config.RaylineARCAlgorithmConfig{
				Episode: config.RaylineARCEpisodeConfig{
					IDHeader: decisionServiceEpisodeHeader, AcquireTimeoutSeconds: 5, LeaseTTLSeconds: 30,
				},
				RoutesAPI: config.RaylineARCRoutesAPIConfig{
					Enabled: true, EpisodeWrites: true, Routing: &config.RaylineARCRoutesRoutingConfig{},
				},
			},
		},
	}
}

// oneCellRouter is a cell shaped like prod: "main" serves every tier as a
// recipe behind its versioned entrypoint; "sage" is a tier cell whose one
// (default) recipe is Sage, named by an auto model name.
func oneCellRouter(t *testing.T, shape string) (*OpenAIRouter, map[config.RecipeName]*tierStubSelector) {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{oneCellDecision()}
	recipes := []config.RecipeName{config.DefaultRecipeName}
	switch shape {
	case "main":
		cfg.AutoModelNames = []string{"rayline/arc-dev"}
		cfg.Recipes = []config.RoutingRecipe{
			{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Decisions: cfg.Decisions}},
			{Name: "arc-swift", Profile: config.RoutingProfile{Decisions: []config.Decision{oneCellDecision()}}},
			{Name: "arc-sage", Profile: config.RoutingProfile{Decisions: []config.Decision{oneCellDecision()}}},
		}
		cfg.Entrypoints = []config.EntrypointMapping{
			{ModelNames: []string{"arc-1.0"}, Recipe: config.DefaultRecipeName},
			{ModelNames: []string{"arc-swift-1.0"}, Recipe: "arc-swift"},
			{ModelNames: []string{"arc-sage-1.0"}, Recipe: "arc-sage"},
		}
		recipes = append(recipes, "arc-swift", "arc-sage")
	case "sage":
		cfg.AutoModelNames = []string{"rayline/arc-dev", "arc-sage-1.0"}
	default:
		t.Fatalf("unknown shape %q", shape)
	}
	newStore := func() raylinearc.EpisodeStore {
		store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 16, IdleTTL: time.Minute})
		if err != nil {
			t.Fatal(err)
		}
		return store
	}
	router := &OpenAIRouter{
		Config:                        cfg,
		RecipeModelSelectors:          map[config.RecipeName]*selection.Registry{},
		RaylineARCEpisodeStore:        newStore(),
		RaylineARCRecipeEpisodeStores: map[config.RecipeName]raylinearc.EpisodeStore{},
	}
	selectors := map[config.RecipeName]*tierStubSelector{}
	for _, recipe := range recipes {
		arm := oneCellArms[recipe]
		if shape == "sage" {
			arm = oneCellArms["arc-sage"]
		}
		selector := &tierStubSelector{stubARCSelector: stubARCSelector{arm: arm, workers: oneCellWorkers()}}
		registry := selection.NewRegistry()
		registry.Register(selection.MethodRaylineARC, selector)
		router.RecipeModelSelectors[recipe] = registry
		if recipe != config.DefaultRecipeName {
			router.RaylineARCRecipeEpisodeStores[recipe] = newStore()
		}
		selectors[recipe] = selector
	}
	router.ModelSelector = router.RecipeModelSelectors[config.DefaultRecipeName]
	return router, selectors
}

// oneCellSpellings are every public spelling of each tier.
var oneCellSpellings = map[string][]string{
	"arc-1.0":       {"arc-1.0", "arc-1", "rayline/arc-1", "rayline/arc-1.0"},
	"arc-swift-1.0": {"arc-swift", "rayline/arc-swift", "arc-swift-1.0", "rayline/arc-swift-1.0"},
	"arc-sage-1.0":  {"arc-sage", "rayline/arc-sage", "arc-sage-1.0", "rayline/arc-sage-1.0"},
}

// oneCellExpect is what a cell answers for a tier: the recipe that decides it
// and that recipe's arm, or "" for a tier the cell does not serve.
func oneCellExpect(shape, tier string) (config.RecipeName, int) {
	if shape == "main" {
		recipe := map[string]config.RecipeName{
			"arc-1.0": config.DefaultRecipeName, "arc-swift-1.0": "arc-swift", "arc-sage-1.0": "arc-sage",
		}[tier]
		return recipe, oneCellArms[recipe]
	}
	if tier == "arc-sage-1.0" {
		return config.DefaultRecipeName, oneCellArms["arc-sage"]
	}
	return "", -1
}

const oneCellPrices = `{"prices":{"w-default":{"input":1,"cache_read":0,"cache_write":1,"output":2},` +
	`"w-swift":{"input":1,"cache_read":0,"cache_write":1,"output":2},"w-sage":{"input":1,"cache_read":0,"cache_write":1,"output":2}}}`

func oneCellBody(model string, priced bool) []byte {
	body := `{"model":"` + model + `","max_tokens":64,"messages":[{"role":"user","content":"hi"}]`
	if priced {
		body += `,"routing":` + oneCellPrices
	}
	return []byte(body + `}`)
}

func identityHeadersOf(set map[string]string) [5]string {
	return [5]string{
		set[headers.RaylineTier], set[headers.RaylineRecipe], set[headers.RaylineArm],
		set[headers.RaylineModel], set[headers.RaylinePriceProfile],
	}
}

func immediateIdentity(response *ext_proc.ProcessingResponse) [5]string {
	set := map[string]string{}
	for _, name := range raylineARCIdentityHeaders {
		set[name] = immediateHeaderValue(response, name)
	}
	return identityHeadersOf(set)
}

func mutationIdentity(mutation *ext_proc.HeaderMutation) ([5]string, []string) {
	set := map[string]string{}
	for _, option := range mutation.GetSetHeaders() {
		set[option.GetHeader().GetKey()] = string(option.GetHeader().GetRawValue())
	}
	return identityHeadersOf(set), mutation.GetRemoveHeaders()
}

// The matrix: cell shape × tier × spelling × priced/unpriced × endpoint.
func TestOneCellResolvesEveryTierSpellingOnEveryEndpoint(t *testing.T) {
	workers := oneCellWorkers()
	for _, shape := range []string{"main", "sage"} {
		for tier, spellings := range oneCellSpellings {
			for _, model := range spellings {
				for _, priced := range []bool{false, true} {
					recipe, arm := oneCellExpect(shape, tier)
					name := fmt.Sprintf("%s/%s/priced=%t", shape, model, priced)

					t.Run(name+"/routes", func(t *testing.T) {
						router, selectors := oneCellRouter(t, shape)
						ctx := routesContext(nil)
						response := router.handleRaylineRoutesAPI(oneCellBody(model, priced), ctx)
						stateRaylineARCIdentity(response, ctx)
						if recipe == "" {
							status, errType, code := routesErrorCode(t, response)
							if status != 404 || errType != "not_found_error" || code != "model_not_served" {
								t.Fatalf("answered %d %s %s, want 404 model_not_served", status, errType, code)
							}
							for name, selector := range selectors {
								if selector.calls != 0 {
									t.Fatalf("recipe %s decided a tier this cell does not serve", name)
								}
							}
							return
						}
						if status := immediateStatusCode(t, response); status != 200 {
							t.Fatalf("status %d: %s", status, response.GetImmediateResponse().Body)
						}
						var payload raylineRoutesResponse
						if err := json.Unmarshal(response.GetImmediateResponse().Body, &payload); err != nil {
							t.Fatal(err)
						}
						if payload.Model != workers[arm].Model {
							t.Fatalf("decided %s, want %s (recipe %s)", payload.Model, workers[arm].Model, recipe)
						}
						for name, selector := range selectors {
							if want := map[bool]int{true: 1, false: 0}[name == recipe]; selector.calls != want {
								t.Fatalf("recipe %s selected %d times, want %d", name, selector.calls, want)
							}
						}
						profile := headers.RaylinePriceProfileCatalog
						if priced {
							profile = headers.RaylinePriceProfileInline
						}
						want := [5]string{tier, string(recipe), workers[arm].ID, workers[arm].Model, profile}
						if got := immediateIdentity(response); got != want {
							t.Fatalf("identity %v, want %v", got, want)
						}
					})

					t.Run(name+"/generation", func(t *testing.T) {
						router, _ := oneCellRouter(t, shape)
						ctx := &RequestContext{Headers: map[string]string{}}
						router.resolveEntrypointForRequest(model, ctx)
						if recipe == "" {
							// Not served here: generation keeps its old
							// passthrough, and states no ARC identity.
							if !ctx.Routing.IsPassthrough() || ctx.RaylineARCIdentity != nil ||
								router.requestModelActsAsAuto(model) {
								t.Fatalf("a tier this cell does not serve resolved to %q", ctx.Routing.RecipeName())
							}
							return
						}
						if ctx.Routing.RecipeName() != recipe || !router.requestModelActsAsAuto(model) {
							t.Fatalf("recipe %q, want %q", ctx.Routing.RecipeName(), recipe)
						}
						// A refusal before selection (prices on generation
						// are one) already says which tier refused it.
						refusal := router.createErrorResponse(400, "refused")
						stateRaylineARCIdentity(refusal, ctx)
						if got, want := immediateIdentity(refusal), [5]string{tier, string(recipe), "", "", "catalog"}; got != want {
							t.Fatalf("refusal identity %v, want %v", got, want)
						}
						// Selected and dispatched: success and upstream error
						// both state the arm and the served model.
						ctx.VSRRaylineARC = &selection.RaylineARCTrace{SelectedArm: arm}
						ctx.VSRSelectedModel = workers[arm].ID
						ctx.DispatchedOutbound = &dispatchedOutboundParams{Model: workers[arm].Model}
						for _, ok := range []bool{true, false} {
							got, _ := mutationIdentity(buildResponseHeaderMutation(ctx, ok))
							if want := [5]string{tier, string(recipe), workers[arm].ID, workers[arm].Model, "catalog"}; got != want {
								t.Fatalf("success=%t identity %v, want %v", ok, got, want)
							}
						}
					})
				}
			}
		}
	}
}

// A name outside the ARC namespace is not a tier: a lookup decides it on the
// default recipe as before, and an ARC-shaped name that is no ARC id is
// refused rather than decided on any recipe.
func TestOneCellNonTierNames(t *testing.T) {
	for model, want := range map[string]string{
		"claude-sonnet-5": "200", "rayline/arc-dev": "200", "arc-sage-2.0": "unknown_model", "arc": "unknown_model",
	} {
		router, selectors := oneCellRouter(t, "main")
		response := router.handleRaylineRoutesAPI(oneCellBody(model, false), routesContext(nil))
		if want == "200" {
			if status := immediateStatusCode(t, response); status != 200 || selectors[config.DefaultRecipeName].calls != 1 {
				t.Fatalf("%s: status %d, default calls %d", model, status, selectors[config.DefaultRecipeName].calls)
			}
			continue
		}
		if status, _, code := routesErrorCode(t, response); status != 400 || code != want {
			t.Fatalf("%s: %d %s, want 400 %s", model, status, code, want)
		}
	}
}

// Generation keeps every configured name's recipe exactly as before.
func TestOneCellGenerationKeepsConfiguredNames(t *testing.T) {
	router, _ := oneCellRouter(t, "main")
	for model, want := range map[string]config.RecipeName{
		"arc-1.0": config.DefaultRecipeName, "arc-swift-1.0": "arc-swift", "arc-sage-1.0": "arc-sage",
		"rayline/arc-dev": config.DefaultRecipeName,
	} {
		ctx := &RequestContext{Headers: map[string]string{}}
		router.resolveEntrypointForRequest(model, ctx)
		recipe, ok := router.Config.RecipeForRoutingModel(model)
		if !ok || ctx.Routing.RecipeName() != want || recipe.Name != want {
			t.Fatalf("%s: recipe %q, want %q", model, ctx.Routing.RecipeName(), want)
		}
	}
	// A concrete arm name stays passthrough, with no ARC identity.
	ctx := &RequestContext{Headers: map[string]string{}}
	router.resolveEntrypointForRequest("w-sage", ctx)
	if !ctx.Routing.IsPassthrough() || ctx.RaylineARCIdentity != nil {
		t.Fatal("a concrete model resolved to a recipe")
	}
	_, removed := mutationIdentity(buildResponseHeaderMutation(ctx, true))
	for _, name := range raylineARCIdentityHeaders {
		if !slices.Contains(removed, name) {
			t.Fatalf("%s is not stripped from a non-ARC response", name)
		}
	}
}

// A priced lookup joins no episode, whatever episode_writes says: the
// caller's price world never becomes trajectory state. The same lookup
// unpriced does join the conversation's episode.
func TestOneCellPricedLookupIsEphemeral(t *testing.T) {
	for _, priced := range []bool{false, true} {
		router, _ := oneCellRouter(t, "main")
		ctx := routesContext(map[string]string{raylineRoutesSessionHeader: "conversation-1"})
		response := router.handleRaylineRoutesAPI(oneCellBody("arc-sage", priced), ctx)
		if status := immediateStatusCode(t, response); status != 200 {
			t.Fatalf("priced=%t status %d: %s", priced, status, response.GetImmediateResponse().Body)
		}
		var payload raylineRoutesResponse
		if err := json.Unmarshal(response.GetImmediateResponse().Body, &payload); err != nil {
			t.Fatal(err)
		}
		joined := payload.Episode != nil
		warned := slices.ContainsFunc(payload.Warnings, func(w string) bool { return w == raylineRoutesPricesUntracked })
		if joined == priced || warned != priced {
			t.Fatalf("priced=%t: joined episode %t, prices_untracked warning %t", priced, joined, warned)
		}
	}
}

// Both unknown_price_model forms carry error.models; the service's caller
// price refusals answer their own codes; legs are bounded at 1e9.
func TestOneCellPriceRefusals(t *testing.T) {
	router, _ := oneCellRouter(t, "main")
	body := []byte(`{"model":"arc-sage","messages":[{"role":"user","content":"hi"}],"routing":{"prices":{` +
		`"acme/b":{"input":1,"cache_read":0,"cache_write":0,"output":1},"acme/a":{"input":1,"cache_read":0,"cache_write":0,"output":1}}}}`)
	response := router.handleRaylineRoutesAPI(body, routesContext(nil))
	var envelope struct {
		Error struct {
			Code   string   `json:"code"`
			Models []string `json:"models"`
		} `json:"error"`
	}
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Error.Code != "unknown_price_model" || !slices.Equal(envelope.Error.Models, []string{"acme/a", "acme/b"}) {
		t.Fatalf("cell form %+v", envelope.Error)
	}

	priced := []string{"m/a", "m/b"}
	if got := routeRefusedPriceModels(map[string]any{"models": []any{"m/b", "elsewhere/x", 7}}, priced); !slices.Equal(got, []string{"m/b"}) {
		t.Fatalf("service form names %v", got)
	}
	if got := routeRefusedPriceModels(map[string]any{}, priced); got == nil || len(got) != 0 {
		t.Fatalf("service form without names %#v, want an empty list", got)
	}
	refused := routesUnknownPriceModel("x", raylineRoutesRefusedPriceModels(selectionFailureError(arcSelectionFailure(arcFailureRoutesUnknownPriceModel))))
	if refused.models == nil {
		t.Fatal("the service form carries no error.models")
	}

	for reason, code := range map[string]string{
		"caller_prices_unsupported":  "caller_prices_unsupported",
		"caller_prices_inconsistent": "caller_prices_inconsistent",
	} {
		class := routeCallerPricesRefusal(map[string]any{"reason": reason})
		answer := router.raylineRoutesFailure(routesContext(nil), context.Background(), "rt", selectionFailureError(arcSelectionFailure(class)))
		if status, _, got := routesErrorCode(t, answer); status != 400 || got != code {
			t.Fatalf("%s answered %d %s", reason, status, got)
		}
	}
	if routeCallerPricesRefusal(map[string]any{"reason": "media_refused"}) != "" {
		t.Fatal("an unrelated unsupported_request was taken for a caller price refusal")
	}

	for value, ok := range map[string]bool{"1000000000": true, "1000000001": false, "1e300": false} {
		_, refusal := routesReadPrices(json.RawMessage(`{"m/a":{"input":` + value + `,"cache_read":0,"cache_write":0,"output":1}}`))
		if (refusal == nil) != ok || (!ok && refusal.code != "price_leg_invalid") {
			t.Fatalf("leg %s: refusal %v", value, refusal)
		}
	}
}

// Resolution edges: a configured spelling with the rayline/ prefix still
// serves every other spelling of its tier; arc-1.0 on a configurable cell
// named only by an alias is that cell's default recipe and says arc-1.0; on a
// cell with no ARC decision, a tier spelling resolves nowhere.
func TestOneCellResolutionEdges(t *testing.T) {
	prefixed := &config.RouterConfig{}
	prefixed.Decisions = []config.Decision{oneCellDecision()}
	prefixed.AutoModelNames = []string{"rayline/arc-sage-1.0"}
	for _, model := range []string{"arc-sage", "arc-sage-1.0", "rayline/arc-sage"} {
		recipe, ok := raylineARCRecipeForModel(prefixed, model)
		if !ok || recipe.Name != config.DefaultRecipeName {
			t.Fatalf("prefixed Sage cell: %s resolved %v", model, ok)
		}
		if identity := raylineARCIdentityFor(prefixed, recipe, model); identity.tier != "arc-sage-1.0" {
			t.Fatalf("prefixed Sage cell: %s tier %q", model, identity.tier)
		}
	}
	if _, ok := raylineARCRecipeForModel(prefixed, "arc-swift"); ok {
		t.Fatal("prefixed Sage cell served Swift")
	}

	configurable := &config.RouterConfig{}
	configurable.Decisions = []config.Decision{oneCellDecision()}
	configurable.AutoModelNames = []string{"rayline/arc-dev"}
	recipe, ok := raylineARCRecipeForModel(configurable, "arc-1")
	if !ok || recipe.Name != config.DefaultRecipeName {
		t.Fatal("configurable cell refused arc-1")
	}
	for _, model := range []string{"arc-1", "rayline/arc-dev"} {
		if identity := raylineARCIdentityFor(configurable, recipe, model); identity.tier != "arc-1.0" {
			t.Fatalf("configurable cell called %s: tier %q, want arc-1.0", model, identity.tier)
		}
	}

	plain := &config.RouterConfig{}
	plain.Decisions = []config.Decision{{Name: "static", Algorithm: &config.AlgorithmConfig{Type: "static"}}}
	plain.AutoModelNames = []string{"auto"}
	router := &OpenAIRouter{Config: plain}
	for _, model := range []string{"arc-1.0", "arc-1", "arc-sage"} {
		if _, ok := raylineARCRecipeForModel(plain, model); ok || router.requestModelActsAsAuto(model) {
			t.Fatalf("a non-ARC cell resolved %s", model)
		}
	}
}

// A generation turn refused for its routing block, before its recipe is
// resolved, still says which tier refused it.
func TestOneCellGenerationRoutingRefusalStatesTheTier(t *testing.T) {
	router, _ := oneCellRouter(t, "main")
	router.Config.Decisions[0].Algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{
		GenerationRouting: &config.RaylineARCGenerationRoutingConfig{},
	}
	ctx := &RequestContext{Headers: map[string]string{":path": "/v1/messages", ":method": "POST"}}
	refusal, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{
		Body: []byte(`{"model":"arc-sage-1.0","messages":[{"role":"user","content":"hi"}],"routing":` + oneCellPrices + `}`),
	}}, ctx)
	if err != nil || refusal.GetImmediateResponse() == nil {
		t.Fatalf("prices on generation were not refused: %v", err)
	}
	stateRaylineARCIdentity(refusal, ctx)
	if got, want := immediateIdentity(refusal), [5]string{"arc-sage-1.0", "arc-sage", "", "", "catalog"}; got != want {
		t.Fatalf("refusal identity %v, want %v", got, want)
	}
}

// Every immediate answer on the generation path states the tier: a refusal
// from ingress validation, before the recipe is resolved, and a re-send's
// own error.
func TestOneCellGenerationEarlyRefusalsStateTheTier(t *testing.T) {
	router, _ := oneCellRouter(t, "main")
	ctx := &RequestContext{Headers: map[string]string{":path": "/v1/messages"}}
	router.stateGenerationIdentity("arc-swift-1.0", ctx)
	refusal := router.createErrorResponse(400, "thread continuation refused")
	stateRaylineARCIdentity(refusal, ctx)
	if got, want := immediateIdentity(refusal), [5]string{"arc-swift-1.0", "arc-swift", "", "", "catalog"}; got != want {
		t.Fatalf("early refusal identity %v, want %v", got, want)
	}
	// A concrete model states nothing.
	other := &RequestContext{Headers: map[string]string{}}
	router.stateGenerationIdentity("w-sage", other)
	if other.RaylineARCIdentity != nil {
		t.Fatal("a concrete model got an ARC identity")
	}
}

// No ARC identity where no ARC selection stands: a non-ARC decision that won
// on a recipe holding an ARC one, and an internal looper request.
func TestOneCellIdentityOnlyForARCDecisions(t *testing.T) {
	router, _ := oneCellRouter(t, "main")
	ctx := &RequestContext{Headers: map[string]string{}}
	router.resolveEntrypointForRequest("arc-1.0", ctx)
	ctx.VSRSelectedDecision = &config.Decision{Name: "keyword", Algorithm: &config.AlgorithmConfig{Type: "static"}}
	if got, removed := mutationIdentity(buildResponseHeaderMutation(ctx, true)); got != ([5]string{}) || len(removed) == 0 {
		t.Fatalf("a non-ARC decision stated %v", got)
	}
	// No decision matched: the default model served, not ARC.
	fallback := &RequestContext{Headers: map[string]string{}}
	router.resolveEntrypointForRequest("arc-sage-1.0", fallback)
	fallback.DispatchedOutbound = &dispatchedOutboundParams{Model: "vendor/default-model"}
	if got, _ := mutationIdentity(buildResponseHeaderMutation(fallback, true)); got != ([5]string{}) {
		t.Fatalf("a default-model fallback stated %v", got)
	}
	breach := router.createErrorResponse(502, "upstream")
	stateRaylineARCIdentity(breach, fallback)
	if got := immediateIdentity(breach); got != ([5]string{}) {
		t.Fatalf("a default-model fallback error stated %v", got)
	}
	looper := &RequestContext{Headers: map[string]string{}, LooperRequest: true}
	router.stateGenerationIdentity("arc-sage-1.0", looper)
	refusal := router.createErrorResponse(400, "refused")
	stateRaylineARCIdentity(refusal, looper)
	if got := immediateIdentity(refusal); got != ([5]string{}) {
		t.Fatalf("a looper request stated %v", got)
	}
}
