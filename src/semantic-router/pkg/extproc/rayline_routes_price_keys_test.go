//go:build !windows && cgo

package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// remapScorer is the shipped deepseek shape: the steered flash actions are
// sent on flash-1017, a twin checkpoint that serves the same package model
// and is bound to that model's native action, beside an unremapped opus.
func remapScorer() *policyServiceScorer {
	return &policyServiceScorer{
		workerIDs: []string{"flash", "flash-1017", "opus"},
		workers: []raylinearc.WorkerManifest{
			{ID: "flash", Model: "deepseek/deepseek-v4.1-flash"},
			{ID: "flash-1017", Model: "deepseek/deepseek-v4.1-flash-1017"},
			{ID: "opus", Model: "anthropic/claude-opus-5"},
		},
		actionOrder: []string{"flash-up", "flash-down", "flash-native", "opus-max"},
		bindings: map[string]policyBinding{
			"flash-up":     {arm: 0, dispatchArm: 1, model: "deepseek-v4.1-flash"},
			"flash-down":   {arm: 0, dispatchArm: 1, model: "deepseek-v4.1-flash"},
			"flash-native": {arm: 1, dispatchArm: 1, model: "deepseek-v4.1-flash"},
			"opus-max":     {arm: 2, dispatchArm: 2, model: "claude-opus-5"},
		},
	}
}

// unboundTwinScorer is remapScorer with no action bound to flash-1017.
func unboundTwinScorer() *policyServiceScorer {
	scorer := remapScorer()
	scorer.actionOrder = []string{"flash-up", "flash-down", "opus-max"}
	delete(scorer.bindings, "flash-native")
	return scorer
}

func callerRate(input, output float64) raylinearc.PolicyRoutingRate {
	zero, cacheWrite := 0.0, input
	return raylinearc.PolicyRoutingRate{Input: &input, CacheRead: &zero, CacheWrite: &cacheWrite, Output: &output}
}

func callerWorld(rates map[string]raylinearc.PolicyRoutingRate) *raylinearc.PolicyRoutingPrices {
	return &raylinearc.PolicyRoutingPrices{
		Schema: raylinearc.RoutingPricesSchema, StoreVersion: "caller:h", AsOf: "2026-11-10T00:00:00Z",
		Rates: rates, Basis: raylinearc.RoutingPricesBasisCaller,
	}
}

func rateKeys(prices *raylinearc.PolicyRoutingPrices) []string {
	keys := make([]string, 0, len(prices.Rates))
	for key := range prices.Rates {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

// A remapped arm's price, keyed by the caller at the worker it is served on,
// reaches the service under the package model's id, whether or not the twin
// serves an action of its own; an unremapped key goes out as sent; the served
// id no arm reads is not sent.
func TestRouteServicePricesKeyARemappedArmByItsOwnID(t *testing.T) {
	for name, scorer := range map[string]*policyServiceScorer{"bound twin": remapScorer(), "unbound twin": unboundTwinScorer()} {
		t.Run(name, func(t *testing.T) { checkRemappedArmKeyedByItsOwnID(t, scorer) })
	}
}

func checkRemappedArmKeyedByItsOwnID(t *testing.T, scorer *policyServiceScorer) {
	t.Helper()
	caller := callerWorld(map[string]raylinearc.PolicyRoutingRate{
		"deepseek/deepseek-v4.1-flash-1017": callerRate(0.2, 0.8),
		"anthropic/claude-opus-5":           callerRate(5, 25),
	})
	sent := routeServicePrices(scorer, caller)
	if got := rateKeys(sent); !slices.Equal(got, []string{"anthropic/claude-opus-5", "deepseek/deepseek-v4.1-flash"}) {
		t.Fatalf("sent keys = %v, want the arm's own id for the remapped worker", got)
	}
	if routeRateLegs(sent.Rates["deepseek/deepseek-v4.1-flash"]) != [4]float64{0.2, 0, 0.2, 0.8} ||
		routeRateLegs(sent.Rates["anthropic/claude-opus-5"]) != [4]float64{5, 0, 5, 25} {
		t.Fatalf("sent rates = %v", sent.Rates)
	}
	if len(caller.Rates) != 2 || caller.Rates["deepseek/deepseek-v4.1-flash-1017"].Input == nil {
		t.Fatal("the caller's world was changed in place")
	}
	if sent.StoreVersion != caller.StoreVersion || sent.Basis != caller.Basis || sent.Exclusive != caller.Exclusive {
		t.Fatalf("sent %+v lost the world's identity", sent)
	}

	// The decide request carries the translated world.
	request := &raylinearc.PolicyDecisionRequest{}
	applyRouteControls(request, scorer, &selection.RaylineARCRouteControls{Prices: caller})
	if got := rateKeys(request.RoutingPrices); !slices.Equal(got, []string{"anthropic/claude-opus-5", "deepseek/deepseek-v4.1-flash"}) {
		t.Fatalf("decide request keys = %v", got)
	}
}

// Without a remapped arm whose served worker the caller priced, the world
// goes out exactly as the caller sent it.
func TestRouteServicePricesLeaveAnUnremappedWorldAsSent(t *testing.T) {
	plain := remapScorer()
	for _, actionID := range []string{"flash-up", "flash-down"} {
		plain.bindings[actionID] = policyBinding{arm: 0, dispatchArm: 0, model: "deepseek-v4.1-flash"}
	}
	plain.bindings["flash-native"] = policyBinding{arm: 1, dispatchArm: 1, model: "deepseek-v4.1-flash-1017"}
	for name, tc := range map[string]struct {
		scorer *policyServiceScorer
		rates  map[string]raylinearc.PolicyRoutingRate
	}{
		"no remap": {plain, map[string]raylinearc.PolicyRoutingRate{"deepseek/deepseek-v4.1-flash": callerRate(0.2, 0.8)}},
		"no remap, both ids": {plain, map[string]raylinearc.PolicyRoutingRate{
			"deepseek/deepseek-v4.1-flash": callerRate(0.2, 0.8), "deepseek/deepseek-v4.1-flash-1017": callerRate(0.1, 0.4),
		}},
		"remap, served unpriced": {remapScorer(), map[string]raylinearc.PolicyRoutingRate{"anthropic/claude-opus-5": callerRate(5, 25)}},
		// A caller still pricing the package model's own id is read as before.
		"remap, package id priced": {remapScorer(), map[string]raylinearc.PolicyRoutingRate{"deepseek/deepseek-v4.1-flash": callerRate(0.2, 0.8)}},
	} {
		t.Run(name, func(t *testing.T) {
			caller := callerWorld(tc.rates)
			if sent := routeServicePrices(tc.scorer, caller); sent != caller {
				t.Fatalf("sent %+v, want the caller's world itself", sent)
			}
		})
	}
}

// When the to-worker is itself an arm (glm sent on a bound opus), the opus
// price prices both arms: each under its own id.
func TestRouteServicePricesKeepABoundToWorkersOwnKey(t *testing.T) {
	scorer := &policyServiceScorer{
		workers:     []raylinearc.WorkerManifest{{ID: "glm", Model: "z-ai/glm-5"}, {ID: "opus", Model: "anthropic/claude-opus-5"}},
		actionOrder: []string{"glm-up", "opus-max"},
		bindings: map[string]policyBinding{
			"glm-up": {arm: 0, dispatchArm: 1, model: "glm-5"}, "opus-max": {arm: 1, dispatchArm: 1, model: "claude-opus-5"},
		},
	}
	sent := routeServicePrices(scorer, callerWorld(map[string]raylinearc.PolicyRoutingRate{"anthropic/claude-opus-5": callerRate(5, 25)}))
	if !slices.Equal(rateKeys(sent), []string{"anthropic/claude-opus-5", "z-ai/glm-5"}) ||
		routeRateLegs(sent.Rates["z-ai/glm-5"]) != routeRateLegs(sent.Rates["anthropic/claude-opus-5"]) {
		t.Fatalf("sent %v", sent)
	}
}

// A dispatched id that another remapped arm reads as its own package id is
// kept for that arm: remaps a -> b and b -> c over distinct workers, with
// only b priced, send b's price under a and under b.
func TestRouteServicePricesKeepADispatchedIDAnotherArmReads(t *testing.T) {
	scorer := &policyServiceScorer{
		workers: []raylinearc.WorkerManifest{
			{ID: "a", Model: "v/a"}, {ID: "b", Model: "v/b"}, {ID: "c", Model: "v/c"},
		},
		actionOrder: []string{"a-up", "b-up"},
		bindings: map[string]policyBinding{
			"a-up": {arm: 0, dispatchArm: 1, model: "m-a"}, "b-up": {arm: 1, dispatchArm: 2, model: "m-b"},
		},
	}
	sent := routeServicePrices(scorer, callerWorld(map[string]raylinearc.PolicyRoutingRate{"v/b": callerRate(0.2, 0.8)}))
	if !slices.Equal(rateKeys(sent), []string{"v/a", "v/b"}) ||
		routeRateLegs(sent.Rates["v/a"]) != routeRateLegs(sent.Rates["v/b"]) {
		t.Fatalf("sent %v, want v/b's price under v/a and v/b", sent.Rates)
	}
	// With c priced too, b's arm takes its dispatched worker's price.
	sent = routeServicePrices(scorer, callerWorld(map[string]raylinearc.PolicyRoutingRate{
		"v/b": callerRate(0.2, 0.8), "v/c": callerRate(0.4, 1.6),
	}))
	if !slices.Equal(rateKeys(sent), []string{"v/a", "v/b"}) || routeRateLegs(sent.Rates["v/b"]) != [4]float64{0.4, 0, 0.4, 1.6} {
		t.Fatalf("sent %v", sent.Rates)
	}
}

// Both ids priced is never refused: the dispatched id's price, what the call
// is billed at, wins for every arm dispatched to it, whether the two are
// priced unlike or alike; the package id alone is read as before.
func TestRouteServicePricesPreferTheDispatchedID(t *testing.T) {
	served, own := "deepseek/deepseek-v4.1-flash-1017", "deepseek/deepseek-v4.1-flash"
	for name, ownRate := range map[string]raylinearc.PolicyRoutingRate{
		"priced unlike": callerRate(0.3, 1.2), "priced alike": callerRate(0.2, 0.8),
	} {
		t.Run(name, func(t *testing.T) {
			sent := routeServicePrices(remapScorer(), callerWorld(map[string]raylinearc.PolicyRoutingRate{
				served: callerRate(0.2, 0.8), own: ownRate, "anthropic/claude-opus-5": callerRate(5, 25),
			}))
			if !slices.Equal(rateKeys(sent), []string{"anthropic/claude-opus-5", own}) ||
				routeRateLegs(sent.Rates[own]) != [4]float64{0.2, 0, 0.2, 0.8} {
				t.Fatalf("sent %v, want %s at %s's price", sent.Rates, own, served)
			}
		})
	}
	// An unremapped arm on the package id: the dispatched id's price still
	// wins, and nothing is refused.
	scorer := remapScorer()
	scorer.workers = append(scorer.workers, raylinearc.WorkerManifest{ID: "flash-twin", Model: own})
	scorer.actionOrder = append(scorer.actionOrder, "twin-up")
	scorer.bindings["twin-up"] = policyBinding{arm: 3, dispatchArm: 3, model: "deepseek-v4.1-flash-0910"}
	sent := routeServicePrices(scorer, callerWorld(map[string]raylinearc.PolicyRoutingRate{served: callerRate(0.2, 0.8), own: callerRate(0.3, 1.2)}))
	if !slices.Equal(rateKeys(sent), []string{own}) || routeRateLegs(sent.Rates[own]) != [4]float64{0.2, 0, 0.2, 0.8} {
		t.Fatalf("sent %v", sent.Rates)
	}
}

// routing.exclusive keeps a remapped action whose served worker the caller
// priced, and the answer is checked against the hash of the world sent.
func TestRouteControlsOnARemappedArmKeepItAndCheckTheSentWorld(t *testing.T) {
	scorer := remapScorer()
	served := "deepseek/deepseek-v4.1-flash-1017"
	if got := routeExclusiveOffer(scorer, scorer.actionOrder, []string{served}); !slices.Equal(got, []string{"flash-up", "flash-down", "flash-native"}) {
		t.Fatalf("exclusive offer = %v, want the remapped actions", got)
	}
	prices := map[string][4]float64{served: {0.2, 0, 0.2, 0.8}}
	controls := &selection.RaylineARCRouteControls{
		Prices:       callerWorld(map[string]raylinearc.PolicyRoutingRate{served: callerRate(0.2, 0.8)}),
		PricedModels: []string{served}, Exclusive: true, PriceSHA256: routesPriceSHA256(prices, true),
	}
	controls.Prices.Exclusive = true
	sentSHA := routesPriceSHA256(map[string][4]float64{"deepseek/deepseek-v4.1-flash": {0.2, 0, 0.2, 0.8}}, true)
	echo := func(sha string) *raylinearc.PolicyDecisionResponse {
		exclusive, mode := true, raylinearc.RoutingPricesModeCaller
		response := &raylinearc.PolicyDecisionResponse{RoutingPricesUsed: &raylinearc.PolicyRoutingPricesUsed{
			Mode: mode, PricesSHA256: &sha, Exclusive: &exclusive, FallbackArms: []string{}, FilledLegs: map[string][]string{},
		}}
		return response
	}
	if failure := recordRouteControls(&selection.RaylineARCTrace{}, scorer, controls, echo(sentSHA), scorer.actionOrder, false); failure != "" {
		t.Fatalf("the sent world's echo = %q", failure)
	}
	if failure := recordRouteControls(&selection.RaylineARCTrace{}, scorer, controls, echo(controls.PriceSHA256), scorer.actionOrder, false); failure != arcFailureRoutesPricesNotApplied {
		t.Fatalf("an echo of a world not sent = %q", failure)
	}
}

// Through the selector: the decide request carries the translated world, the
// dispatched id winning when the caller prices both ids, and the service is
// asked with it.
func TestRouteLookupSendsRemappedPricesUnderTheArmsID(t *testing.T) {
	remapped := func(t *testing.T) (*policySelectorFixture, []string) {
		fixture := routesFixture(t)
		policy := fixture.decision.Algorithm.RaylineARC.PolicyService
		from, to := fixture.decision.ModelRefs[0].Model, fixture.decision.ModelRefs[1].Model
		policy.DispatchModelOverrides = map[string]string{from: to}
		fixture.selector.arm(&raylineARCArmedComponents{
			scorer:    newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision),
			admission: raylinearc.NewAdmissionGate(0),
			policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second}),
		})
		// The service restates the hash of the rates it received.
		fixture.fake.echo = func(request raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
			legs := map[string][4]float64{}
			for key, rate := range request.RoutingPrices.Rates {
				legs[key] = routeRateLegs(rate)
			}
			store, asOf, digest := request.RoutingPrices.StoreVersion, request.RoutingPrices.AsOf, strings.Repeat("d", 64)
			hash, exclusive := routesPriceSHA256(legs, request.RoutingPrices.Exclusive), request.RoutingPrices.Exclusive
			return &raylinearc.PolicyRoutingPricesUsed{
				Mode: "caller", StoreVersion: &store, AsOf: &asOf, Digest: &digest,
				FallbackArms: []string{}, FilledLegs: map[string][]string{},
				PricesSHA256: &hash, Exclusive: &exclusive, ExcludedArms: []string{},
			}
		}
		return fixture, workerModels(fixture)
	}
	t.Run("sent under the arm's id", func(t *testing.T) {
		fixture, models := remapped(t)
		_, controls := routesControlsFor(t, fixture, `{"exclusive":true,"prices":{"`+models[1]+
			`":{"input":1,"cache_read":0.1,"cache_write":1.25,"output":5}}}`)
		if _, err := selectWithControls(t, fixture, controls); err != nil {
			t.Fatalf("select: %v", err)
		}
		rates := decodeJSONObject(t, fixture.fake.receivedBodies()[0])["routing_prices"].(map[string]any)["rates"].(map[string]any)
		if len(rates) != 2 || rates[models[0]] == nil || rates[models[1]] == nil {
			t.Fatalf("rates sent = %v, want %s priced at %s's legs beside it", rates, models[0], models[1])
		}
	})
	t.Run("both ids priced unlike", func(t *testing.T) {
		fixture, models := remapped(t)
		_, controls := routesControlsFor(t, fixture, `{"prices":{"`+models[1]+
			`":{"input":1,"cache_read":0.1,"cache_write":1.25,"output":5},"`+models[0]+`":{"input":2,"cache_read":0.2,"cache_write":2.5,"output":10}}}`)
		if _, err := selectWithControls(t, fixture, controls); err != nil {
			t.Fatalf("select: %v", err)
		}
		rates := decodeJSONObject(t, fixture.fake.receivedBodies()[0])["routing_prices"].(map[string]any)["rates"].(map[string]any)
		if got := rates[models[0]].(map[string]any)["input"]; got != 1.0 {
			t.Fatalf("rates sent = %v, want %s at the dispatched %s's price", rates, models[0], models[1])
		}
	})
}

// v5ConfigWith parses the v5 fixture config, with edit applied, as the cell
// loads it (config validation included).
func v5ConfigWith(t *testing.T, edit func(string) string) *config.RouterConfig {
	t.Helper()
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	manifest, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v5.json")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "package.json")
	if err := os.WriteFile(manifestPath, manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(manifest)
	rendered := strings.NewReplacer(
		"{{POLICY_URL}}", "http://127.0.0.1:1", "{{ALIAS}}", policyTestAlias, "{{PACKAGE}}", hex.EncodeToString(sum[:]),
		"{{MANIFEST}}", manifestPath, "{{GLM_FORMAT}}", "api_format: openai",
		"{{OPUS}}", v5OpusAction, "{{GLM_NONE}}", v5GLMNone, "{{GLM_UP}}", v5GLMUp,
	).Replace(v5ConfigTemplate)
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(edit(rendered)), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return cfg
}

// Every dispatch_model_overrides entry of a cell config loaded as the cell
// loads it: a caller price keyed by the to-worker's provider id passes the
// cell's precheck and reaches the service under the from-worker's. "onto a
// twin" is the shipped deepseek shape (flash sent on flash-1017): the steered
// glm actions are sent on glm-twin, another provider model of the same
// trained model that serves one of that model's actions itself.
func TestEveryDispatchModelOverrideSendsTheArmsPriceKey(t *testing.T) {
	twin := func(rendered string) string {
		rendered = strings.Replace(rendered, "\nrouting:\n  modelCards:\n",
			"    - name: glm-twin\n      provider_model_id: z-ai/glm-5.3\n      api_format: anthropic\n"+
				"      pricing:\n        currency: USD\n        prompt_per_1m: 0.2\n        completion_per_1m: 0.8\n"+
				"      backend_refs:\n        - name: openrouter-glm-twin\n          base_url: https://openrouter.ai/api/v1\n"+
				"          provider: openrouter\n          api_key_env: POLICY_E2E_PROVIDER_KEY\n"+
				"\nrouting:\n  modelCards:\n    - name: glm-twin\n      modality: text\n", 1)
		rendered = strings.Replace(rendered, "        - model: opus\n          use_reasoning: true\n",
			"        - model: opus\n          use_reasoning: true\n        - model: glm-twin\n          use_reasoning: true\n", 1)
		rendered = strings.Replace(rendered, "              glm: glm-5.3-flash\n",
			"              glm: glm-5.3-flash\n              glm-twin: glm-5.3-flash\n", 1)
		return strings.Replace(rendered, "- action_id: "+v5GLMNone+"\n                worker: glm\n",
			"- action_id: "+v5GLMNone+"\n                worker: glm-twin\n", 1)
	}
	for name, tc := range map[string]struct {
		entries string
		// servedSent: an action bound to the to-worker serves another
		// model, so its arm reads the to-worker's own id too.
		servedSent bool
	}{
		"onto a twin":                    {"              glm: glm-twin\n", false},
		"onto a worker of another model": {"              glm: opus\n", true},
		"two onto one twin":              {"              glm: glm-twin\n              opus: glm-twin\n", false},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := v5ConfigWith(t, func(rendered string) string {
				if strings.Contains(tc.entries, "glm-twin") {
					rendered = twin(rendered)
				}
				return strings.Replace(rendered, "            allow_experimental_controls: true\n",
					"            allow_experimental_controls: true\n            dispatch_model_overrides:\n"+tc.entries, 1)
			})
			router := &OpenAIRouter{Config: cfg}
			decision := &router.Config.Decisions[0]
			policy := decision.Algorithm.RaylineARC.PolicyService
			if len(policy.DispatchModelOverrides) != strings.Count(tc.entries, "\n") {
				t.Fatalf("the config carries overrides %v", policy.DispatchModelOverrides)
			}
			refs := map[string]config.ModelRef{}
			for _, ref := range decision.ModelRefs {
				refs[ref.Model] = ref
			}
			scorer := newPolicyServiceScorer(router.Config, decision)
			offered := router.raylineRoutesDecisionWorkerModels(decision)
			for from, to := range policy.DispatchModelOverrides {
				served := policyWorkerManifest(router.Config, refs[to]).Model
				own := policyWorkerManifest(router.Config, refs[from]).Model
				if served == own {
					t.Fatalf("%s -> %s serve one id %s", from, to, served)
				}
				if !offered[served] {
					t.Fatalf("the precheck refuses %s's served id %s", from, served)
				}
				caller := callerWorld(map[string]raylinearc.PolicyRoutingRate{served: callerRate(0.2, 0.8)})
				request := &raylinearc.PolicyDecisionRequest{}
				applyRouteControls(request, scorer, &selection.RaylineARCRouteControls{Prices: caller})
				sent := request.RoutingPrices
				if sent == nil {
					t.Fatalf("%s -> %s: no prices sent", from, to)
				}
				if rate, ok := sent.Rates[own]; !ok || routeRateLegs(rate) != [4]float64{0.2, 0, 0.2, 0.8} {
					t.Fatalf("%s -> %s: sent %v, want %s at the caller's legs", from, to, sent.Rates, own)
				}
				if _, ok := sent.Rates[served]; ok != tc.servedSent {
					t.Fatalf("%s -> %s: %s sent = %v, want %v", from, to, served, ok, tc.servedSent)
				}
			}
		})
	}
}
