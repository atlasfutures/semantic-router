//go:build !windows && cgo

package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// testdata/arc_trained_provider_routes.v1.json is Pathfinder's
// tests/fixtures/arc_policy_service/trained_provider_routes.v1.json
// (pathfinder #3245), verbatim: for every action of the served packages, the
// provider route its arms were trained under, read from each package's own
// policy.json. Refresh it from there, never by hand.
const (
	trainedRoutesFixture       = "testdata/arc_trained_provider_routes.v1.json"
	trainedRoutesFixtureSHA256 = "3acb60bec1cc7e6f55333a9fdc21b38e85af0fffe347f64fa69fcffbe06a5d61"
)

type trainedRoutes struct {
	Schema   string `json:"schema"`
	Packages []struct {
		PackageSHA256 string               `json:"package_sha256"`
		SchemaVersion string               `json:"schema_version"`
		Actions       []trainedRouteAction `json:"actions"`
	} `json:"packages"`
}

type trainedRouteAction struct {
	ActionID    string `json:"action_id"`
	Model       string `json:"model"`
	WireModel   string `json:"wire_model"`
	ServedRoute struct {
		Wire struct {
			Model     string          `json:"model"`
			Provider  map[string]any  `json:"provider"`
			Reasoning json.RawMessage `json:"reasoning"`
		} `json:"wire"`
		TrainedProviders []struct {
			Provider map[string]any `json:"provider"`
		} `json:"trained_providers"`
	} `json:"served_route"`
	MinorityRoutes []json.RawMessage `json:"minority_routes"`
}

// trainedRoute is one model's route in one package: what a cell configures
// as that worker's provider_preferences.
type trainedRoute struct {
	pkg       string
	v5        bool
	model     string
	wireModel string
	provider  map[string]any
}

func loadTrainedRoutes(t *testing.T) trainedRoutes {
	t.Helper()
	raw, err := os.ReadFile(trainedRoutesFixture)
	if err != nil {
		t.Fatal(err)
	}
	if digest := sha256.Sum256(raw); hex.EncodeToString(digest[:]) != trainedRoutesFixtureSHA256 {
		t.Fatalf("%s is not the mirrored Pathfinder fixture (sha256 %x)", trainedRoutesFixture, digest)
	}
	var fixture trainedRoutes
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != "rayline.arc-trained-provider-routes.v1" || len(fixture.Packages) == 0 {
		t.Fatalf("unexpected fixture: schema %q, %d packages", fixture.Schema, len(fixture.Packages))
	}
	return fixture
}

// canonicalProviderRoute is the routing a provider object asks OpenRouter
// for. With fallbacks off, an order that names exactly the one provider only
// already allows changes nothing, so it is dropped: the two forms the arms
// were collected with are one route (the R7 ruling in Pathfinder's
// docs/arc_serving_contract.md, "Trained routes (R7)").
func canonicalProviderRoute(provider map[string]any) map[string]any {
	canonical := make(map[string]any, len(provider))
	for key, value := range provider {
		canonical[key] = value
	}
	only, _ := canonical["only"].([]any)
	if canonical["allow_fallbacks"] == false && len(only) == 1 && reflect.DeepEqual(canonical["order"], only) {
		delete(canonical, "order")
	}
	return canonical
}

// A worker's provider_preferences is one object for every action that names
// it, so a package can be served as trained only if each model's actions
// share one canonical route. Every action of every served package does, and
// no action was trained on a second route.
func TestTrainedRoutesGiveEachModelOneProviderRoute(t *testing.T) {
	routes := trainedRoutesByModel(t, loadTrainedRoutes(t))
	if len(routes) == 0 {
		t.Fatal("the fixture names no routes")
	}
}

func trainedRoutesByModel(t *testing.T, fixture trainedRoutes) []trainedRoute {
	t.Helper()
	var routes []trainedRoute
	for _, pkg := range fixture.Packages {
		byModel := map[string]int{}
		for _, action := range pkg.Actions {
			served := action.ServedRoute.Wire
			if len(action.MinorityRoutes) != 0 {
				t.Errorf("package %.8s action %.8s (%s) was trained on a second route", pkg.PackageSHA256, action.ActionID, action.Model)
			}
			if served.Model != action.WireModel {
				t.Errorf("package %.8s action %.8s serves %q, not its wire model %q", pkg.PackageSHA256, action.ActionID, served.Model, action.WireModel)
			}
			for _, literal := range action.ServedRoute.TrainedProviders {
				if !reflect.DeepEqual(canonicalProviderRoute(literal.Provider), served.Provider) {
					t.Errorf("package %.8s action %.8s: trained %v is not the served route %v", pkg.PackageSHA256, action.ActionID, literal.Provider, served.Provider)
				}
			}
			if v5 := pkg.SchemaVersion == "rayline.arc-policy-package.v5"; v5 && string(served.Reasoning) != "null" {
				t.Errorf("package %.8s action %.8s: a v5 action renders reasoning %s", pkg.PackageSHA256, action.ActionID, served.Reasoning)
			}
			index, seen := byModel[action.Model]
			if !seen {
				byModel[action.Model] = len(routes)
				routes = append(routes, trainedRoute{
					pkg:       pkg.PackageSHA256[:8],
					v5:        pkg.SchemaVersion == "rayline.arc-policy-package.v5",
					model:     action.Model,
					wireModel: action.WireModel,
					provider:  served.Provider,
				})
				continue
			}
			if routes[index].wireModel != action.WireModel {
				t.Errorf("package %.8s: %s actions name two wire models, %q and %q",
					pkg.PackageSHA256, action.Model, routes[index].wireModel, action.WireModel)
			}
			if !reflect.DeepEqual(routes[index].provider, served.Provider) {
				t.Errorf("package %.8s: %s actions were trained on two routes, %v and %v",
					pkg.PackageSHA256, action.Model, routes[index].provider, served.Provider)
			}
		}
	}
	return routes
}

// R7: a worker configured with its model's trained route sends that route,
// for every model of every served package and every client format. The
// provider object is compared canonically; the model is compared exactly. A
// v5 action sends no reasoning control (every v5 action is on the native
// default); v4 reasoning follows the provider_default dispatch ruling
// (pathfinder #2655) and is not compared here.
func TestTrainedProviderRouteReachesTheWire(t *testing.T) {
	formats := []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1}
	for _, route := range trainedRoutesByModel(t, loadTrainedRoutes(t)) {
		router := trainedRouteRouter(t, route)
		for _, format := range formats {
			t.Run(route.pkg+"/"+route.model+"/"+string(format), func(t *testing.T) {
				request := testNeutralRequest("auto", "route this arm")
				response, err := router.handleEntrypointModelRouting(
					request, "auto", providerPinDecision, entropy.ReasoningDecision{}, route.model, routingTestContext(format, request),
				)
				if err != nil {
					t.Fatalf("route: %v", err)
				}
				var body map[string]any
				if err := json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &body); err != nil {
					t.Fatalf("decode the provider body: %v", err)
				}
				if body["model"] != route.wireModel {
					t.Fatalf("model = %v, want the trained %q", body["model"], route.wireModel)
				}
				provider, _ := body["provider"].(map[string]any)
				if !reflect.DeepEqual(canonicalProviderRoute(provider), route.provider) {
					t.Fatalf("provider = %v, want the trained route %v", provider, route.provider)
				}
				if route.v5 {
					for _, key := range []string{"reasoning", "reasoning_effort", "thinking", "output_config"} {
						if _, present := body[key]; present {
							t.Fatalf("a v5 worker sent %s: %v", key, body[key])
						}
					}
				}
			})
		}
	}
}

// trainedRouteRouter serves one model on OpenRouter with its trained route as
// provider_preferences, decoded through the config type a cell's YAML fills.
func trainedRouteRouter(t *testing.T, route trainedRoute) *OpenAIRouter {
	t.Helper()
	raw, err := json.Marshal(route.provider)
	if err != nil {
		t.Fatal(err)
	}
	var preferences config.OpenRouterProviderPreferences
	if err := json.Unmarshal(raw, &preferences); err != nil {
		t.Fatalf("the trained route is not a provider_preferences object: %v", err)
	}
	router := providerPinRouter("https://openrouter.ai/api/v1", &preferences)
	arm := router.Config.ModelConfig[providerPinArm]
	// As the canonical loader records an OpenRouter binding's provider model id.
	router.Config.VLLMEndpoints[0].Type = "openai"
	arm.ExternalModelIDs = map[string]string{"openai": route.wireModel}
	delete(router.Config.ModelConfig, providerPinArm)
	router.Config.ModelConfig[route.model] = arm
	router.Config.DefaultModel = route.model
	if !slices.Contains([]string{"claude-opus-5", "glm-5.3-flash", "gpt-5.6-sol", "kimi-k3", "qwen3.8-27b"}, route.model) {
		t.Fatalf("the fixture names a model this test was not reviewed for: %s", route.model)
	}
	return router
}
