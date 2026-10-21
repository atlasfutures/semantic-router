package extproc

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The golden fixtures are pathfinder#5664's, copied unchanged
// (testdata/arc_public_contract/SOURCE.md).
const arcPublicContractDir = "testdata/arc_public_contract"

type arcContractFixture struct {
	Surface string          `json:"surface"`
	Body    json.RawMessage `json:"body"`
	Expect  struct {
		Status       int     `json:"status"`
		Code         string  `json:"code"`
		Type         string  `json:"type"`
		PriceSHA256  *string `json:"price_sha256"`
		ResultPoints []struct {
			Name       string  `json:"name"`
			ValueUSD   float64 `json:"value_usd"`
			CostWeight float64 `json:"cost_weight"`
		} `json:"result_points"`
		Routing map[string]json.RawMessage `json:"routing"`
	} `json:"expect"`
}

func readContractFixture(t *testing.T, name string) arcContractFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(arcPublicContractDir, name))
	if err != nil {
		t.Fatal(err)
	}
	var fixture arcContractFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return fixture
}

func contractFixtures(t *testing.T, pattern string) []string {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(arcPublicContractDir, pattern))
	if err != nil || len(matches) == 0 {
		t.Fatalf("no fixtures match %s", pattern)
	}
	names := make([]string, 0, len(matches))
	for _, match := range matches {
		names = append(names, filepath.Base(match))
	}
	return names
}

// The cell checks a lookup's model only beside a routing block: without one,
// the gateway has already resolved the model and the cell answers as before.
var modelOnlyFixtures = map[string]bool{
	"request_invalid_model_bare_arc.json":    true,
	"request_invalid_model_missing.json":     true,
	"request_invalid_model_unknown_pin.json": true,
}

// Every invalid Decisions fixture is refused with the contract's code, type
// and status, in the contract's check order. The generation fixtures are the
// gateway's (routing on /v1/messages and its siblings is not this endpoint).
func TestRoutesRoutingRefusesEveryInvalidContractFixture(t *testing.T) {
	for _, name := range contractFixtures(t, "request_invalid_*.json") {
		fixture := readContractFixture(t, name)
		if fixture.Surface != "decisions" || modelOnlyFixtures[name] {
			continue
		}
		t.Run(name, func(t *testing.T) {
			_, _, refusal := readRaylineRoutesRouting(fixture.Body, "", []string{"rayline/arc-dev"})
			if refusal == nil {
				t.Fatal("accepted")
			}
			if refusal.code != fixture.Expect.Code || refusal.errType != fixture.Expect.Type || refusal.status != fixture.Expect.Status {
				t.Fatalf("refused %d %s %s, want %d %s %s", refusal.status, refusal.errType, refusal.code,
					fixture.Expect.Status, fixture.Expect.Type, fixture.Expect.Code)
			}
		})
	}
	// With a routing block the model is checked too.
	for _, name := range []string{"request_invalid_model_bare_arc.json", "request_invalid_model_unknown_pin.json", "request_invalid_model_missing.json"} {
		fixture := readContractFixture(t, name)
		var body map[string]any
		_ = json.Unmarshal(fixture.Body, &body)
		body["routing"] = map[string]any{}
		encoded, _ := json.Marshal(body)
		if _, _, refusal := readRaylineRoutesRouting(encoded, "", []string{"rayline/arc-dev"}); refusal == nil || refusal.code != fixture.Expect.Code {
			t.Errorf("%s with routing: refusal %+v, want %s", name, refusal, fixture.Expect.Code)
		}
	}
	// The cell's own alias is a model, not an unknown ARC id.
	if _, _, refusal := readRaylineRoutesRouting([]byte(`{"model":"rayline/arc-dev","messages":[{"role":"user","content":"x"}],"routing":{}}`), "", []string{"rayline/arc-dev"}); refusal != nil {
		t.Fatalf("the cell's alias refused: %+v", refusal)
	}
}

// Every valid Decisions fixture parses to the contract's routing, price hash
// and result points (swift and sage always, custom at the caller's V, each at
// cost_weight = 1 / V), and the routing block leaves the body.
func TestRoutesRoutingReadsEveryValidContractFixture(t *testing.T) {
	settings := &config.RaylineARCRoutesRoutingConfig{ExplorationSalt: "salt"}
	for _, name := range contractFixtures(t, "request_valid_decisions_*.json") {
		fixture := readContractFixture(t, name)
		t.Run(name, func(t *testing.T) {
			routing, stripped, refusal := readRaylineRoutesRouting(fixture.Body, "", nil)
			if refusal != nil {
				t.Fatalf("refused: %+v", refusal)
			}
			var body map[string]json.RawMessage
			if err := json.Unmarshal(stripped, &body); err != nil {
				t.Fatal(err)
			}
			if _, present := body["routing"]; present {
				t.Fatal("routing left in the body the codec reads")
			}
			hash := ""
			if routing != nil {
				hash = routing.priceSHA256
			}
			if want := fixture.Expect.PriceSHA256; (want == nil) != (hash == "") || (want != nil && *want != hash) {
				t.Fatalf("price_sha256 %q, want %v", hash, fixture.Expect.PriceSHA256)
			}
			controls := raylineRoutesControls(routing, settings, time.Unix(0, 0))
			if len(controls.Points) != len(fixture.Expect.ResultPoints) {
				t.Fatalf("points %+v, want %+v", controls.Points, fixture.Expect.ResultPoints)
			}
			for index, want := range fixture.Expect.ResultPoints {
				got := controls.Points[index]
				if got.Name != want.Name || got.CostWeight != want.CostWeight {
					t.Fatalf("point %d = %+v, want %+v", index, got, want)
				}
			}
			if raw, sent := fixture.Expect.Routing["temperature"]; sent {
				var temperature float64
				_ = json.Unmarshal(raw, &temperature)
				explore := controls.Exploration
				switch {
				case temperature == 0 && explore != nil:
					t.Fatalf("T 0 sent exploration %+v", explore)
				case temperature > 0 && (explore == nil || explore.Temperature.Temperature != temperature ||
					explore.Temperature.Surface != "decisions" || explore.Temperature.Source != "caller" ||
					explore.Temperature.TrainingData != "not_allowed" || explore.Salt != "salt"):
					t.Fatalf("exploration %+v for T %g", explore, temperature)
				}
			}
		})
	}
}

// The canonical price hash matches every vector, preimage and digest: key
// order and integer-versus-float spelling do not move it.
func TestRoutesPriceSHA256MatchesTheContractVectors(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(arcPublicContractDir, "price_sha256_vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var vectors []struct {
		Prices      json.RawMessage `json:"prices"`
		Exclusive   bool            `json:"exclusive"`
		JCS         string          `json:"jcs"`
		PriceSHA256 string          `json:"price_sha256"`
	}
	if err := json.Unmarshal(raw, &vectors); err != nil || len(vectors) == 0 {
		t.Fatalf("vectors: %v", err)
	}
	for index, vector := range vectors {
		prices, refusal := routesReadPrices(vector.Prices)
		if refusal != nil {
			t.Fatalf("vector %d refused: %+v", index, refusal)
		}
		if got := routesPriceJCS(prices, vector.Exclusive); got != vector.JCS {
			t.Errorf("vector %d preimage\n got %s\nwant %s", index, got, vector.JCS)
		}
		if got := routesPriceSHA256(prices, vector.Exclusive); got != vector.PriceSHA256 {
			t.Errorf("vector %d hash %s, want %s", index, got, vector.PriceSHA256)
		}
	}
}

// ECMAScript number formatting at the notation boundaries.
func TestJCSNumber(t *testing.T) {
	for value, want := range map[float64]string{
		0: "0", 1: "1", 15: "15", 0.3: "0.3", 3.75: "3.75", 1e21: "1e+21", 1e20: "100000000000000000000",
		0.000001: "0.000001", 0.0000001: "1e-7", -2.5: "-2.5", 123456.789: "123456.789", 1.0 / 3: "0.3333333333333333",
		math.MaxFloat64: "1.7976931348623157e+308",
	} {
		if got := jcsNumber(value); got != want {
			t.Errorf("jcsNumber(%v) = %s, want %s", value, got, want)
		}
	}
}

// memex-desktop#7531's header stands for routing.temperature when the block
// states none, and must agree with it when both are sent.
func TestRoutesTemperatureHeaderCompatibility(t *testing.T) {
	plain := []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"x"}]}`)
	routing, body, refusal := readRaylineRoutesRouting(plain, "1.5", nil)
	if refusal != nil || routing == nil || routing.bodySent || *routing.temperature != 1.5 || string(body) != string(plain) {
		t.Fatalf("header alone: %+v %s %+v", routing, body, refusal)
	}
	both := []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"x"}],"routing":{"temperature":1}}`)
	if _, _, refusal := readRaylineRoutesRouting(both, "2", nil); refusal == nil || refusal.code != "routing_invalid" {
		t.Fatalf("disagreeing header and block: %+v", refusal)
	}
	if _, _, refusal := readRaylineRoutesRouting(plain, "-1", nil); refusal == nil || refusal.code != "temperature_invalid" {
		t.Fatalf("negative header: %+v", refusal)
	}
	if routing, body, refusal := readRaylineRoutesRouting(plain, "", nil); routing != nil || refusal != nil || string(body) != string(plain) {
		t.Fatal("a body with neither changed")
	}
}

// routesFixture is a decisions cell whose fake service echoes caller prices
// as applied.
func routesFixture(t *testing.T) *policySelectorFixture {
	t.Helper()
	fixture := newDecisionsV2Fixture(t, nil)
	fixture.fake.echo = func(request raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
		if request.RoutingPrices == nil {
			return nil
		}
		store, asOf, digest := request.RoutingPrices.StoreVersion, request.RoutingPrices.AsOf, strings.Repeat("d", 64)
		return &raylinearc.PolicyRoutingPricesUsed{
			Mode: "live", StoreVersion: &store, AsOf: &asOf, Digest: &digest,
			FallbackArms: []string{}, FilledLegs: map[string][]string{},
		}
	}
	return fixture
}

func routesControlsFor(t *testing.T, fixture *policySelectorFixture, routingJSON string) (*raylineRoutesRouting, *selection.RaylineARCRouteControls) {
	t.Helper()
	body := `{"model":"arc-1.0","messages":[{"role":"user","content":"x"}],"routing":` + routingJSON + `}`
	routing, _, refusal := readRaylineRoutesRouting([]byte(body), "", nil)
	if refusal != nil {
		t.Fatalf("refused: %+v", refusal)
	}
	return routing, raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{ExplorationSalt: "salt"}, time.Now())
}

func selectWithControls(t *testing.T, fixture *policySelectorFixture, controls *selection.RaylineARCRouteControls) (*selection.SelectionResult, error) {
	t.Helper()
	state, _ := raylinearc.NewEpisodeState(2)
	return fixture.selectOnWith(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}),
		func(arcContext *selection.RaylineARCSelectionContext) { arcContext.RouteControls = controls })
}

func workerModels(fixture *policySelectorFixture) []string {
	scorer := newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
	models := make([]string, 0, len(scorer.workers))
	for _, worker := range scorer.workers {
		models = append(models, worker.Model)
	}
	return models
}

// A lookup asks the service at swift, sage and custom in one decide, sends
// the caller's prices and exploration, and records a checked pick per point.
func TestRouteControlsReachTheDecideRequestAndComeBackAsResults(t *testing.T) {
	fixture := routesFixture(t)
	models := workerModels(fixture)
	fixture.fake.tierPickExploration = true
	_, controls := routesControlsFor(t, fixture, `{"value_usd":1.5,"temperature":0.7,"prices":{"`+models[0]+
		`":{"input":3,"cache_read":0.3,"cache_write":3.75,"output":15}}}`)
	result, err := selectWithControls(t, fixture, controls)
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := decodeJSONObject(t, fixture.fake.receivedBodies()[0])
	selectionSent := sent["selection"].(map[string]any)
	points := selectionSent["tier_points"].([]any)
	if len(points) != 3 || points[0].(map[string]any)["name"] != "swift" || points[0].(map[string]any)["cost_weight"] != 10.0 ||
		points[1].(map[string]any)["cost_weight"] != 1/4.5 || points[2].(map[string]any)["cost_weight"] != 1/1.5 {
		t.Fatalf("tier_points %v", points)
	}
	explore := selectionSent["exploration"].(map[string]any)
	if explore["surface"] != "decisions" || explore["training_data"] != "not_allowed" || explore["temperature"] != 0.7 {
		t.Fatalf("exploration %v", explore)
	}
	prices := sent["routing_prices"].(map[string]any)
	if !strings.HasPrefix(prices["store_version"].(string), "caller:") || len(prices["rates"].(map[string]any)) != 1 {
		t.Fatalf("routing_prices %v", prices)
	}
	trace := result.RaylineARC
	if len(trace.RouteResults) != 3 || trace.RouteExploration == nil {
		t.Fatalf("results %+v exploration %+v", trace.RouteResults, trace.RouteExploration)
	}
	// Decisions draws never reach llm_usage or the episode.
	if trace.PolicyExploration != nil || trace.PolicyExplorationSent {
		t.Fatalf("a Decisions draw was recorded as the cell's: %+v", trace.PolicyExploration)
	}
}

// Fail closed: a missing point, unapplied prices or a record contradicting
// the requested T withholds the answer as inconsistent (502), and an
// exclusive price world no offered model belongs to is undecidable (422).
func TestRouteControlsFailClosed(t *testing.T) {
	t.Run("missing point", func(t *testing.T) {
		fixture := routesFixture(t)
		fixture.fake.omitTierPick = "custom"
		_, controls := routesControlsFor(t, fixture, `{"value_usd":1.5}`)
		_, err := selectWithControls(t, fixture, controls)
		if !errors.Is(selectionFailureError(err), routerruntime.ErrRouteDecisionInconsistent) {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("prices not applied", func(t *testing.T) {
		fixture := routesFixture(t)
		fixture.fake.echo = func(request raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed {
			store, asOf := request.RoutingPrices.StoreVersion, request.RoutingPrices.AsOf
			return &raylinearc.PolicyRoutingPricesUsed{
				Mode: "package", StoreVersion: &store, AsOf: &asOf, FallbackArms: []string{}, FilledLegs: map[string][]string{},
			}
		}
		models := workerModels(fixture)
		_, controls := routesControlsFor(t, fixture, `{"prices":{"`+models[0]+`":{"input":1,"cache_read":0,"cache_write":0,"output":2}}}`)
		_, err := selectWithControls(t, fixture, controls)
		if !errors.Is(selectionFailureError(err), routerruntime.ErrRouteDecisionInconsistent) {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("exclusive with nothing offered", func(t *testing.T) {
		fixture := routesFixture(t)
		_, controls := routesControlsFor(t, fixture, `{"prices":{"acme/none":{"input":1,"cache_read":0,"cache_write":0,"output":2}},"exclusive":true}`)
		_, err := selectWithControls(t, fixture, controls)
		if !errors.Is(selectionFailureError(err), routerruntime.ErrRouteDecisionUndecidable) {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("exclusive narrows the offer", func(t *testing.T) {
		fixture := routesFixture(t)
		models := workerModels(fixture)
		_, controls := routesControlsFor(t, fixture, `{"prices":{"`+models[0]+`":{"input":1,"cache_read":0,"cache_write":0,"output":2}},"exclusive":true}`)
		if _, err := selectWithControls(t, fixture, controls); err != nil {
			t.Fatalf("select: %v", err)
		}
		offered := decodeJSONObject(t, fixture.fake.receivedBodies()[0])["selection"].(map[string]any)["available_action_ids"].([]any)
		scorer := newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
		for _, action := range offered {
			if scorer.workers[scorer.bindings[action.(string)].arm].Model != models[0] {
				t.Fatalf("an unpriced action %v was offered under exclusive", action)
			}
		}
	})
	t.Run("record contradicts T", func(t *testing.T) {
		record := &raylinearc.PolicyExplorationRecord{Applied: true}
		requested, higher, propensity, explored := 0.5, 0.9, 0.3, true
		record.RequestedTemperature, record.Temperature, record.Propensity, record.Explored = &requested, &higher, &propensity, &explored
		if routeExplorationConsistent(record, 0.5) {
			t.Fatal("an applied T above the requested one stood")
		}
		record.Temperature = &requested
		if !routeExplorationConsistent(record, 0.5) || routeExplorationConsistent(record, 0.7) {
			t.Fatal("consistency misread")
		}
	})
}

// The answer: presets always, custom exactly with value_usd, every result in
// the caller's price world, a temperature applied only where the service
// reported a draw, and the echo listing what was not applied.
func TestRoutesAnswerRendersTheContractEnvelope(t *testing.T) {
	routing, _, _ := readRaylineRoutesRouting([]byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"x"}],"routing":`+
		`{"value_usd":1.5,"temperature":0.7,"prices":{"m/a":{"input":1,"cache_read":0,"cache_write":0,"output":2}},"exclusive":true}}`), "", nil)
	applied, propensity, explored, key := 0.7, 0.375, true, "k"
	offered := "m/a"
	record := &raylinearc.PolicyExplorationRecord{
		Applied: true, RequestedTemperature: &applied, Temperature: &applied, Propensity: &propensity, Explored: &explored,
		DecisionKey: &key, Offered: []raylinearc.PolicyStageOneCandidate{{"model": &offered}}, Pi: []float64{1},
	}
	result := func(name string, weight float64, draw *raylinearc.PolicyExplorationRecord) routerruntime.RouteResult {
		return routerruntime.RouteResult{Name: name, CostWeight: weight, SelectedWorker: "w", WorkerModel: "m/a", Exploration: draw}
	}
	decision := routerruntime.RouteDecision{Results: []routerruntime.RouteResult{
		result("swift", 10, record), result("sage", 1/4.5, record), result("custom", 1/1.5, record),
	}, Exploration: record}
	var payload raylineRoutesResponse
	if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(payload)
	var body map[string]any
	_ = json.Unmarshal(encoded, &body)
	presets := body["presets"].(map[string]any)
	swift := presets["swift"].(map[string]any)
	if swift["value_usd"] != 0.1 || swift["cost_weight"] != 10.0 ||
		!reflect.DeepEqual(swift["prices"], map[string]any{"basis": "caller", "price_sha256": routing.priceSHA256}) ||
		swift["temperature"].(map[string]any)["status"] != "applied" {
		t.Fatalf("swift %v", swift)
	}
	if body["custom"].(map[string]any)["value_usd"] != 1.5 {
		t.Fatalf("custom %v", body["custom"])
	}
	echo := body["routing"].(map[string]any)
	if echo["temperature"].(map[string]any)["status"] != "applied" || len(echo["unapplied_keys"].([]any)) != 0 ||
		echo["prices"].(map[string]any)["exclusive"] != true || body["exploration"] == nil {
		t.Fatalf("routing echo %v", echo)
	}

	// A service that draws only the lookup's own decision leaves the
	// presets' temperature unsupported: the echo says so.
	decision.Results[0].Exploration, decision.Results[1].Exploration = nil, nil
	payload = raylineRoutesResponse{}
	if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
		t.Fatal(err)
	}
	if payload.Presets.Swift.Temperature.Status != "unsupported" || payload.Presets.Swift.Temperature.Propensity != nil ||
		payload.Routing.Temperature.Status != "unsupported" || !slices.Equal(payload.Routing.UnappliedKeys, []string{"temperature"}) {
		t.Fatalf("unsupported presets: %+v %+v", payload.Presets.Swift.Temperature, payload.Routing)
	}

	// No routing: presets alone, catalog prices, no echo.
	payload = raylineRoutesResponse{}
	if err := raylineRoutesAddResults(&payload, nil, routerruntime.RouteDecision{Results: decision.Results[:2]}); err != nil {
		t.Fatal(err)
	}
	if payload.Routing != nil || payload.Custom != nil || payload.Presets.Sage.Prices.Basis != "catalog" ||
		payload.Presets.Sage.Prices.PriceSHA256 != nil || payload.Presets.Sage.ValueUSD != 4.5 {
		t.Fatalf("no routing: %+v", payload)
	}
	// A missing preset is not answered.
	if err := raylineRoutesAddResults(&raylineRoutesResponse{}, nil, routerruntime.RouteDecision{Results: decision.Results[:1]}); err == nil {
		t.Fatal("an answer without sage was rendered")
	}
}

// A cell serving routing needs tier points and decisions_v2 over the whole
// public V range, and the temperature shape when it seeds lookup draws.
func TestRoutesRoutingReadinessRequirements(t *testing.T) {
	var required raylinearc.PolicyServiceRequirements
	routesRequirements(&required, &config.RaylineARCRoutesRoutingConfig{ExplorationSalt: "s"})
	for _, capability := range []string{
		raylinearc.PolicyCapabilityDecisionsV2, raylinearc.PolicyCapabilityTierPoints, raylinearc.PolicyCapabilityExplorationTemperature,
	} {
		if !slices.Contains(required.Capabilities, capability) {
			t.Fatalf("capabilities %v lack %s", required.Capabilities, capability)
		}
	}
	for _, weight := range []float64{0.2, 10, 1 / 4.5} {
		if !slices.Contains(required.CostWeights, weight) {
			t.Fatalf("cost weights %v lack %g", required.CostWeights, weight)
		}
	}
	none := raylinearc.PolicyServiceRequirements{}
	routesRequirements(&none, nil)
	if len(none.Capabilities) != 0 || len(none.CostWeights) != 0 {
		t.Fatalf("a cell without routing requires %+v", none)
	}
}

func routesErrorCode(t *testing.T, response *ext_proc.ProcessingResponse) (int, string, string) {
	t.Helper()
	var envelope struct {
		Error struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &envelope); err != nil {
		t.Fatalf("error body: %v", err)
	}
	return immediateStatusCode(t, response), envelope.Error.Type, envelope.Error.Code
}

// At the endpoint: a contract refusal answers with its code before any
// decision; a body's routing on a cell that does not serve it is refused,
// not ignored; a priced model the cell does not offer is unknown_price_model.
func TestRaylineRoutesAnswersRoutingRefusalsWithTheirCodes(t *testing.T) {
	served := routesRouter(true)
	served.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.Routing = &config.RaylineARCRoutesRoutingConfig{}
	served.Config.Decisions[0].ModelRefs = []config.ModelRef{{Model: "m/a"}}
	body := func(routing string) []byte {
		return []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"hi"}],"routing":` + routing + `}`)
	}
	for name, tc := range map[string]struct {
		router           *OpenAIRouter
		body             []byte
		status           int
		errType, errCode string
	}{
		"out of bounds":        {served, body(`{"value_usd":9}`), 400, "invalid_request_error", "value_out_of_bounds"},
		"tier V":               {served, []byte(`{"model":"arc-sage","messages":[{"role":"user","content":"hi"}],"routing":{"value_usd":1}}`), 400, "invalid_request_error", "tier_value_immutable"},
		"unknown price model":  {served, body(`{"prices":{"acme/x":{"input":1,"cache_read":0,"cache_write":0,"output":1}}}`), 400, "invalid_request_error", "unknown_price_model"},
		"cell without routing": {routesRouter(true), body(`{"value_usd":1}`), 503, "api_error", "unavailable"},
	} {
		t.Run(name, func(t *testing.T) {
			response := tc.router.handleRaylineRoutesAPI(tc.body, routesContext(nil))
			if response == nil {
				t.Fatal("no answer")
			}
			status, errType, errCode := routesErrorCode(t, response)
			if status != tc.status || errType != tc.errType || errCode != tc.errCode {
				t.Fatalf("answered %d %s %s, want %d %s %s", status, errType, errCode, tc.status, tc.errType, tc.errCode)
			}
		})
	}
}
