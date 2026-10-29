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
			_, _, refusal := readRaylineRoutesRouting(fixture.Body, []string{"rayline/arc-dev"})
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
		if _, _, refusal := readRaylineRoutesRouting(encoded, []string{"rayline/arc-dev"}); refusal == nil || refusal.code != fixture.Expect.Code {
			t.Errorf("%s with routing: refusal %+v, want %s", name, refusal, fixture.Expect.Code)
		}
	}
	// The cell's own alias is a model, not an unknown ARC id.
	if _, _, refusal := readRaylineRoutesRouting([]byte(`{"model":"rayline/arc-dev","messages":[{"role":"user","content":"x"}],"routing":{}}`), []string{"rayline/arc-dev"}); refusal != nil {
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
			routing, stripped, refusal := readRaylineRoutesRouting(fixture.Body, nil)
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
		// Decoded directly: the vectors exercise number formatting beyond
		// the admitted leg bound (1e21), which the hash must still match.
		var legs map[string]map[string]float64
		if err := json.Unmarshal(vector.Prices, &legs); err != nil {
			t.Fatalf("vector %d: %v", index, err)
		}
		prices := map[string][4]float64{}
		for model, leg := range legs {
			prices[model] = [4]float64{leg["input"], leg["cache_read"], leg["cache_write"], leg["output"]}
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

// The body member is the one transport: #7531's retired header is never
// read, with or without a block.
func TestRoutesIgnoreTheRetiredTemperatureHeader(t *testing.T) {
	router := routesRouter(true)
	router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.Routing = &config.RaylineARCRoutesRoutingConfig{ExplorationSalt: "s"}
	ctx := routesContext(map[string]string{raylineARCExplorationTemperatureHeader: "-1"})
	// A header the old transport would have refused (negative T) changes
	// nothing: the lookup proceeds to selection, which this router lacks.
	response := router.handleRaylineRoutesAPI([]byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"x"}]}`), ctx)
	if status, _, code := routesErrorCode(t, response); code == "temperature_invalid" || status == 400 {
		t.Fatalf("the retired header was read: %d %s", status, code)
	}
	plain := []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"x"}]}`)
	if routing, body, refusal := readRaylineRoutesRouting(plain, nil); routing != nil || refusal != nil || string(body) != string(plain) {
		t.Fatal("a body with no block changed")
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
		// A caller world's store version names its hash.
		store, asOf, digest := request.RoutingPrices.StoreVersion, request.RoutingPrices.AsOf, strings.Repeat("d", 64)
		hash, exclusive := strings.TrimPrefix(store, "caller:"), request.RoutingPrices.Exclusive
		return &raylinearc.PolicyRoutingPricesUsed{
			Mode: "caller", StoreVersion: &store, AsOf: &asOf, Digest: &digest,
			FallbackArms: []string{}, FilledLegs: map[string][]string{},
			PricesSHA256: &hash, Exclusive: &exclusive, ExcludedArms: []string{},
		}
	}
	return fixture
}

func routesControlsFor(t *testing.T, fixture *policySelectorFixture, routingJSON string) (*raylineRoutesRouting, *selection.RaylineARCRouteControls) {
	t.Helper()
	body := `{"model":"arc-1.0","messages":[{"role":"user","content":"x"}],"routing":` + routingJSON + `}`
	routing, _, refusal := readRaylineRoutesRouting([]byte(body), nil)
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
	t.Run("caller prices unechoed", func(t *testing.T) {
		fixture := routesFixture(t)
		fixture.fake.echo = func(raylinearc.PolicyDecisionRequest) *raylinearc.PolicyRoutingPricesUsed { return nil }
		models := workerModels(fixture)
		_, controls := routesControlsFor(t, fixture, `{"prices":{"`+models[0]+`":{"input":1,"cache_read":0,"cache_write":0,"output":2}}}`)
		_, err := selectWithControls(t, fixture, controls)
		if !errors.Is(selectionFailureError(err), routerruntime.ErrRouteDecisionInconsistent) {
			t.Fatalf("an unechoed caller price world answered %v, want inconsistent (502)", err)
		}
	})
	t.Run("applied record with an impossible propensity", func(t *testing.T) {
		requested, propensity, explored, reason := 0.5, 1.5, true, (*string)(nil)
		record := &raylinearc.PolicyExplorationRecord{
			Applied: true, Reason: reason, RequestedTemperature: &requested, Temperature: &requested,
			Propensity: &propensity, Explored: &explored,
		}
		if routeExplorationConsistent(record, 0.5) {
			t.Fatal("a propensity above 1 stood")
		}
		zero := 0.0
		record.Propensity = &zero
		if routeExplorationConsistent(record, 0.5) {
			t.Fatal("a propensity of 0 for the served model stood")
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
		`{"value_usd":1.5,"temperature":0.7,"prices":{"m/a":{"input":1,"cache_read":0,"cache_write":0,"output":2}},"exclusive":true}}`), nil)
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
		raylinearc.PolicyCapabilityCallerPrices,
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
		// Presence decides on such a cell, before any member is read.
		"cell without routing, invalid block": {routesRouter(true), body(`{"value_usd":9}`), 503, "api_error", "unavailable"},
		"cell without routing, null block":    {routesRouter(true), body(`null`), 503, "api_error", "unavailable"},
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

// A caller's prices go out as routing_prices.basis caller (pathfinder#5668):
// exactly four legs per model, beside exclusive; the feed's keep their form.
func TestCallerPricesWireForm(t *testing.T) {
	legs := func(v float64) *float64 { return &v }
	caller, err := json.Marshal(raylinearc.PolicyRoutingPrices{
		Schema: raylinearc.RoutingPricesSchema, StoreVersion: "caller:h", AsOf: "2026-10-21T00:00:00Z",
		Rates:     map[string]raylinearc.PolicyRoutingRate{"m/a": {Input: legs(1), CacheRead: legs(0), CacheWrite: legs(0), Output: legs(2)}},
		Basis:     raylinearc.RoutingPricesBasisCaller,
		Exclusive: true,
	})
	if err != nil || string(caller) != `{"schema":"arc-routing-prices.v1","store_version":"caller:h","as_of":"2026-10-21T00:00:00Z",`+
		`"rates":{"m/a":{"input":1,"cache_read":0,"cache_write":0,"output":2}},"basis":"caller","exclusive":true}` {
		t.Fatalf("caller form %s %v", caller, err)
	}
	var decoded raylinearc.PolicyRoutingPrices
	if err := json.Unmarshal(caller, &decoded); err != nil || decoded.Basis != "caller" || !decoded.Exclusive {
		t.Fatalf("decoded %+v %v", decoded, err)
	}
	feed, _ := json.Marshal(raylinearc.PolicyRoutingPrices{Schema: "s", StoreVersion: "v", AsOf: "a", Rates: map[string]raylinearc.PolicyRoutingRate{"m": {}}})
	if strings.Contains(string(feed), "basis") || !strings.Contains(string(feed), `"valid_from":null`) {
		t.Fatalf("feed form %s", feed)
	}
}

// The service's own unknown_price_model refusal is the caller's 400, not an
// undecidable turn.
func TestServiceUnknownPriceModelIsTheCallers(t *testing.T) {
	fixture := routesFixture(t)
	fixture.fake.refuseCold = func(raylinearc.PolicyDecisionRequest) bool { return true }
	fixture.fake.coldReason = "unknown_price_model"
	models := workerModels(fixture)
	_, controls := routesControlsFor(t, fixture, `{"prices":{"`+models[0]+`":{"input":1,"cache_read":0,"cache_write":0,"output":2}}}`)
	_, err := selectWithControls(t, fixture, controls)
	mapped := selectionFailureError(err)
	if !errors.Is(mapped, routerruntime.ErrRouteDecisionInvalidBody) || !errors.Is(mapped, errRaylineRoutesUnknownPriceModel) {
		t.Fatalf("err = %v", mapped)
	}
}

// A lookup with routing controls is never held: a capacity or media hold
// asks the service nothing, so it has no preset, and holding would commit
// the episode before the lookup failed for want of one. It fails as an
// unheld turn does, with nothing committed.
func TestRouteControlsAreNeverHeld(t *testing.T) {
	for _, class := range []string{raylinearc.PolicyCapacityRefusalClass, raylinearc.PolicyMediaRefusedClass} {
		t.Run(class, func(t *testing.T) {
			fixture := routesFixture(t)
			bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
			state, _ := raylinearc.NewEpisodeState(2)
			servedTurn(t, fixture, state, bindings[1].ActionID)
			if class == raylinearc.PolicyMediaRefusedClass {
				mediaRefusal(fixture, "document_unsupported")
			} else {
				fixture.fake.mu.Lock()
				fixture.fake.failWith = class
				fixture.fake.mu.Unlock()
			}
			_, controls := routesControlsFor(t, fixture, `{"value_usd":1.5}`)
			result, err := fixture.selectOnWith(t, state, nextTurn(t),
				func(arcContext *selection.RaylineARCSelectionContext) { arcContext.RouteControls = controls })
			if err == nil {
				t.Fatalf("a lookup with routing controls was held: offer %q", result.RaylineARC.OfferKind)
			}
			// The same refusal without controls still holds.
			if held, err := fixture.selectOn(t, state, nextTurn(t)); err != nil || held.RaylineARC.OfferKind == "" {
				t.Fatalf("an unrouted turn was not held: %v", err)
			}
		})
	}
}

// A result whose record says the service did not apply T is not shown as
// applied in the echo: the echo says unsupported and lists temperature.
func TestRoutesEchoNeverAppliesAnUnappliedTemperature(t *testing.T) {
	routing, _, _ := readRaylineRoutesRouting([]byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"x"}],"routing":{"temperature":0.7}}`), nil)
	requested, reason := 0.7, "clamped_to_zero"
	unapplied := &raylinearc.PolicyExplorationRecord{Applied: false, RequestedTemperature: &requested, Reason: &reason}
	result := func(name string, weight float64) routerruntime.RouteResult {
		return routerruntime.RouteResult{Name: name, CostWeight: weight, SelectedWorker: "w", WorkerModel: "m/a", Exploration: unapplied}
	}
	var payload raylineRoutesResponse
	decision := routerruntime.RouteDecision{Results: []routerruntime.RouteResult{result("swift", 10), result("sage", 1/4.5)}}
	if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
		t.Fatal(err)
	}
	if payload.Presets.Swift.Temperature.Status != "not_applied" || payload.Routing.Temperature.Status != "unsupported" ||
		!slices.Equal(payload.Routing.UnappliedKeys, []string{"temperature"}) {
		t.Fatalf("result %+v echo %+v", payload.Presets.Swift.Temperature, payload.Routing)
	}
}

// The tiers' versioned ids are the tiers: a routed lookup on them is admitted
// and keeps the tier's fixed value, like arc-swift and arc-sage.
func TestRoutesRoutingTreatsTheVersionedTierIdsAsTheTiers(t *testing.T) {
	body := func(model, routing string) []byte {
		return []byte(`{"model":"` + model + `","messages":[{"role":"user","content":"hi"}],"routing":` + routing + `}`)
	}
	for model, value := range map[string]string{"arc-swift-1.0": "0.1", "arc-sage-1.0": "4.5", "rayline/arc-sage-1.0": "4.5"} {
		if _, _, refusal := readRaylineRoutesRouting(body(model, `{"prices":{"openai/gpt-6-luna":{"input":5000,"cache_read":500,"cache_write":5000,"output":40000}},"exclusive":true}`), nil); refusal != nil {
			t.Fatalf("%s: refused (%s: %s)", model, refusal.code, refusal.message)
		}
		if _, _, refusal := readRaylineRoutesRouting(body(model, `{"value_usd":`+value+`}`), nil); refusal != nil {
			t.Fatalf("%s at its own value: refused (%s)", model, refusal.code)
		}
		if _, _, refusal := readRaylineRoutesRouting(body(model, `{"value_usd":2}`), nil); refusal == nil || refusal.code != "tier_value_immutable" {
			t.Fatalf("%s at another value: got %v, want tier_value_immutable", model, refusal)
		}
	}
	if _, _, refusal := readRaylineRoutesRouting(body("arc-sage-2.0", `{"value_usd":1}`), nil); refusal == nil || refusal.code != "unknown_model" {
		t.Fatalf("arc-sage-2.0: got %v, want unknown_model", refusal)
	}
}
