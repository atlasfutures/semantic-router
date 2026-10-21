package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The predictions fixtures are pathfinder#5717's (SOURCE.md): the policy
// service's decide answer and the public answer project_predictions makes
// of it at V = 2.

func readJSONFixture(t *testing.T, name string, into any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(arcPublicContractDir, name))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, into); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}

// predictionsFixtureDecision is the route decision the selector records for
// the fixture's decide answer: one result per tier pick at the contract's
// points, each carrying its pick's stage-one row.
func predictionsFixtureDecision(t *testing.T) routerruntime.RouteDecision {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(arcPublicContractDir, "decision_response_predictions.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	response, err := raylinearc.DecodePolicyDecisionResponse(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	decision := routerruntime.RouteDecision{
		Predictions: response.Decision.Predictions, PredictionsApplied: true,
		PackageSHA256: response.Package.PackageSHA256,
	}
	// The alternatives the selector scored, as the contract's answer lists
	// them with their actions.
	var answer struct {
		Body struct {
			Alternatives []struct {
				Model, Worker, Provider string
				Score                   float64
				ActionID                string `json:"action_id"`
				ThinkingLevel           string `json:"thinking_level"`
			} `json:"alternatives"`
		} `json:"body"`
	}
	readJSONFixture(t, "response_predictions.json", &answer)
	for _, alternative := range answer.Body.Alternatives {
		decision.Alternatives = append(decision.Alternatives, routerruntime.RouteAlternative{
			Model: alternative.Model, Worker: alternative.Worker, Provider: alternative.Provider, Score: alternative.Score,
			ActionID: alternative.ActionID, ThinkingLevel: alternative.ThinkingLevel,
		})
	}
	for _, point := range []raylineRoutesPoint{{"swift", 10}, {"sage", 1 / 4.5}, {"custom", 0.5}} {
		pick := response.Decision.TierPicks[point.name]
		decision.Results = append(decision.Results, routerruntime.RouteResult{
			Name: point.name, CostWeight: point.costWeight, SelectedWorker: "w", WorkerModel: pick.Model,
			StageOneCandidate: pick.StageOneCandidate,
		})
	}
	return decision
}

func predictionsRouting(t *testing.T) *raylineRoutesRouting {
	t.Helper()
	fixture := readContractFixture(t, "request_valid_decisions_predictions.json")
	routing, _, refusal := readRaylineRoutesRouting(fixture.Body, nil)
	if refusal != nil {
		t.Fatalf("refused: %+v", refusal)
	}
	return routing
}

func renderRoutesPayload(t *testing.T, routing *raylineRoutesRouting, decision routerruntime.RouteDecision) map[string]any {
	t.Helper()
	payload := raylineRoutesPayload("rte_test", "checkpoint", decision, nil, 0)
	if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatal(err)
	}
	return body
}

// assertPredictionFields compares, field for field, the keys the
// predictions projection owns: candidates, candidates_context, alternatives
// (their action fields), the routing
// echo, and each result's model_choice and provenance (present or absent
// as the fixture has them).
func assertPredictionFields(t *testing.T, got, want map[string]any) {
	t.Helper()
	for _, key := range []string{"candidates", "candidates_context", "routing", "alternatives"} {
		gotValue, gotPresent := got[key]
		wantValue, wantPresent := want[key]
		if gotPresent != wantPresent || !reflect.DeepEqual(gotValue, wantValue) {
			t.Errorf("%s = %v (present %v), want %v (present %v)", key, gotValue, gotPresent, wantValue, wantPresent)
		}
	}
	results := func(body map[string]any) map[string]map[string]any {
		out := map[string]map[string]any{}
		if presets, ok := body["presets"].(map[string]any); ok {
			for name, result := range presets {
				out[name] = result.(map[string]any)
			}
		}
		if custom, ok := body["custom"].(map[string]any); ok {
			out["custom"] = custom
		}
		return out
	}
	gotResults, wantResults := results(got), results(want)
	if len(gotResults) != len(wantResults) {
		t.Fatalf("results %v, want %v", gotResults, wantResults)
	}
	for name, wantResult := range wantResults {
		gotResult := gotResults[name]
		for _, key := range []string{"model_choice", "provenance", "value_usd", "cost_weight"} {
			gotValue, gotPresent := gotResult[key]
			wantValue, wantPresent := wantResult[key]
			if gotPresent != wantPresent || !reflect.DeepEqual(gotValue, wantValue) {
				t.Errorf("%s.%s = %v (present %v), want %v (present %v)", name, key, gotValue, gotPresent, wantValue, wantPresent)
			}
		}
	}
}

func TestRoutesPredictionsReproduceTheContractFixture(t *testing.T) {
	var want struct {
		Body map[string]any `json:"body"`
	}
	readJSONFixture(t, "response_predictions.json", &want)
	got := renderRoutesPayload(t, predictionsRouting(t), predictionsFixtureDecision(t))
	assertPredictionFields(t, got, want.Body)
	// Offered rows only, in the selector's order: the fixture's 13 of its
	// table's rows.
	if candidates := got["candidates"].([]any); len(candidates) != 13 {
		t.Fatalf("%d candidates", len(candidates))
	}
}

// Off a boundary the service states stage_one null: candidates, the
// context and every model_choice answer null, and provenance stays.
func TestRoutesPredictionsOnAHeldTurn(t *testing.T) {
	var want struct {
		Body map[string]any `json:"body"`
	}
	readJSONFixture(t, "response_predictions_held_turn.json", &want)
	decision := predictionsFixtureDecision(t)
	decision.Predictions = &raylinearc.PolicyDecisionPredictions{ActionHorizon: decision.Predictions.ActionHorizon}
	for index := range decision.Results {
		decision.Results[index].StageOneCandidate = nil
	}
	assertPredictionFields(t, renderRoutesPayload(t, predictionsRouting(t), decision), want.Body)
}

// A service that does not advertise predictions: the echo says
// unsupported, lists response_fields, and no prediction key is written.
func TestRoutesPredictionsUnsupported(t *testing.T) {
	var want struct {
		Body map[string]any `json:"body"`
	}
	readJSONFixture(t, "response_predictions_unsupported.json", &want)
	decision := predictionsFixtureDecision(t)
	decision.Predictions, decision.PredictionsApplied = nil, false
	assertPredictionFields(t, renderRoutesPayload(t, predictionsRouting(t), decision), want.Body)
}

// A pick naming a row the table does not offer is refused, never answered.
func TestRoutesPredictionsRefuseAPickOffTheOffer(t *testing.T) {
	decision := predictionsFixtureDecision(t)
	notOffered := 0
	decision.Results[0].StageOneCandidate = &notOffered
	if err := raylineRoutesAddResults(&raylineRoutesResponse{}, predictionsRouting(t), decision); err == nil {
		t.Fatal("a pick naming a row off the offer was answered")
	}
	decision = predictionsFixtureDecision(t)
	decision.Results[1].StageOneCandidate = nil
	if err := raylineRoutesAddResults(&raylineRoutesResponse{}, predictionsRouting(t), decision); err == nil {
		t.Fatal("a pick naming no row on a boundary was answered")
	}
}

// point_id is the policy's: sha256 of Python's canonical JSON of the weight.
func TestRoutesPointIDIsThePolicys(t *testing.T) {
	for _, tc := range []struct {
		value float64
		repr  string
	}{
		{10, "10.0"},
		{0.5, "0.5"},
		{1 / 4.5, "0.2222222222222222"},
		{1e16, "1e+16"},
		{1.5e-5, "1.5e-05"},
		{0.0001, "0.0001"},
		{123456789012345.6, "123456789012345.6"},
		{2.5e20, "2.5e+20"},
	} {
		if got := pythonFloatRepr(tc.value); got != tc.repr {
			t.Errorf("repr(%g) = %s, want %s", tc.value, got, tc.repr)
		}
	}
}

// The selector forwards response_fields only to a service advertising
// predictions, checks the table against its picks, and records it.
func TestRoutesPredictionsReachTheDecideRequest(t *testing.T) {
	advertised := []string{
		raylinearc.PolicyCapabilityDecisionsV2, raylinearc.PolicyCapabilityTierPoints,
		raylinearc.PolicyCapabilityResponseFields, raylinearc.PolicyCapabilityExploration,
		raylinearc.PolicyCapabilityPredictions,
	}
	// The readiness probe reads the listing's capabilities onto the scorer,
	// as the cell's arming does.
	probe := func(fixture *policySelectorFixture) {
		armed := fixture.selector.armed.Load()
		scorer := armed.scorer.(*policyServiceScorer)
		if err := requireRaylineARCPolicyServing(context.Background(), armed.policy,
			fixture.decision.Algorithm.RaylineARC.PolicyService, nil, scorer.setAdvertised); err != nil {
			t.Fatalf("probe: %v", err)
		}
	}
	sentFields := func(fixture *policySelectorFixture) []any {
		fields, _ := decodeJSONObject(t, fixture.fake.receivedBodies()[0])["response_fields"].([]any)
		return fields
	}
	t.Run("advertised", func(t *testing.T) {
		fixture := routesFixture(t)
		row := 1
		fixture.fake.advertised, fixture.fake.predictionsRow = advertised, &row
		probe(fixture)
		_, controls := routesControlsFor(t, fixture, `{"value_usd":2,"response_fields":["predictions"]}`)
		result, err := selectWithControls(t, fixture, controls)
		if err != nil {
			t.Fatalf("select: %v", err)
		}
		if !slices.Contains(sentFields(fixture), any("predictions")) {
			t.Fatalf("response_fields %v", sentFields(fixture))
		}
		trace := result.RaylineARC
		if !trace.RoutePredictionsApplied || trace.RoutePredictions == nil || trace.RoutePredictions.StageOne == nil {
			t.Fatalf("predictions not recorded: %+v", trace.RoutePredictions)
		}
		// Each alternative names the action its score came from.
		if len(trace.PolicyArmActions) == 0 || trace.PolicyArmActions[0].ActionID == "" {
			t.Fatalf("arm actions %+v", trace.PolicyArmActions)
		}
	})
	t.Run("not advertised", func(t *testing.T) {
		fixture := routesFixture(t)
		row := 1
		fixture.fake.predictionsRow = &row
		probe(fixture)
		_, controls := routesControlsFor(t, fixture, `{"value_usd":2,"response_fields":["predictions"]}`)
		result, err := selectWithControls(t, fixture, controls)
		if err != nil {
			t.Fatalf("select: %v", err)
		}
		if slices.Contains(sentFields(fixture), any("predictions")) || result.RaylineARC.RoutePredictionsApplied {
			t.Fatalf("predictions asked of a service that does not advertise them: %v", sentFields(fixture))
		}
	})
	t.Run("pick off the offer", func(t *testing.T) {
		fixture := routesFixture(t)
		row := 0
		fixture.fake.advertised, fixture.fake.predictionsRow = advertised, &row
		probe(fixture)
		_, controls := routesControlsFor(t, fixture, `{"value_usd":2,"response_fields":["predictions"]}`)
		_, err := selectWithControls(t, fixture, controls)
		if !errors.Is(selectionFailureError(err), routerruntime.ErrRouteDecisionInconsistent) {
			t.Fatalf("err = %v, want inconsistent", err)
		}
	})
}
