package extproc

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// newDecisionsV2Fixture is a policy cell with this decisions_v2 block.
func newDecisionsV2Fixture(t *testing.T, v2 *config.RaylineARCDecisionsV2Config) *policySelectorFixture {
	t.Helper()
	fixture := newPolicySelectorFixture(t, "")
	fixture.decision.Algorithm.RaylineARC.PolicyService.DecisionsV2 = v2
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy: raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
			BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
		}),
	})
	policy := fixture.decision.Algorithm.RaylineARC.PolicyService
	actionID := policy.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	// Each tier pick names its action's bound model, as the service's
	// catalog does.
	fixture.fake.tierPickModel = func(picked string) string {
		for _, binding := range policy.Bindings {
			if binding.ActionID == picked {
				return binding.Model
			}
		}
		return ""
	}
	return fixture
}

// fixtureDecisionsV2 is the decisions_v2 block that sends what pathfinder's
// Decisions API v2 request fixture carries.
func fixtureDecisionsV2() *config.RaylineARCDecisionsV2Config {
	weight := func(value float64) *float64 { return &value }
	return &config.RaylineARCDecisionsV2Config{
		CostWeight: weight(1),
		TierPoints: []config.RaylineARCTierPoint{
			{Name: "swift", CostWeight: weight(2)}, {Name: "keen", CostWeight: weight(1)}, {Name: "sage", CostWeight: weight(0)},
		},
		ResponseFields: []string{config.RaylineARCResponseFieldPricePerMtok, config.RaylineARCResponseFieldThinking},
	}
}

func decodeJSONObject(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal(body, &value); err != nil {
		t.Fatal(err)
	}
	return value
}

// A cell without decisions_v2 sends the v1 request: a null cost weight and
// neither tier_points nor response_fields. Its trace and log record no
// operating point.
func TestPolicyCellWithoutDecisionsV2SendsNoV2Members(t *testing.T) {
	fixture := newDecisionsV2Fixture(t, nil)
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	body := decodeJSONObject(t, fixture.fake.receivedBodies()[0])
	selection := body["selection"].(map[string]any)
	if _, sent := body["response_fields"]; sent {
		t.Fatal("response_fields sent without decisions_v2")
	}
	if _, sent := selection["tier_points"]; sent {
		t.Fatal("tier_points sent without decisions_v2")
	}
	if weight, sent := selection["operating_point"].(map[string]any)["cost_weight"]; !sent || weight != nil {
		t.Fatalf("cost_weight = %v, want null", weight)
	}
	if result.RaylineARC.PolicyOperatingPoint != nil || result.RaylineARC.PolicyTierPicks != nil {
		t.Fatalf("trace recorded %+v / %+v", result.RaylineARC.PolicyOperatingPoint, result.RaylineARC.PolicyTierPicks)
	}
	fields := map[string]interface{}{}
	addPolicyDecisionsV2Fields(fields, result.RaylineARC)
	if len(fields) != 0 {
		t.Fatalf("selection log fields %v", fields)
	}
}

// A decisions_v2 cell sends exactly its configured members, which here are
// the pathfinder fixture's; it serves the decision's own action, and records
// the echoed operating point and the configured tiers' picks.
func TestPolicyCellSendsDecisionsV2MembersAndRecordsTheEcho(t *testing.T) {
	fixture := newDecisionsV2Fixture(t, fixtureDecisionsV2())
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	raw, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/decision_request_decisions_v2.v1.json")
	if err != nil {
		t.Fatal(err)
	}
	want := decodeJSONObject(t, raw)
	got := decodeJSONObject(t, fixture.fake.receivedBodies()[0])
	wantSelection, gotSelection := want["selection"].(map[string]any), got["selection"].(map[string]any)
	if !reflect.DeepEqual(gotSelection["tier_points"], wantSelection["tier_points"]) {
		t.Fatalf("tier_points %v, want the fixture's %v", gotSelection["tier_points"], wantSelection["tier_points"])
	}
	if !reflect.DeepEqual(got["response_fields"], want["response_fields"]) {
		t.Fatalf("response_fields %v, want the fixture's %v", got["response_fields"], want["response_fields"])
	}
	gotPoint, wantPoint := gotSelection["operating_point"].(map[string]any), wantSelection["operating_point"].(map[string]any)
	if gotPoint["cost_weight"] != wantPoint["cost_weight"] {
		t.Fatalf("cost_weight %v, want the fixture's %v", gotPoint["cost_weight"], wantPoint["cost_weight"])
	}
	// No member beyond the fixture's: the selection keys are the fixture's.
	for key := range gotSelection {
		if _, ok := wantSelection[key]; !ok {
			t.Fatalf("selection sends %q, which the fixture does not carry", key)
		}
	}

	served := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	if result.RaylineARC.PolicyActionID != served {
		t.Fatalf("served %s, want the decision's %s", result.RaylineARC.PolicyActionID, served)
	}
	point := result.RaylineARC.PolicyOperatingPoint
	if point == nil || point.PointID != "point-1" || point.CostWeight == nil || *point.CostWeight != 1 {
		t.Fatalf("operating point %+v", point)
	}
	picks := result.RaylineARC.PolicyTierPicks
	if len(picks) != 3 || picks["sage"].SelectedActionID != served || result.RaylineARC.PolicyTierPicksMissing != nil {
		t.Fatalf("tier picks %+v, want only swift, keen and sage", picks)
	}
	fields := map[string]interface{}{}
	addPolicyDecisionsV2Fields(fields, result.RaylineARC)
	logged := fields["policy_operating_point"].(map[string]interface{})
	if logged["point_id"] != "point-1" || logged["cost_weight"] != 1.0 {
		t.Fatalf("logged operating point %v", logged)
	}
	if _, missing := fields["policy_tier_picks_missing"]; missing {
		t.Fatalf("complete picks logged as missing: %v", fields)
	}
	if loggedPicks := fields["policy_tier_picks"].(map[string]interface{}); len(loggedPicks) != 3 {
		t.Fatalf("logged tier picks %v", loggedPicks)
	}
}

// The Decisions API v2 refusals fail the turn as the cell's configuration
// against its package: not undecidable (the caller's 422), not held, and
// unavailable to the caller. A decision echoed at another cost weight is not
// the cell's tier and fails too.
func TestPolicyCellFailsOnDecisionsV2Refusals(t *testing.T) {
	for _, tc := range []struct{ code, reason string }{
		{"selection_refused", "operating_point_out_of_bounds"},
		{"selection_refused", "operating_point_not_adjustable"},
		{"unsupported_request", "tier_points_unsupported"},
	} {
		fixture := newDecisionsV2Fixture(t, fixtureDecisionsV2())
		bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
		state, _ := raylinearc.NewEpisodeState(2)
		servedTurn(t, fixture, state, bindings[1].ActionID)
		fixture.fake.mu.Lock()
		fixture.fake.failWith, fixture.fake.failDetail = tc.code, map[string]any{"reason": tc.reason}
		fixture.fake.mu.Unlock()
		_, err := fixture.selectOn(t, state, nextTurn(t))
		class := selectionFailureClassOf(t, err)
		if class != "policy_service_"+raylinearc.PolicyDecisionsV2RefusedClass {
			t.Fatalf("%s: class %s", tc.reason, class)
		}
		if public := publicSelectionFailureClass(class); public != selectionFailureUnavailable {
			t.Fatalf("%s: public class %s, want %s", tc.reason, public, selectionFailureUnavailable)
		}
	}

	other := 3.0
	for name, echo := range map[string]func(*fakePolicyService){
		"another cost weight": func(fake *fakePolicyService) { fake.echoCostWeight = &other },
		"a null cost weight":  func(fake *fakePolicyService) { fake.echoNoCostWeight = true },
	} {
		fixture := newDecisionsV2Fixture(t, fixtureDecisionsV2())
		echo(fixture.fake)
		state, _ := raylinearc.NewEpisodeState(2)
		_, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
		if class := selectionFailureClassOf(t, err); class != arcFailurePolicyOperatingPointMismatch {
			t.Fatalf("echo at %s: class %s", name, class)
		}
	}
	// V 0 echoed as 0 is the cell's point, not a missing one.
	zero := fixtureDecisionsV2()
	weight := 0.0
	zero.CostWeight = &weight
	fixture := newDecisionsV2Fixture(t, zero)
	state, _ := raylinearc.NewEpisodeState(2)
	if _, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"})); err != nil {
		t.Fatalf("V 0: %v", err)
	}
}

// A configured tier the response has no pick for is named as missing, so the
// logged picks are never read as a complete set; the turn is still served.
func TestPolicyCellNamesAMissingTierPick(t *testing.T) {
	fixture := newDecisionsV2Fixture(t, fixtureDecisionsV2())
	fixture.fake.omitTierPick = "keen"
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	if len(result.RaylineARC.PolicyTierPicks) != 2 || !reflect.DeepEqual(result.RaylineARC.PolicyTierPicksMissing, []string{"keen"}) {
		t.Fatalf("picks %+v, missing %v", result.RaylineARC.PolicyTierPicks, result.RaylineARC.PolicyTierPicksMissing)
	}
	fields := map[string]interface{}{}
	addPolicyDecisionsV2Fields(fields, result.RaylineARC)
	if !reflect.DeepEqual(fields["policy_tier_picks_missing"], []string{"keen"}) {
		t.Fatalf("logged missing %v", fields["policy_tier_picks_missing"])
	}
}

// A tier pick the cell could not have served -- an action it does not bind or
// did not offer, or a model its binding does not serve -- is named invalid
// and not kept; the turn is still served.
func TestPolicyCellNamesInvalidTierPicks(t *testing.T) {
	for name, corrupt := range map[string]func(*policySelectorFixture){
		"unbound action": func(f *policySelectorFixture) { f.fake.tierPickAction = strings.Repeat("f", 64) },
		"another model": func(f *policySelectorFixture) {
			f.fake.tierPickModel = func(string) string { return "some-other-model" }
		},
	} {
		fixture := newDecisionsV2Fixture(t, fixtureDecisionsV2())
		corrupt(fixture)
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
		if err != nil {
			t.Fatalf("%s: select: %v", name, err)
		}
		if len(result.RaylineARC.PolicyTierPicks) != 0 ||
			!reflect.DeepEqual(result.RaylineARC.PolicyTierPicksInvalid, []string{"swift", "keen", "sage"}) {
			t.Fatalf("%s: picks %+v, invalid %v", name, result.RaylineARC.PolicyTierPicks, result.RaylineARC.PolicyTierPicksInvalid)
		}
		fields := map[string]interface{}{}
		addPolicyDecisionsV2Fields(fields, result.RaylineARC)
		if _, logged := fields["policy_tier_picks"]; logged || fields["policy_tier_picks_invalid"] == nil {
			t.Fatalf("%s: logged %v", name, fields)
		}
	}
	// An action the cell binds but did not offer this turn.
	if policyTierPickValid(&policyServiceScorer{bindings: map[string]policyBinding{"a": {}}}, []string{"b"})(
		raylinearc.PolicyTierPick{SelectedActionID: "a"}) {
		t.Fatal("an unoffered tier pick was admitted")
	}
}
