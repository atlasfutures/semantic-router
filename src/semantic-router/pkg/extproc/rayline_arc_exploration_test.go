package extproc

import (
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// fixtureExploration is the exploration block that sends what pathfinder's
// exploration request fixture carries.
func fixtureExploration() *config.RaylineARCExplorationConfig {
	value := func(v float64) *float64 { return &v }
	return &config.RaylineARCExplorationConfig{
		Tau: value(0.02), TauMax: value(0.05),
		Basis: config.RaylineARCExplorationBasis{
			R: value(0.05), PackageSHA256: "81e4a7af2c28c3be9ecbe850888c6773d196d58cad100474cbfb48622af7abe4",
			ReadoutPin: "qwen3.8-27b-arc-encoder@depth8",
		},
		Salt: "vsr-explore-2026-10",
	}
}

// newExplorationFixture is a policy cell with this exploration block, its
// signal headers trusted, and possibly marked an eval cell.
func newExplorationFixture(t *testing.T, explore *config.RaylineARCExplorationConfig, shape func(*config.RaylineARCPolicyServiceConfig)) *policySelectorFixture {
	t.Helper()
	fixture := newPolicySelectorFixture(t, "")
	policy := fixture.decision.Algorithm.RaylineARC.PolicyService
	policy.Exploration, policy.TrustTurnSignalHeaders = explore, true
	if shape != nil {
		shape(policy)
	}
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy: raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
			BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
		}),
	})
	actionID := policy.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	return fixture
}

// eligibleHeaders are what the gateway sends for a user outside privacy mode
// who is not opted out of training-data collection.
func eligibleHeaders() map[string]string {
	return map[string]string{raylineARCPrivacyModeHeader: "false", raylineARCTrainingDataHeader: "allowed"}
}

func sentSelection(t *testing.T, fixture *policySelectorFixture) map[string]any {
	t.Helper()
	bodies := fixture.fake.receivedBodies()
	return decodeJSONObject(t, bodies[len(bodies)-1])["selection"].(map[string]any)
}

func usageLine(t *testing.T, trace *selection.RaylineARCTrace) map[string]interface{} {
	t.Helper()
	logs := captureLogs(t)
	ctx := &RequestContext{RequestID: "rt_explore", RequestModel: "priced-model", VSRRaylineARC: trace}
	usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, streamingFlagUsage())
	return findLogEvent(t, logs, "llm_usage")
}

func assertNoExplorationUsage(t *testing.T, line map[string]interface{}) {
	t.Helper()
	for _, key := range []string{"router_propensity", "router_tau", "router_explored", "exploration_decision_key"} {
		if value, present := line[key]; !present || value != nil {
			t.Errorf("%s = %v (present %v), want null", key, value, present)
		}
	}
}

// A cell without exploration sends no selection.exploration, records nothing
// and leaves the llm_usage exploration members null, even for an eligible
// user.
func TestPolicyCellWithoutExplorationSendsNone(t *testing.T) {
	fixture := newExplorationFixture(t, nil, nil)
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectWithHeaders(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}), eligibleHeaders())
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	if _, sent := sentSelection(t, fixture)["exploration"]; sent {
		t.Fatal("exploration sent without the config block")
	}
	trace := result.RaylineARC
	if trace.PolicyExploration != nil || trace.PolicyExplorationSent || trace.PolicyExplorationExcluded != "" {
		t.Fatalf("trace %+v / %v / %q", trace.PolicyExploration, trace.PolicyExplorationSent, trace.PolicyExplorationExcluded)
	}
	fields := map[string]interface{}{}
	addPolicyExplorationFields(fields, trace)
	if len(fields) != 0 {
		t.Fatalf("selection log fields %v", fields)
	}
	assertNoExplorationUsage(t, usageLine(t, trace))
}

// An eligible user's turn sends exactly the pathfinder fixture's member; the
// service's record lands whole on the selection log and its propensity, tau,
// explored and decision key on llm_usage.
func TestPolicyCellSendsExplorationForAnEligibleUserAndRecordsIt(t *testing.T) {
	fixture := newExplorationFixture(t, fixtureExploration(), nil)
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectWithHeaders(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}), eligibleHeaders())
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	raw, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/decision_request_exploration.v1.json")
	if err != nil {
		t.Fatal(err)
	}
	want := decodeJSONObject(t, raw)["selection"].(map[string]any)["exploration"]
	if got := sentSelection(t, fixture)["exploration"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("exploration %v, want the fixture's %v", got, want)
	}

	trace := result.RaylineARC
	record := trace.PolicyExploration
	if !trace.PolicyExplorationSent || trace.PolicyExplorationExcluded != "" || record == nil ||
		!record.Applied || *record.Propensity != 0.3318122278318338 || !*record.Explored {
		t.Fatalf("trace sent %v excluded %q record %+v", trace.PolicyExplorationSent, trace.PolicyExplorationExcluded, record)
	}
	fields := map[string]interface{}{}
	addPolicyExplorationFields(fields, trace)
	logged, ok := fields["exploration"].(map[string]interface{})
	if !ok || fields["exploration_sent"] != true {
		t.Fatalf("selection log fields %v", fields)
	}
	// The whole record, under the contract's names.
	for _, key := range []string{
		"applied", "reason", "tau", "requested_tau", "basis", "salt", "u", "offered", "scores", "pi",
		"greedy", "chosen", "propensity", "explored", "package_sha256", "decision_key", "held",
		"origin_decision_key", "override",
	} {
		if _, present := logged[key]; !present {
			t.Errorf("logged record lacks %s", key)
		}
	}
	if len(logged) != 19 || logged["override"] != "proxy_remap" {
		t.Fatalf("logged record %v", logged)
	}

	line := usageLine(t, trace)
	if line["router_propensity"] != 0.3318122278318338 || line["router_tau"] != 0.02 || line["router_explored"] != true ||
		line["exploration_decision_key"] != *record.DecisionKey {
		t.Fatalf("llm_usage exploration members %v %v %v %v",
			line["router_propensity"], line["router_tau"], line["router_explored"], line["exploration_decision_key"])
	}
}

// A held turn's record names no decision key of its own: llm_usage joins it
// to its boundary through origin_decision_key. A tau-0 cell sends the member
// and gets no record, as greedy, and a request that asked for tau > 0 and got
// none back is marked sent, never read as greedy.
func TestPolicyCellExplorationHeldAndRecordlessTurns(t *testing.T) {
	fixture := newExplorationFixture(t, fixtureExploration(), nil)
	fixture.fake.explorationHeld = true
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectWithHeaders(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}), eligibleHeaders())
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	line := usageLine(t, result.RaylineARC)
	origin := result.RaylineARC.PolicyExploration.OriginDecisionKey
	if origin == nil || line["exploration_decision_key"] != *origin || line["router_propensity"] != 1.0 ||
		line["router_tau"] != 0.0 || line["router_explored"] != false {
		t.Fatalf("held llm_usage %v", line)
	}

	zero := fixtureExploration()
	tau := 0.0
	zero.Tau = &tau
	for name, fixture := range map[string]*policySelectorFixture{
		"tau 0":          newExplorationFixture(t, zero, nil),
		"missing record": newExplorationFixture(t, fixtureExploration(), nil),
	} {
		fixture.fake.omitExploration = true
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectWithHeaders(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}), eligibleHeaders())
		if err != nil {
			t.Fatalf("%s: select: %v", name, err)
		}
		if _, sent := sentSelection(t, fixture)["exploration"]; !sent {
			t.Fatalf("%s: exploration not sent", name)
		}
		fields := map[string]interface{}{}
		addPolicyExplorationFields(fields, result.RaylineARC)
		if fields["exploration_sent"] != true || fields["exploration"] != nil {
			t.Fatalf("%s: selection log fields %v", name, fields)
		}
		assertNoExplorationUsage(t, usageLine(t, result.RaylineARC))
	}
}

// Exploration is limited to users who are not opted out of training-data
// collection, never in privacy mode and never on an eval cell. Each fact
// fails closed: a request that does not state it is excluded too.
func TestPolicyCellExcludesIneligibleTurnsFromExploration(t *testing.T) {
	with := func(change func(map[string]string)) map[string]string {
		headers := eligibleHeaders()
		change(headers)
		return headers
	}
	for name, tc := range map[string]struct {
		shape   func(*config.RaylineARCPolicyServiceConfig)
		headers map[string]string
		want    string
	}{
		"privacy mode":          {nil, with(func(h map[string]string) { h[raylineARCPrivacyModeHeader] = "true" }), explorationExcludedPrivacyMode},
		"privacy mode garbled":  {nil, with(func(h map[string]string) { h[raylineARCPrivacyModeHeader] = "no" }), explorationExcludedPrivacyMode},
		"privacy mode unstated": {nil, with(func(h map[string]string) { delete(h, raylineARCPrivacyModeHeader) }), explorationExcludedPrivacyUnknown},
		"privacy mode empty":    {nil, with(func(h map[string]string) { h[raylineARCPrivacyModeHeader] = "" }), explorationExcludedPrivacyUnknown},
		"opted out":             {nil, with(func(h map[string]string) { h[raylineARCTrainingDataHeader] = "opted_out" }), explorationExcludedTrainingOptOut},
		"consent unstated":      {nil, with(func(h map[string]string) { delete(h, raylineARCTrainingDataHeader) }), explorationExcludedTrainingUnknown},
		"no headers":            {nil, nil, explorationExcludedPrivacyUnknown},
		"eval cell": {
			func(p *config.RaylineARCPolicyServiceConfig) { p.EvalCell = true },
			eligibleHeaders(), explorationExcludedEvalCell,
		},
		"untrusted headers": {
			func(p *config.RaylineARCPolicyServiceConfig) { p.TrustTurnSignalHeaders = false },
			eligibleHeaders(), explorationExcludedUntrustedHeaders,
		},
		"side call": {nil, with(func(h map[string]string) { h[raylineARCCallKindHeader] = "side" }), explorationExcludedSideCall},
	} {
		fixture := newExplorationFixture(t, fixtureExploration(), tc.shape)
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectWithHeaders(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}), tc.headers)
		if err != nil {
			t.Fatalf("%s: select: %v", name, err)
		}
		if _, sent := sentSelection(t, fixture)["exploration"]; sent {
			t.Fatalf("%s: exploration sent", name)
		}
		trace := result.RaylineARC
		if trace.PolicyExplorationExcluded != tc.want || trace.PolicyExplorationSent || trace.PolicyExploration != nil {
			t.Fatalf("%s: excluded %q (want %q), sent %v", name, trace.PolicyExplorationExcluded, tc.want, trace.PolicyExplorationSent)
		}
		fields := map[string]interface{}{}
		addPolicyExplorationFields(fields, trace)
		if fields["exploration_excluded"] != tc.want || len(fields) != 1 {
			t.Fatalf("%s: selection log fields %v", name, fields)
		}
		assertNoExplorationUsage(t, usageLine(t, trace))
	}
}

// The eligibility headers are read only from a trusted gateway, as the turn
// signals are, and never forwarded upstream.
func TestExplorationHeadersAreTrustedSignalsAndNeverForwarded(t *testing.T) {
	request := map[string]string{raylineARCPrivacyModeHeader: "false", raylineARCTrainingDataHeader: "allowed"}
	trusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{TrustTurnSignalHeaders: true}, request)
	if trusted[raylineARCPrivacyModeHeader] != "false" || trusted[raylineARCTrainingDataHeader] != "allowed" {
		t.Fatalf("trusted headers %v", trusted)
	}
	if untrusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{}, request); len(untrusted) != 0 {
		t.Fatalf("untrusted headers read: %v", untrusted)
	}
	removed := raylineARCTurnSignalHeadersForRemoval()
	if !slices.Contains(removed, raylineARCPrivacyModeHeader) || !slices.Contains(removed, raylineARCTrainingDataHeader) {
		t.Fatalf("headers forwarded upstream: removal list %v", removed)
	}
}

// A record whose usage members are null or missing keeps them null on
// llm_usage: a propensity nobody stated is never 0, and explored never false.
func TestExplorationUsageKeepsNullRecordFieldsNull(t *testing.T) {
	for _, body := range []string{
		`{"tau": null, "propensity": null, "explored": null, "decision_key": null}`,
		`{}`,
	} {
		var record raylinearc.PolicyExplorationRecord
		if err := json.Unmarshal([]byte(body), &record); err != nil {
			t.Fatal(err)
		}
		assertNoExplorationUsage(t, usageLine(t, &selection.RaylineARCTrace{PolicyExploration: &record}))
	}
}

// A two-stage cold turn whose attested re-ask is refused for its exploration
// member fails on that refusal: it is never asked again narrowed, where a
// greedy-looking success would hide the cell's misconfiguration.
func TestAttestedAskRefusedForExplorationIsNotReaskedNarrowed(t *testing.T) {
	fixture := newExplorationFixture(t, fixtureExploration(), nil)
	asTwoStage(fixture)
	fixture.fake.refuseCold = func(request raylinearc.PolicyDecisionRequest) bool {
		return request.Selection.AttestedHeldActionID == nil && len(request.Attribution) == 0
	}
	fixture.fake.refuseAttestedWith = "exploration_basis_mismatch"
	state, _ := raylinearc.NewEpisodeState(2)
	held := 0
	state.PreviousArm, state.TurnIndex = &held, 2
	state.Policy = &raylinearc.PolicyEpisodeState{PrefixDigest: raylinearc.MessagesDigest(nil, 0)}
	_, err := fixture.selectWith(t, state, typedFollowUp(t), func(arc *selection.RaylineARCSelectionContext) {
		arc.ServedModels = []selection.RaylineARCServedModel{{Source: derivedHoldSessionRecord, Model: "vendor/off"}}
		arc.PolicySignalHeaders = eligibleHeaders()
	})
	if class := selectionFailureClassOf(t, err); class != "policy_service_"+raylinearc.PolicyExplorationRefusedClass {
		t.Fatalf("class %s", class)
	}
	calls := fixture.fake.received()
	if len(calls) != 2 || calls[1].Selection.AttestedHeldActionID == nil || calls[1].Selection.Exploration == nil {
		t.Fatalf("%d calls, offers %v; want the cold refusal and the attested ask only", len(calls), offersOf(calls))
	}
}

// The exploration refusals fail an eligible turn as the cell's configuration
// against its service and package, as decisions_v2 refusals do: not
// undecidable (the caller's 422), not held, unavailable to the caller, and the
// failure log names the reason. An excluded turn sends no member and is served.
func TestPolicyCellFailsOnExplorationRefusals(t *testing.T) {
	for _, reason := range []string{
		"exploration_tau_exceeds_ceiling", "exploration_basis_mismatch", "exploration_unsupported_by_package",
	} {
		fixture := newExplorationFixture(t, fixtureExploration(), nil)
		bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
		state, _ := raylinearc.NewEpisodeState(2)
		servedTurn(t, fixture, state, bindings[1].ActionID)
		fixture.fake.mu.Lock()
		fixture.fake.failWith, fixture.fake.failDetail = "selection_refused", map[string]any{"reason": reason}
		fixture.fake.mu.Unlock()
		logs := captureLogs(t)
		_, err := fixture.selectWithHeaders(t, state, nextTurn(t), eligibleHeaders())
		class := selectionFailureClassOf(t, err)
		if class != "policy_service_"+raylinearc.PolicyExplorationRefusedClass {
			t.Fatalf("%s: class %s", reason, class)
		}
		if public := publicSelectionFailureClass(class); public != selectionFailureUnavailable {
			t.Fatalf("%s: public class %s, want %s", reason, public, selectionFailureUnavailable)
		}
		failed := findLogEvent(t, logs, "rayline_arc_policy_service_failed")
		if failed["class"] != raylinearc.PolicyExplorationRefusedClass || failed["reason"] != reason {
			t.Fatalf("%s: failure log %v", reason, failed)
		}
	}
}
