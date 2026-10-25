package extproc

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The v2 golden vectors are pathfinder's (testdata/arc_public_contract/
// SOURCE.md), produced by arc_execution_v2.parse_execution_v2, bind_v2 and
// decide_inputs_v2; the adapter must reproduce them.
type executionV2Vectors struct {
	Package struct {
		PackageSHA256          string            `json:"package_sha256"`
		ThinkingControlsSHA256 string            `json:"thinking_controls_sha256"`
		OfferedActionIDs       []string          `json:"offered_action_ids"`
		ActionModels           map[string]string `json:"action_models"`
	} `json:"package"`
	Vectors []struct {
		Name      string          `json:"name"`
		Execution json.RawMessage `json:"execution"`
		Expect    struct {
			Refused string          `json:"refused"`
			Decide  json.RawMessage `json:"decide"`
		} `json:"expect"`
	} `json:"vectors"`
}

func readExecutionV2Vectors(t *testing.T) executionV2Vectors {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(arcPublicContractDir, "execution_v2_vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var vectors executionV2Vectors
	if err := json.Unmarshal(raw, &vectors); err != nil {
		t.Fatal(err)
	}
	return vectors
}

// The body the vectors index: assistant messages at 1, 3 and 5.
const executionV2Messages = `[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"},` +
	`{"role":"assistant","content":"d"},{"role":"user","content":"e"},{"role":"assistant","content":"f"},{"role":"user","content":"g"}]`

func executionV2Body(routing string) []byte {
	return []byte(`{"model":"arc-1.0","messages":` + executionV2Messages + `,"routing":` + routing + `}`)
}

// decideMembers is the decide request's v2 members as the vectors state
// them: the reference omits an attribution's null arm_id; VSR's form states
// it null. Both say unknown.
func decideMembers(t *testing.T, request *raylinearc.PolicyDecisionRequest, notAdmitted []routesExecutionNotAdmitted) any {
	t.Helper()
	attribution := []map[string]any{}
	for _, entry := range request.Attribution {
		row := map[string]any{"message": entry.Message, "action_id": entry.ActionID}
		if entry.ArmID != nil {
			row["arm_id"] = *entry.ArmID
		}
		attribution = append(attribution, row)
	}
	selected := map[string]string{}
	if id := request.Selection.HeldActionID; id != nil {
		selected["held_action_id"] = *id
	}
	if id := request.Selection.LastServedActionID; id != nil {
		selected["last_served_action_id"] = *id
	}
	capacity := []raylinearc.PolicyActionCapacity{}
	capacity = append(capacity, request.Capacity...)
	refused := []map[string]string{}
	for _, entry := range notAdmitted {
		refused = append(refused, map[string]string{"action_id": entry.ActionID, "field": entry.Field, "reason": entry.Reason})
	}
	members := map[string]any{
		"context_epoch": request.ContextEpoch, "attribution": attribution, "selection": selected,
		"cache_context": request.CacheContext, "event": request.Event, "capacity": capacity,
		"turn_key": request.TurnKey, "reconsult": request.Reconsult, "not_admitted": refused,
	}
	return jsonValue(t, executionJSON(t, members))
}

func TestRoutesExecutionV2ReproducesTheGoldenVectors(t *testing.T) {
	vectors := readExecutionV2Vectors(t)
	scorer := &policyServiceScorer{
		sha256: vectors.Package.PackageSHA256, thinkingControls: vectors.Package.ThinkingControlsSHA256,
		bindings: map[string]policyBinding{},
	}
	arm := 0
	for actionID, model := range vectors.Package.ActionModels {
		scorer.bindings[actionID] = policyBinding{arm: arm, model: model}
		arm++
	}
	for _, vector := range vectors.Vectors {
		t.Run(vector.Name, func(t *testing.T) {
			routing, _, refusal := readRaylineRoutesRouting(executionV2Body(`{"execution":`+string(vector.Execution)+`}`), nil)
			if refusal != nil {
				if refusal.code != vector.Expect.Refused {
					t.Fatalf("parse refused %s (%s), want %q", refusal.code, refusal.message, vector.Expect.Refused)
				}
				return
			}
			controls := raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
			request := &raylinearc.PolicyDecisionRequest{
				ContextEpoch: "0", Selection: raylinearc.PolicySelection{AvailableActionIDs: vectors.Package.OfferedActionIDs},
			}
			class := applyRoutePolicyAndExecutionV2(request, scorer, controls)
			if vector.Expect.Refused != "" {
				if class == "" || routeExecutionClasses[class].code != vector.Expect.Refused {
					t.Fatalf("class %q, want %s", class, vector.Expect.Refused)
				}
				return
			}
			if class != "" {
				t.Fatalf("refused %s", class)
			}
			if request.EpisodeMode != raylinearc.PolicyEpisodeModeRelaxed {
				t.Fatalf("episode mode %q", request.EpisodeMode)
			}
			request.CacheContext = routeExecutionV2CacheContext(controls.ExecutionV2)
			got, want := decideMembers(t, request, routing.executionNotAdmitted), jsonValue(t, vector.Expect.Decide)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("decide\n got %s\nwant %s", executionJSON(t, got), vector.Expect.Decide)
			}
		})
	}
}

// routing.policy: the preset and schedule override pass through as sent;
// the contract's invalid policies and a generation request are refused.
func TestRoutesPolicy(t *testing.T) {
	routing, _, refusal := readRaylineRoutesRouting(executionV2Body(
		`{"policy":{"preset":"arc-policy.hybrid-default.v1","overrides":{"schedule":{"model":["cache_boundaries","user_messages"]}}}}`), nil)
	if refusal != nil {
		t.Fatal(refusal)
	}
	controls := raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
	encoded := string(executionJSON(t, controls.Policy))
	if encoded != `{"preset":"arc-policy.hybrid-default.v1","overrides":{"schedule":{"model":["cache_boundaries","user_messages"]}}}` {
		t.Fatalf("policy member %s", encoded)
	}
	for name, body := range map[string]string{ //nolint:gosec // request bodies, not credentials
		"null member":        `{"policy":{"preset":"arc-policy.hybrid-default.v1","overrides":null}}`,
		"repeated class":     `{"policy":{"preset":"arc-policy.hybrid-default.v1","overrides":{"schedule":{"model":["all_turns","all_turns"]}}}}`,
		"unknown member":     `{"policy":{"preset":"arc-policy.hybrid-default.v1","dsl":"x"}}`,
		"missing preset":     `{"policy":{}}`,
		"unknown preset":     `{"policy":{"preset":"arc-policy.prefer-local.v2"}}`,
		"selection off rule": `{"policy":{"preset":"arc-policy.hybrid-default.v1","overrides":{"selection":{"min_p_pass":0.5}}}}`,
		"p_pass above 1":     `{"policy":{"preset":"arc-policy.prefer-local.v1","overrides":{"selection":{"min_p_pass":1.5}}}}`,
		"zero rest tokens":   `{"policy":{"preset":"arc-policy.prefer-local.v1","overrides":{"selection":{"max_rest_tokens":0}}}}`,
	} {
		if _, _, refusal := readRaylineRoutesRouting(executionV2Body(body), nil); refusal == nil || refusal.code != "policy_invalid" {
			t.Errorf("%s: refusal %+v", name, refusal)
		}
	}
	fixture := readContractFixture(t, "request_invalid_policy_generation.json")
	if _, _, refusal := readRaylineGenerationRouting(fixture.Body, generationPolicy(true, nil)); refusal == nil || refusal.code != fixture.Expect.Code {
		t.Fatalf("generation: %+v, want %s", refusal, fixture.Expect.Code)
	}
}

// A marker-inferred retained-tail compaction is admitted (pathfinder
// #5907): the old explicit_client-only rule is not mirrored.
func TestRoutesExecutionV2AdmitsMarkerInferredRetainedTail(t *testing.T) {
	vectors := readExecutionV2Vectors(t)
	for _, vector := range vectors.Vectors {
		if vector.Name != "marker_inferred_retained_tail_compaction" {
			continue
		}
		if _, _, refusal := readRaylineRoutesRouting(executionV2Body(`{"execution":`+string(vector.Execution)+`}`), nil); refusal != nil {
			t.Fatalf("refused %+v", refusal)
		}
		return
	}
	t.Fatal("no marker_inferred_retained_tail_compaction vector")
}

// Through the selector: routing.policy and a v2 envelope reach the decide
// request as its v2 members, the receipt is recorded, and a hold the
// request's own capacity marks infeasible is refused before deciding.
func TestRouteExecutionV2ReachesTheDecideRequest(t *testing.T) {
	fixture := executionFixture(t, true)
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	think, off := bindings[0].ActionID, bindings[2].ActionID
	model := "schedule"
	fixture.fake.receipt = &raylinearc.PolicyDecisionReceipt{DecidedBy: raylinearc.PolicyReceiptDecidedBy{Model: &model, Thinking: "held"}}
	_, controls := routesControlsFor(t, fixture, `{"value_usd":1.0}`)
	kind := "tool_result"
	fitsFalse, reason := false, "context_capacity"
	controls.Policy = &raylinearc.PolicyMember{Preset: "arc-policy.hybrid-default.v1"}
	controls.ExecutionV2 = &selection.RaylineARCRouteExecutionV2{
		ContextEpoch: "e1", Attribution: []raylinearc.PolicyAttribution{}, LastServedActionID: &think,
		Event:    raylinearc.PolicyExecutorEvent{Kind: &kind, Basis: "explicit_client"},
		Capacity: []raylinearc.PolicyActionCapacity{{ActionID: think, Placement: "local", Fits: &fitsFalse, Reason: &reason}},
		TurnKey:  routesTurnKey("t1"), ActionIDs: []string{think},
	}
	result, err := selectWithControls(t, fixture, controls)
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := fixture.fake.received()[0]
	if sent.Policy == nil || sent.Policy.Preset != "arc-policy.hybrid-default.v1" || sent.Event == nil || *sent.Event.Kind != "tool_result" ||
		len(sent.Capacity) != 1 || sent.TurnKey != routesTurnKey("t1") || sent.Selection.LastServedActionID == nil ||
		*sent.Selection.LastServedActionID != think || sent.EpisodeMode != raylinearc.PolicyEpisodeModeRelaxed ||
		slices.Contains(sent.ResponseFields, "cache_trace") {
		t.Fatalf("decide request %+v", sent)
	}
	if result.RaylineARC.RouteReceipt == nil || *result.RaylineARC.RouteReceipt.DecidedBy.Model != "schedule" {
		t.Fatalf("receipt %+v", result.RaylineARC.RouteReceipt)
	}

	fixture = executionFixture(t, false)
	_, controls = routesControlsFor(t, fixture, `{}`)
	controls.ExecutionV2 = &selection.RaylineARCRouteExecutionV2{
		ContextEpoch: "e1", HoldActionID: &off, Infeasible: []string{off}, ActionIDs: []string{off},
		Event: raylinearc.PolicyExecutorEvent{Basis: "unknown"},
	}
	_, err = selectWithControls(t, fixture, controls)
	var refusal *routeExecutionRefusal
	if !errors.As(selectionFailureError(err), &refusal) || refusal.contract().code != "execution_hold_infeasible" || refusal.contract().status != 409 {
		t.Fatalf("err = %v", err)
	}
	if len(fixture.fake.received()) != 0 {
		t.Fatal("a refused envelope reached decide")
	}
}

// Receipts and echoes: each result's receipt is the decide's receipt with
// its own action and control; the v2 echo is applied with a receipt and
// unsupported without one; the policy echo states the preset as sent.
func TestRoutesExecutionV2ReceiptsAndEchoes(t *testing.T) {
	vectors := readExecutionV2Vectors(t)
	routing, _, refusal := readRaylineRoutesRouting(executionV2Body(`{"policy":{"preset":"arc-policy.hybrid-default.v1"},"execution":`+
		string(vectors.Vectors[0].Execution)+`}`), nil)
	if refusal != nil {
		t.Fatal(refusal)
	}
	raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
	qwen, opus := strings.Repeat("1", 64), strings.Repeat("2", 64)
	handoff, preset := "capacity_handoff", "arc-policy.hybrid-default.v1"
	receipt := &raylinearc.PolicyDecisionReceipt{
		Policy: &preset, DecidedBy: raylinearc.PolicyReceiptDecidedBy{Model: &handoff, Thinking: "schedule"},
		HeldFrom: &qwen, Excluded: []raylinearc.PolicyReceiptExcluded{{ActionID: qwen, Reason: "context_capacity"}},
	}
	result := func(name string, weight float64) routerruntime.RouteResult {
		return routerruntime.RouteResult{
			Name: name, CostWeight: weight, SelectedWorker: "w", WorkerModel: "m/a",
			PolicyActionID: opus, ArmID: "arm", ControlID: strings.Repeat("d", 64),
		}
	}
	decision := routerruntime.RouteDecision{
		Results:        []routerruntime.RouteResult{result("swift", 10), result("sage", 1/4.5)},
		Identity:       &routerruntime.RouteIdentity{PackageSHA256: vectors.Package.PackageSHA256, OfferSHA256: "o"},
		PolicyActionID: opus, Receipt: receipt,
	}
	var payload raylineRoutesResponse
	if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
		t.Fatal(err)
	}
	swift := payload.Presets.Swift.Receipt
	if swift == nil || swift.ActionID != opus || *swift.ControlID != strings.Repeat("d", 64) || *swift.DecidedBy.Model != "capacity_handoff" ||
		len(swift.Excluded) != 1 || swift.CapacityUnknown == nil || payload.Receipt == nil || payload.Receipt.ControlID != nil {
		t.Fatalf("receipts %+v / %+v", swift, payload.Receipt)
	}
	echo := jsonValue(t, executionJSON(t, payload.Routing)).(map[string]any)
	execution := echo["execution"].(map[string]any)
	if execution["status"] != "applied" || execution["version"] != "arc-execution-context.v2" ||
		execution["turn"].(map[string]any)["epoch"] != "e1" || execution["pinned"].(map[string]any)["package_sha256"] != "matched" ||
		execution["episode_identity"] != "per_lookup_random" {
		t.Fatalf("execution echo %v", execution)
	}
	if policy := echo["policy"].(map[string]any); policy["status"] != "applied" || policy["preset"] != preset {
		t.Fatalf("policy echo %v", policy)
	}
	// No receipt: both unsupported, both unapplied, no receipts.
	decision.Receipt = nil
	payload = raylineRoutesResponse{}
	if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
		t.Fatal(err)
	}
	echo = jsonValue(t, executionJSON(t, payload.Routing)).(map[string]any)
	if echo["execution"].(map[string]any)["status"] != "unsupported" || echo["execution"].(map[string]any)["pinned"] != nil ||
		echo["policy"].(map[string]any)["status"] != "unsupported" || payload.Presets.Swift.Receipt != nil ||
		!slices.Contains(payload.Routing.UnappliedKeys, "execution") || !slices.Contains(payload.Routing.UnappliedKeys, "policy") {
		t.Fatalf("unsupported echo %v", echo)
	}
}

// Absent routing.policy and routing.execution the echo is written exactly
// as before (no re-keyed members), and no receipt appears.
func TestRoutesEchoWithoutV2IsUnchanged(t *testing.T) {
	routing, _, refusal := readRaylineRoutesRouting(executionV2Body(`{"value_usd":2.0,"temperature":0.5}`), nil)
	if refusal != nil {
		t.Fatal(refusal)
	}
	type plain raylineRoutesRoutingEcho
	two, half := 2.0, 0.5
	echo := raylineRoutesRoutingEcho{
		ValueUSD:      &raylineRoutesEchoStatus{Requested: two, Status: "applied"},
		Temperature:   &raylineRoutesEchoStatus{Requested: half, Status: "unsupported"},
		UnappliedKeys: []string{"temperature"},
	}
	if got, want := string(executionJSON(t, echo)), string(executionJSON(t, plain(echo))); got != want {
		t.Fatalf("echo bytes changed\n got %s\nwant %s", got, want)
	}
	if routing.wantsReceipt() {
		t.Fatal("a plain lookup wants receipts")
	}
}

// The policy service's v2 refusal strings map onto their public codes and
// statuses exactly as pathfinder pins them; the contract-side codes are
// the binding's, never mapped from a service reason.
func TestRouteExecutionV2ServiceRefusals(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(arcPublicContractDir, "selection_refusals_v2.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	var pinned struct {
		SelectionRefused struct {
			Reasons  []string `json:"reasons"`
			V2Public map[string]struct {
				Code   string `json:"code"`
				Status int    `json:"status"`
			} `json:"v2_public"`
		} `json:"selection_refused"`
		ContractSide []string `json:"contract_side"`
	}
	if err := json.Unmarshal(raw, &pinned); err != nil {
		t.Fatal(err)
	}
	for reason, want := range pinned.SelectionRefused.V2Public {
		class := routeExecutionServiceRefusal(&raylinearc.PolicyServiceError{Class: "selection_refused", Detail: map[string]any{"reason": reason}})
		entry, ok := routeExecutionClasses[class]
		if !ok || entry.code != want.Code || entry.status != want.Status {
			t.Errorf("%s: class %q -> %+v, want %s %d", reason, class, entry, want.Code, want.Status)
		}
	}
	// Every reason the service states survives the client's bounded reason
	// set, so the mapping above can see it.
	for _, reason := range pinned.SelectionRefused.Reasons {
		if raylinearc.PolicyRefusalReason(map[string]any{"reason": reason}) != reason {
			t.Errorf("service reason %s is bounded to other", reason)
		}
	}
	for _, code := range pinned.ContractSide {
		if class := routeExecutionServiceRefusal(&raylinearc.PolicyServiceError{Class: "selection_refused", Detail: map[string]any{"reason": code}}); class != "" &&
			code != "execution_hold_unavailable" {
			t.Errorf("contract-side %s mapped from a service reason to %s", code, class)
		}
		found := false
		for _, entry := range routeExecutionClasses {
			found = found || entry.code == code
		}
		if !found {
			t.Errorf("contract-side %s has no public refusal", code)
		}
	}
}

// prefer-local (pathfinder #5913): its selection thresholds pass through as
// sent, and a receipt's selection comparison is copied verbatim.
func TestRoutesPolicyPreferLocal(t *testing.T) {
	routing, _, refusal := readRaylineRoutesRouting(executionV2Body(
		`{"policy":{"preset":"arc-policy.prefer-local.v1","overrides":{"selection":{"min_p_pass":0.7,"max_rest_tokens":40000}}}}`), nil)
	if refusal != nil {
		t.Fatal(refusal)
	}
	controls := raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
	if got := string(executionJSON(t, controls.Policy)); got != `{"preset":"arc-policy.prefer-local.v1","overrides":{"selection":{"min_p_pass":0.7,"max_rest_tokens":40000}}}` {
		t.Fatalf("policy member %s", got)
	}
	comparison := json.RawMessage(`{"rule":"prefer_local","min_p_pass":0.7,"max_rest_tokens":40000,"local":[],"outcome":"no_local_offered"}`)
	receipt := &raylinearc.PolicyDecisionReceipt{
		DecidedBy: raylinearc.PolicyReceiptDecidedBy{Thinking: "schedule"}, Selection: comparison,
		Excluded: []raylinearc.PolicyReceiptExcluded{{ActionID: strings.Repeat("1", 64), Reason: "capacity_unknown_local"}},
	}
	rendered := raylineRoutesReceiptOf(receipt, strings.Repeat("2", 64), "")
	encoded := string(executionJSON(t, rendered))
	if !strings.Contains(encoded, `"selection":{"rule":"prefer_local"`) || !strings.Contains(encoded, `"reason":"capacity_unknown_local"`) {
		t.Fatalf("receipt %s", encoded)
	}
	if strings.Contains(string(executionJSON(t, raylineRoutesReceiptOf(&raylinearc.PolicyDecisionReceipt{}, strings.Repeat("2", 64), ""))), "selection") {
		t.Fatal("a receipt without a comparison carries selection")
	}
}

// A service refusal of a v2 member reaches the caller with its public code
// through the selector, not the generic undecidable mapping.
func TestRouteExecutionV2ServiceRefusalThroughTheSelector(t *testing.T) {
	for reason, want := range map[string][2]any{
		"policy_preset_not_admitted": {"policy_preset_not_admitted", 400},
		"no_feasible_target":         {"execution_no_feasible_target", 409},
	} {
		fixture := executionFixture(t, false)
		fixture.fake.mu.Lock()
		fixture.fake.failWith = raylinearc.PolicySelectionRefusedClass
		fixture.fake.failDetail = map[string]any{"reason": reason}
		fixture.fake.mu.Unlock()
		_, controls := routesControlsFor(t, fixture, `{}`)
		controls.Policy = &raylinearc.PolicyMember{Preset: "arc-policy.hybrid-default.v1"}
		_, err := selectWithControls(t, fixture, controls)
		var refusal *routeExecutionRefusal
		if !errors.As(selectionFailureError(err), &refusal) || refusal.contract().code != want[0] || refusal.contract().status != want[1] {
			t.Fatalf("%s: err = %v", reason, err)
		}
	}
}
