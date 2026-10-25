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

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The golden vectors are pathfinder#5875's (testdata/arc_public_contract/
// SOURCE.md), produced by arc_execution_context.parse_execution, bind and
// decide_inputs; this adapter must reproduce them.
type executionVectors struct {
	Package struct {
		PackageSHA256          string              `json:"package_sha256"`
		ThinkingControlsSHA256 string              `json:"thinking_controls_sha256"`
		OfferedActionIDs       []string            `json:"offered_action_ids"`
		ActionModels           map[string]string   `json:"action_models"`
		ActionArms             map[string][]string `json:"action_arms"`
	} `json:"package"`
	OfferSHA256 string `json:"offer_sha256"`
	OfferChecks []struct {
		ExpectOfferSHA256  *string  `json:"expect_offer_sha256"`
		AvailableActionIDs []string `json:"available_action_ids"`
		Expect             string   `json:"expect"`
	} `json:"offer_checks"`
	Vectors []struct {
		Name      string          `json:"name"`
		Execution json.RawMessage `json:"execution"`
		Expect    struct {
			Refused string            `json:"refused"`
			Pinned  map[string]string `json:"pinned"`
			Decide  struct {
				ContextEpoch string            `json:"context_epoch"`
				Attribution  []map[string]any  `json:"attribution"`
				Selection    map[string]string `json:"selection"`
				CacheContext json.RawMessage   `json:"cache_context"`
				NotAdmitted  []struct {
					ActionID string `json:"action_id"`
					Field    string `json:"field"`
					Reason   string `json:"reason"`
				} `json:"not_admitted"`
			} `json:"decide"`
		} `json:"expect"`
	} `json:"vectors"`
}

// The body the vectors' attributions index: assistant messages at 1 and 3.
const executionMessages = `[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"},` +
	`{"role":"assistant","content":"d"},{"role":"user","content":"e"}]`

func executionBody(execution string) []byte {
	return []byte(`{"model":"arc-1.0","messages":` + executionMessages + `,"routing":{"execution":` + execution + `}}`)
}

func readExecutionVectors(t *testing.T) executionVectors {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(arcPublicContractDir, "execution_vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var vectors executionVectors
	if err := json.Unmarshal(raw, &vectors); err != nil {
		t.Fatal(err)
	}
	return vectors
}

// vectorScorer is a scorer over the vectors' package: its catalog, models,
// arms and thinking-controls registry.
func vectorScorer(vectors executionVectors) *policyServiceScorer {
	scorer := &policyServiceScorer{
		sha256: vectors.Package.PackageSHA256, thinkingControls: vectors.Package.ThinkingControlsSHA256,
		bindings: map[string]policyBinding{}, actionArms: vectors.Package.ActionArms,
	}
	arm := 0
	for actionID, model := range vectors.Package.ActionModels {
		scorer.bindings[actionID] = policyBinding{arm: arm, model: model}
		arm++
	}
	return scorer
}

func jsonValue(t *testing.T, raw []byte) any {
	t.Helper()
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func executionJSON(t *testing.T, value any) []byte {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

// Every vector, through the adapter's parse, package checks and admission:
// the same refusal, or the same decide members and not-admitted list.
func TestRoutesExecutionReproducesTheGoldenVectors(t *testing.T) {
	vectors := readExecutionVectors(t)
	if got := routesOfferSHA256(vectors.Package.PackageSHA256, vectors.Package.OfferedActionIDs); got != vectors.OfferSHA256 {
		t.Fatalf("offer_sha256 %s, want %s", got, vectors.OfferSHA256)
	}
	scorer := vectorScorer(vectors)
	identity := selection.RaylineARCRouteIdentity{
		PackageSHA256: vectors.Package.PackageSHA256, ThinkingControlsSHA256: vectors.Package.ThinkingControlsSHA256,
		OfferSHA256: vectors.OfferSHA256,
	}
	for _, vector := range vectors.Vectors {
		t.Run(vector.Name, func(t *testing.T) {
			routing, _, refusal := readRaylineRoutesRouting(executionBody(string(vector.Execution)), nil)
			if refusal != nil {
				if refusal.code != vector.Expect.Refused {
					t.Fatalf("parse refused %s, want %q", refusal.code, vector.Expect.Refused)
				}
				return
			}
			controls := raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
			request := &raylinearc.PolicyDecisionRequest{
				ContextEpoch: "0", Selection: raylinearc.PolicySelection{AvailableActionIDs: vectors.Package.OfferedActionIDs},
			}
			class := applyRouteExecution(request, scorer, controls.Execution)
			if class == "" {
				class = checkRouteExecutionOffer(controls.Execution, identity)
			}
			if vector.Expect.Refused != "" {
				if class == "" || routeExecutionClasses[class].code != vector.Expect.Refused {
					t.Fatalf("class %q, want %s", class, vector.Expect.Refused)
				}
				return
			}
			if class != "" {
				t.Fatalf("refused %s", class)
			}
			decide := vector.Expect.Decide
			if request.ContextEpoch != decide.ContextEpoch || request.EpisodeMode != raylinearc.PolicyEpisodeModeRelaxed ||
				!slices.Contains(request.ResponseFields, "cache_trace") {
				t.Fatalf("request %+v", request)
			}
			// arm_id: the reference omits a null one; VSR's form states it
			// null. Both say unknown.
			attribution := []map[string]any{}
			for _, entry := range request.Attribution {
				row := map[string]any{"message": entry.Message, "action_id": entry.ActionID}
				if entry.ArmID != nil {
					row["arm_id"] = *entry.ArmID
				}
				attribution = append(attribution, row)
			}
			if !reflect.DeepEqual(jsonValue(t, executionJSON(t, attribution)), jsonValue(t, executionJSON(t, decide.Attribution))) {
				t.Fatalf("attribution %v, want %v", attribution, decide.Attribution)
			}
			selected := map[string]string{}
			if id := request.Selection.AttestedHeldActionID; id != nil {
				selected["attested_held_action_id"] = *id
			}
			if id := request.Selection.HeldActionID; id != nil {
				selected["held_action_id"] = *id
			}
			if len(selected) > 0 || len(decide.Selection) > 0 {
				if !reflect.DeepEqual(selected, decide.Selection) {
					t.Fatalf("selection %v, want %v", selected, decide.Selection)
				}
			}
			got := jsonValue(t, executionJSON(t, routeExecutionCacheContext(controls.Execution)))
			if want := jsonValue(t, decide.CacheContext); !reflect.DeepEqual(got, want) {
				t.Fatalf("cache_context\n got %s\nwant %s", executionJSON(t, got), decide.CacheContext)
			}
			if len(routing.executionNotAdmitted) != len(decide.NotAdmitted) {
				t.Fatalf("not admitted %+v, want %+v", routing.executionNotAdmitted, decide.NotAdmitted)
			}
			for index, want := range decide.NotAdmitted {
				got := routing.executionNotAdmitted[index]
				if got.ActionID != want.ActionID || got.Field != want.Field || got.Reason != want.Reason {
					t.Fatalf("not admitted %+v, want %+v", routing.executionNotAdmitted, decide.NotAdmitted)
				}
			}
		})
	}
}

// Schema refusals the vectors do not cover, each with its code.
func TestRoutesExecutionRefusals(t *testing.T) {
	vectors := readExecutionVectors(t)
	valid := string(vectors.Vectors[0].Execution)
	replace := func(old, new string) string {
		if !strings.Contains(valid, old) {
			t.Fatalf("vector has no %q", old)
		}
		return strings.Replace(valid, old, new, 1)
	}
	for name, tc := range map[string]struct {
		execution string
		code      string
	}{
		"not an object":         {`[]`, "execution_invalid"},
		"unknown key":           {replace(`"hold": null`, `"hold": null, "extra": 1`), "execution_invalid"},
		"wrong version":         {replace(`arc-execution-context.v1`, `arc-execution-context.v2`), "execution_invalid"},
		"attempt out of bounds": {replace(`"attempt": 1`, `"attempt": 65`), "execution_invalid"},
		"float attempt":         {replace(`"attempt": 1`, `"attempt": 1.5`), "execution_invalid"},
		"id with a slash":       {replace(`"id": "t-0007"`, `"id": "t/0007"`), "execution_invalid"},
		"repeated attribution":  {replace(`"message": 3`, `"message": 1`), "execution_invalid"},
		"gap above a week":      {replace(`"gap_s": 412.5`, `"gap_s": 604801`), "execution_invalid"},
		"overlap above prefix":  {replace(`"overlap_tokens": 18210`, `"overlap_tokens": 21545`), "execution_counts_inconsistent"},
		"attribution beyond":    {replace(`"message": 3`, `"message": 9`), "execution_attribution_invalid"},
	} {
		t.Run(name, func(t *testing.T) {
			_, _, refusal := readRaylineRoutesRouting(executionBody(tc.execution), nil)
			if refusal == nil || refusal.code != tc.code || refusal.status != 400 {
				t.Fatalf("refusal %+v, want %s", refusal, tc.code)
			}
		})
	}
}

// The contract's generation fixture: refused execution_not_supported.
func TestGenerationRefusesExecution(t *testing.T) {
	fixture := readContractFixture(t, "request_invalid_execution_generation.json")
	if _, _, refusal := readRaylineGenerationRouting(fixture.Body, generationPolicy(true, nil)); refusal == nil || refusal.code != fixture.Expect.Code {
		t.Fatalf("refusal %+v, want %s", refusal, fixture.Expect.Code)
	}
}

// Responses bodies index input, and an assistant-output item is a message
// with role assistant, a function_call or a reasoning item.
func TestRoutesExecutionResponsesAttribution(t *testing.T) {
	envelope := map[string]json.RawMessage{"input": json.RawMessage(`[{"role":"user","content":"a"},` +
		`{"type":"reasoning"},{"type":"function_call"},{"type":"function_call_output"},{"type":"message","role":"assistant"}]`)}
	for message, valid := range map[int]bool{0: false, 1: true, 2: true, 3: false, 4: true, 5: false} {
		failure := checkRoutesExecutionAttribution([]routesExecutionAttribution{{message: message}}, envelope)
		if valid != (failure == nil) {
			t.Errorf("message %d: %+v", message, failure)
		}
	}
}

// Through the selector: a lookup with an execution context sends the
// caller's evidence in place of VSR's own and asks for the cache trace; an
// expect.offer_sha256 the answer's offer does not match is refused.
func TestRouteExecutionReachesTheDecideRequest(t *testing.T) {
	fixture := routesFixture(t)
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	think := bindings[0].ActionID
	_, controls := routesControlsFor(t, fixture, `{"value_usd":1.0}`)
	gap, prompt := 12.0, 4000
	controls.Execution = &selection.RaylineARCRouteExecution{
		ContextEpoch: "e7", Attribution: []raylinearc.PolicyAttribution{}, ActionIDs: []string{think},
		CacheContext: &cachecontext.PolicyCacheContext{Actions: []cachecontext.PolicyActionCache{
			{ActionID: think, GapS: &gap, PromptTokens: &prompt},
		}},
	}
	result, err := selectWithControls(t, fixture, controls)
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := fixture.fake.received()[0]
	if sent.ContextEpoch != "e7" || sent.EpisodeMode != raylinearc.PolicyEpisodeModeRelaxed || sent.CacheContext == nil ||
		len(sent.CacheContext.Actions) != 1 || *sent.CacheContext.Actions[0].PromptTokens != prompt ||
		!slices.Contains(sent.ResponseFields, "cache_trace") {
		t.Fatalf("decide request %+v", sent)
	}
	trace := result.RaylineARC
	if trace.RouteIdentity == nil || trace.RouteIdentity.PackageSHA256 != policyTestPackage || !trace.RouteExecutionApplied {
		t.Fatalf("trace identity %+v applied %v", trace.RouteIdentity, trace.RouteExecutionApplied)
	}

	fixture = routesFixture(t)
	_, controls = routesControlsFor(t, fixture, `{"value_usd":1.0}`)
	other := strings.Repeat("c", 64)
	controls.Execution = &selection.RaylineARCRouteExecution{ContextEpoch: "e7", ExpectOffer: &other}
	_, err = selectWithControls(t, fixture, controls)
	var refusal *routeExecutionRefusal
	if !errors.As(selectionFailureError(err), &refusal) || refusal.contract().code != "execution_identity_mismatch" ||
		refusal.contract().status != 409 {
		t.Fatalf("err = %v", err)
	}
}

// The echo mirrors execution_echo: per worker, read and pricing from the
// cache trace's candidate rows, admission's refusals attached; no trace is
// unsupported with no rows. Every result names its identity.
func TestRoutesExecutionEchoAndIdentity(t *testing.T) {
	vectors := readExecutionVectors(t)
	routing, _, refusal := readRaylineRoutesRouting(executionBody(string(vectors.Vectors[1].Execution)), nil)
	if refusal != nil {
		t.Fatal(refusal)
	}
	raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
	qwen, opus := strings.Repeat("1", 64), strings.Repeat("2", 64)
	text := func(value string) *string { return &value }
	warm := 0.845
	trace := &raylinearc.PolicyCacheTrace{
		Boundary: text("user_message"), UserTurnCache: text("repartition_multiwarm"), Source: text("service_config"),
		Candidates: []raylinearc.PolicyCacheTraceCandidate{
			{Model: "qwen/qwen3.8-27b", Offered: true, ActionIDs: []string{qwen}, Basis: text("observed_resident"), PWarm: &warm, EvidenceActionID: &qwen},
			{Model: "anthropic/claude-opus-5.5", Held: true, Offered: true, ActionIDs: []string{opus}, Basis: text("assumed_warm_held")},
		},
	}
	result := func(name string, weight float64) routerruntime.RouteResult {
		return routerruntime.RouteResult{
			Name: name, CostWeight: weight, SelectedWorker: "w", WorkerModel: "m/a",
			PolicyActionID: qwen, ArmID: "arm-1", ControlID: strings.Repeat("d", 64),
		}
	}
	decision := routerruntime.RouteDecision{
		Results:        []routerruntime.RouteResult{result("swift", 10), result("sage", 1/4.5)},
		Identity:       &routerruntime.RouteIdentity{PackageSHA256: vectors.Package.PackageSHA256, OfferSHA256: "o"},
		PolicyActionID: opus, ArmID: "arm-2", ExecutionApplied: true, CacheTrace: trace,
	}
	var payload raylineRoutesResponse
	if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
		t.Fatal(err)
	}
	echo := payload.Routing.Execution
	if echo == nil || echo.Status != "applied" || echo.EpisodeIdentity == nil || *echo.EpisodeIdentity != "per_lookup_random" || echo.Pinned["package_sha256"] != "matched" ||
		echo.Pinned["offer_sha256"] != "unpinned" || echo.Multiwarm == nil || !*echo.Multiwarm ||
		*echo.UserTurnCacheSource != "service_config" || len(echo.Workers) != 2 {
		t.Fatalf("echo %+v", echo)
	}
	local, cloud := echo.Workers[0], echo.Workers[1]
	if !local.Read || local.Pricing != "observed" || *local.PWarm != warm || len(local.NotAdmitted) != 1 ||
		local.NotAdmitted[0].Reason != "residency_requires_local_measured" {
		t.Fatalf("local %+v", local)
	}
	if cloud.Read || *cloud.Reason != "package_mode" || cloud.Pricing != "default_held_warm" || !*cloud.Held || len(cloud.NotAdmitted) != 2 {
		t.Fatalf("cloud %+v", cloud)
	}
	swift := payload.Presets.Swift.Identity
	if swift == nil || *swift.ActionID != qwen || *swift.PolicyActionID != qwen || *swift.ArmID != "arm-1" ||
		swift.ThinkingControlsSHA256 != nil || swift.OfferSHA256 != "o" || swift.Prices.Basis != routesBasisCatalog ||
		swift.Prices.Exclusive != nil {
		t.Fatalf("swift identity %+v", swift)
	}
	if payload.Identity == nil || *payload.Identity.ActionID != opus || payload.Identity.ControlID != nil {
		t.Fatalf("route identity %+v", payload.Identity)
	}
	// No trace: unsupported, nothing applied, no rows, listed unapplied.
	decision.CacheTrace = nil
	payload = raylineRoutesResponse{}
	if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
		t.Fatal(err)
	}
	if echo := payload.Routing.Execution; echo.Status != "unsupported" || echo.Pinned != nil || echo.Multiwarm != nil || echo.EpisodeIdentity != nil ||
		len(echo.Workers) != 0 || !slices.Contains(payload.Routing.UnappliedKeys, "execution") {
		t.Fatalf("unsupported echo %+v %v", echo, payload.Routing.UnappliedKeys)
	}
}

// Each package-side refusal reaches the caller with its code and status;
// the cell's action_not_admitted without a hold stays unavailable (503).
func TestRouteExecutionRefusalContracts(t *testing.T) {
	for class, want := range map[string][2]any{
		arcFailureExecutionUnknownAction:    {"execution_unknown_action", 400},
		arcFailureExecutionIdentityControls: {"execution_identity_mismatch", 409},
		arcFailureExecutionHoldConflict:     {"execution_hold_conflict", 409},
		arcFailureExecutionHoldUnavailable:  {"execution_hold_unavailable", 409},
	} {
		var refusal *routeExecutionRefusal
		if !errors.As(selectionFailureError(arcSelectionFailure(class)), &refusal) {
			t.Fatalf("%s: not a refusal", class)
		}
		contract := refusal.contract()
		if contract.code != want[0] || contract.status != want[1] || contract.errType != "invalid_request_error" {
			t.Fatalf("%s: %+v", class, contract)
		}
	}
	notAdmitted := &raylinearc.PolicyServiceError{Class: "selection_refused", Detail: map[string]any{"reason": "action_not_admitted"}}
	if routeExecutionServiceRefusal(notAdmitted) != arcFailureExecutionNotAdmitted {
		t.Fatal("action_not_admitted not mapped")
	}
	var refusal *routeExecutionRefusal
	if errors.As(selectionFailureError(arcSelectionFailure(arcFailureExecutionNotAdmitted)), &refusal) {
		t.Fatal("an unheld action_not_admitted must stay unavailable (503)")
	}
	outside := &raylinearc.PolicyServiceError{Class: "selection_refused", Detail: map[string]any{"reason": "attribution_out_of_catalog"}}
	if routeExecutionServiceRefusal(outside) != arcFailureExecutionUnknownAction {
		t.Fatal("attribution_out_of_catalog not mapped")
	}
}

// check_offer's vectors: the offer pin against the effective offer (the
// decide response's available actions).
func TestRoutesExecutionOfferChecks(t *testing.T) {
	vectors := readExecutionVectors(t)
	if len(vectors.OfferChecks) == 0 {
		t.Fatal("no offer checks")
	}
	scorer := vectorScorer(vectors)
	for index, check := range vectors.OfferChecks {
		response := &raylinearc.PolicyDecisionResponse{}
		for _, id := range check.AvailableActionIDs {
			response.Actions = append(response.Actions, raylinearc.PolicyActionScore{ActionID: id, Available: true})
		}
		response.Actions = append(response.Actions, raylinearc.PolicyActionScore{ActionID: strings.Repeat("3", 64)})
		class := checkRouteExecutionOffer(&selection.RaylineARCRouteExecution{ExpectOffer: check.ExpectOfferSHA256}, routeIdentity(scorer, response))
		got := "unpinned"
		switch {
		case class != "":
			got = routeExecutionClasses[class].code
		case check.ExpectOfferSHA256 != nil:
			got = "matched"
		}
		if got != check.Expect {
			t.Errorf("check %d: %s, want %s", index, got, check.Expect)
		}
	}
}

// A candidate the request did not offer (and not held) prices nothing.
func TestRoutesExecutionEchoUnofferedCandidate(t *testing.T) {
	vectors := readExecutionVectors(t)
	routing, _, refusal := readRaylineRoutesRouting(executionBody(string(vectors.Vectors[0].Execution)), nil)
	if refusal != nil {
		t.Fatal(refusal)
	}
	raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
	qwen := strings.Repeat("1", 64)
	boundary, basis := "user_message", "observed_resident"
	decision := routerruntime.RouteDecision{ExecutionApplied: true, CacheTrace: &raylinearc.PolicyCacheTrace{
		Boundary: &boundary, Candidates: []raylinearc.PolicyCacheTraceCandidate{
			{Model: "qwen/qwen3.8-27b", ActionIDs: []string{qwen}, Basis: &basis, EvidenceActionID: &qwen},
		},
	}}
	echo := raylineRoutesExecutionEchoOf(routing, decision)
	if row := echo.Workers[0]; row.Read || row.Pricing != "not_priced" || *row.Reason != "not_in_offer" || row.Held != nil {
		t.Fatalf("unoffered row %+v", row)
	}
}
