package extproc

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The spec's example (pathfinder #5874), with the fixture's actions: a held
// cloud model with wire-declared TTL and a resident local rival.
func executionExample(qwen, opus string) string {
	return `{"version": "arc-execution-context.v1", "context_epoch": "e3",
	 "turn": {"id": "t-0007", "attempt": 1},
	 "expect": {"package_sha256": null, "thinking_controls_sha256": null, "offer_sha256": null},
	 "attribution": [
	  {"message": 1, "action_id": "` + qwen + `", "arm_id": null, "worker": "local-qwen", "route_id": "rt_1", "result": "custom"},
	  {"message": 3, "action_id": "` + opus + `", "arm_id": null, "worker": "cloud-opus", "route_id": "rt_2", "result": "custom"}],
	 "hold": null,
	 "workers": [
	  {"action_id": "` + qwen + `", "worker": "local-qwen", "placement": "local", "context_epoch": "e3", "gap_s": 412.0,
	   "ttl": null, "residency": {"state": "resident", "provenance": "local_measured"},
	   "usage": {"prompt_tokens": 18210, "cache_read_tokens": 17900, "cache_write_tokens": null, "units": "hf:Qwen/Qwen3.8-27B@r1", "provenance": "provider_reported"},
	   "current_prefix": {"tokens": 21544, "overlap_tokens": 18210, "units": "hf:Qwen/Qwen3.8-27B@r1", "provenance": "local_measured"}},
	  {"action_id": "` + opus + `", "worker": "cloud-opus", "placement": "cloud", "context_epoch": "e3", "gap_s": 35.2,
	   "ttl": {"seconds": 300, "source": "anthropic cache_control ephemeral 5m", "basis": "wire_declared"}, "residency": null,
	   "usage": {"prompt_tokens": 20977, "cache_read_tokens": 18100, "cache_write_tokens": 2877, "units": "provider:cloud-opus", "provenance": "provider_reported"},
	   "current_prefix": {"tokens": 21390, "overlap_tokens": null, "units": "provider:cloud-opus", "provenance": "provider_reported"}}]}`
}

const executionMessages = `[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"},` +
	`{"role":"assistant","content":"d"},{"role":"user","content":"e"}]`

func executionBody(execution string) []byte {
	return []byte(`{"model":"arc-1.0","messages":` + executionMessages + `,"routing":{"value_usd":1.0,"execution":` + execution + `}}`)
}

var (
	executionQwen = strings.Repeat("a", 64)
	executionOpus = strings.Repeat("b", 64)
)

func TestRoutesExecutionReadsTheSpecExample(t *testing.T) {
	routing, stripped, refusal := readRaylineRoutesRouting(executionBody(executionExample(executionQwen, executionOpus)), nil)
	if refusal != nil {
		t.Fatalf("refused: %+v", refusal)
	}
	if strings.Contains(string(stripped), "execution") {
		t.Fatal("routing.execution left in the body the codec reads")
	}
	controls := raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
	execution := controls.Execution
	if execution == nil || execution.ContextEpoch != "e3" || len(execution.Attribution) != 2 ||
		execution.Attribution[1].Message != 3 || execution.Attribution[1].ActionID != executionOpus ||
		execution.UserTurnCache != routesUserTurnCacheMultiwarm || execution.CacheContext == nil ||
		len(execution.CacheContext.Actions) != 2 {
		t.Fatalf("controls %+v", execution)
	}
	qwen, opus := execution.CacheContext.Actions[0], execution.CacheContext.Actions[1]
	// The local worker: residency (local, measured) and both counts admitted;
	// no TTL to admit.
	if qwen.Residency == nil || *qwen.Residency != "resident" || qwen.TTLS != nil || *qwen.PromptTokens != 18210 ||
		*qwen.CurrentPrefixTokens != 21544 || *qwen.OverlapTokens != 18210 || *qwen.GapS != 412 || *qwen.ContextEpoch != "e3" {
		t.Fatalf("qwen %+v", qwen)
	}
	// The cloud worker: the wire-declared TTL and provider counts admitted;
	// an unknown overlap stays null.
	if opus.TTLS == nil || *opus.TTLS != 300 || *opus.TTLSource != "anthropic cache_control ephemeral 5m" ||
		opus.Residency != nil || *opus.CacheWriteTokens != 2877 || *opus.CurrentPrefixTokens != 21390 || opus.OverlapTokens != nil {
		t.Fatalf("opus %+v", opus)
	}
	encoded, _ := json.Marshal(execution.CacheContext.Actions[1])
	if !strings.Contains(string(encoded), `"current_prefix_tokens":21390`) || strings.Contains(string(encoded), "overlap_tokens") {
		t.Fatalf("wire %s", encoded)
	}
}

// Each refusal the parse owns, with its code, in the contract's order.
func TestRoutesExecutionRefusals(t *testing.T) {
	valid := executionExample(executionQwen, executionOpus)
	replace := func(old, new string) string {
		if !strings.Contains(valid, old) {
			t.Fatalf("fixture has no %q", old)
		}
		return strings.Replace(valid, old, new, 1)
	}
	for name, tc := range map[string]struct {
		execution string
		code      string
	}{
		"not an object":           {`[]`, "execution_invalid"},
		"unknown key":             {replace(`"hold": null,`, `"hold": null, "extra": 1,`), "execution_invalid"},
		"missing key":             {replace(`"hold": null,`, ``), "execution_invalid"},
		"wrong version":           {replace(`arc-execution-context.v1`, `arc-execution-context.v2`), "execution_invalid"},
		"attempt out of bounds":   {replace(`"attempt": 1`, `"attempt": 65`), "execution_invalid"},
		"float attempt":           {replace(`"attempt": 1`, `"attempt": 1.5`), "execution_invalid"},
		"id with a slash":         {replace(`"id": "t-0007"`, `"id": "t/0007"`), "execution_invalid"},
		"bad action id":           {replace(`"message": 1, "action_id": "`+executionQwen, `"message": 1, "action_id": "XY`), "execution_invalid"},
		"repeated attribution":    {replace(`"message": 3`, `"message": 1`), "execution_invalid"},
		"repeated worker action":  {strings.Replace(valid, `"action_id": "`+executionOpus+`", "worker": "cloud-opus", "placement"`, `"action_id": "`+executionQwen+`", "worker": "cloud-opus", "placement"`, 1), "execution_invalid"},
		"gap above a week":        {replace(`"gap_s": 412.0`, `"gap_s": 604801`), "execution_invalid"},
		"zero ttl":                {replace(`"seconds": 300`, `"seconds": 0`), "execution_invalid"},
		"bad provenance":          {replace(`"state": "resident", "provenance": "local_measured"`, `"state": "resident", "provenance": "guessed"`), "execution_invalid"},
		"cache leg above prompt":  {replace(`"cache_read_tokens": 17900`, `"cache_read_tokens": 19000`), "execution_counts_inconsistent"},
		"legs together above":     {replace(`"cache_read_tokens": 18100`, `"cache_read_tokens": 18101`), "execution_counts_inconsistent"},
		"overlap above prefix":    {replace(`"overlap_tokens": 18210`, `"overlap_tokens": 21545`), "execution_counts_inconsistent"},
		"attribution beyond body": {replace(`"message": 3`, `"message": 9`), "execution_attribution_invalid"},
		"attribution of a user":   {replace(`"message": 3`, `"message": 2`), "execution_attribution_invalid"},
	} {
		t.Run(name, func(t *testing.T) {
			_, _, refusal := readRaylineRoutesRouting(executionBody(tc.execution), nil)
			if refusal == nil || refusal.code != tc.code || refusal.status != 400 {
				t.Fatalf("refusal %+v, want %s", refusal, tc.code)
			}
		})
	}
}

// routing.execution on a generation request is refused: a turn this router
// executes keeps its own execution state.
func TestGenerationRefusesExecution(t *testing.T) {
	body := []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"x"}],"routing":{"execution":{}}}`)
	if _, _, refusal := readRaylineGenerationRouting(body, generationPolicy(true, nil)); refusal == nil || refusal.code != "execution_not_supported" {
		t.Fatalf("refusal %+v", refusal)
	}
}

// Admission: estimated evidence and unmatched units are passed on as
// unknown and named; measured residency counts only for a local worker.
func TestRoutesExecutionAdmission(t *testing.T) {
	prompt, tokens := 100, 120
	estimatedTTL := routesExecutionWorker{
		actionID: executionOpus, placement: "cloud",
		ttl:       &routesExecutionTTL{seconds: 300, source: "guess", basis: "caller_estimated"},
		residency: &routesExecutionResidency{state: "resident", provenance: provenanceLocalMeasured},
		usage:     &routesExecutionUsage{promptTokens: &prompt, units: "provider:a", provenance: provenanceProviderReported},
		currentPrefix: &routesExecutionPrefix{
			tokens: tokens, units: "hf:other", provenance: provenanceLocalMeasured,
		},
	}
	entry, refused, admitted := admitRoutesExecutionWorker(estimatedTTL)
	if entry.TTLS != nil || entry.Residency != nil || entry.CurrentPrefixTokens != nil || *entry.PromptTokens != prompt || !admitted {
		t.Fatalf("entry %+v admitted %v", entry, admitted)
	}
	want := []routesExecutionNotAdmitted{
		{"ttl", "estimated_not_admitted"}, {"residency", "local_measured_not_local"}, {"current_prefix", "units_mismatch"},
	}
	if len(refused) != len(want) {
		t.Fatalf("not admitted %+v", refused)
	}
	for index := range want {
		if refused[index] != want[index] {
			t.Fatalf("not admitted %+v, want %+v", refused, want)
		}
	}
	// Nothing admissible: no repartition asked for.
	estimated := routesExecutionWorker{
		actionID: executionOpus, placement: "cloud",
		usage: &routesExecutionUsage{promptTokens: &prompt, units: "u", provenance: provenanceCallerEstimated},
	}
	if _, refused, admitted := admitRoutesExecutionWorker(estimated); admitted || refused[0].Field != "usage" {
		t.Fatalf("estimated usage admitted=%v %+v", admitted, refused)
	}
}

func TestRoutesOfferSHA256(t *testing.T) {
	if got := routesOfferSHA256("p", []string{"b", "a"}); got != "eff332dd684e4d69ce14cf733334390659851bf810ebaa28fa206bb6213ee9b3" {
		t.Fatalf("offer sha %s", got)
	}
}

// The package-side checks, in order: catalog, expect, hold; and the
// applied context: the caller's epoch and attribution, relaxed, the hold.
func TestApplyRouteExecution(t *testing.T) {
	fixture := routesFixture(t)
	scorer := newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	think, thinkUp, off := bindings[0].ActionID, bindings[1].ActionID, bindings[2].ActionID
	request := func(available ...string) *raylinearc.PolicyDecisionRequest {
		return &raylinearc.PolicyDecisionRequest{
			ContextEpoch: "0", EpisodeMode: "strict",
			Selection: raylinearc.PolicySelection{AvailableActionIDs: available},
		}
	}
	identity := routeIdentity(scorer, []string{think, thinkUp, off})
	attributed := []raylinearc.PolicyAttribution{{Message: 1, ActionID: think}}
	other := strings.Repeat("c", 64)
	for name, tc := range map[string]struct {
		execution selection.RaylineARCRouteExecution
		available []string
		want      string
	}{
		"unknown action":       {selection.RaylineARCRouteExecution{ActionIDs: []string{strings.Repeat("e", 64)}}, nil, arcFailureExecutionUnknownAction},
		"package mismatch":     {selection.RaylineARCRouteExecution{ExpectPackage: &other}, nil, arcFailureExecutionIdentityPackage},
		"offer mismatch":       {selection.RaylineARCRouteExecution{ExpectOffer: &other}, nil, arcFailureExecutionIdentityOffer},
		"hold model conflicts": {selection.RaylineARCRouteExecution{Attribution: attributed, HoldScope: "model", HoldActionID: off}, nil, arcFailureExecutionHoldConflict},
		"held action excluded": {selection.RaylineARCRouteExecution{HoldScope: "action", HoldActionID: off}, []string{think}, arcFailureExecutionHoldUnavailable},
	} {
		t.Run(name, func(t *testing.T) {
			if got := applyRouteExecution(request(tc.available...), scorer, &tc.execution, identity); got != tc.want {
				t.Fatalf("class %q, want %q", got, tc.want)
			}
		})
	}
	matched := identity.OfferSHA256
	applied := request(think, thinkUp, off)
	execution := selection.RaylineARCRouteExecution{
		ContextEpoch: "e3", Attribution: attributed, HoldScope: "model", HoldActionID: thinkUp,
		UserTurnCache: routesUserTurnCacheMultiwarm, ExpectOffer: &matched, ActionIDs: []string{think, thinkUp},
	}
	if class := applyRouteExecution(applied, scorer, &execution, identity); class != "" {
		t.Fatalf("refused %s", class)
	}
	if applied.ContextEpoch != "e3" || len(applied.Attribution) != 1 || applied.EpisodeMode != raylinearc.PolicyEpisodeModeRelaxed ||
		applied.Selection.AttestedHeldActionID == nil || *applied.Selection.AttestedHeldActionID != thinkUp ||
		applied.Selection.HeldActionID != nil || applied.Selection.UserTurnCache != routesUserTurnCacheMultiwarm {
		t.Fatalf("applied %+v", applied)
	}
}

// Through the selector: a lookup with an execution context sends the
// caller's cache evidence in place of VSR's own, the cell's allow-list, and
// records the identity it decided under.
func TestRouteExecutionReachesTheDecideRequest(t *testing.T) {
	fixture := routesFixture(t)
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	think := bindings[0].ActionID
	_, controls := routesControlsFor(t, fixture, `{"value_usd":1.0}`)
	gap, prompt := 12.0, 4000
	controls.Execution = &selection.RaylineARCRouteExecution{
		ContextEpoch: "e7", Attribution: []raylinearc.PolicyAttribution{}, ActionIDs: []string{think},
		CacheContext: &raylinearc.PolicyCacheContext{Actions: []raylinearc.PolicyActionCache{
			{ActionID: think, GapS: &gap, PromptTokens: &prompt},
		}},
		UserTurnCache: routesUserTurnCacheMultiwarm,
	}
	result, err := selectWithControls(t, fixture, controls)
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := fixture.fake.received()[0]
	if sent.ContextEpoch != "e7" || sent.EpisodeMode != raylinearc.PolicyEpisodeModeRelaxed || sent.CacheContext == nil ||
		len(sent.CacheContext.Actions) != 1 || *sent.CacheContext.Actions[0].PromptTokens != prompt ||
		sent.Selection.UserTurnCache != routesUserTurnCacheMultiwarm {
		t.Fatalf("decide request %+v", sent)
	}
	trace := result.RaylineARC
	if trace.RouteIdentity == nil || trace.RouteIdentity.PackageSHA256 != policyTestPackage || !trace.RouteExecutionApplied ||
		trace.RouteIdentity.OfferSHA256 != routesOfferSHA256(policyTestPackage, sent.Selection.AvailableActionIDs) {
		t.Fatalf("trace identity %+v applied %v", trace.RouteIdentity, trace.RouteExecutionApplied)
	}
}

// The allow-list is the operator's: sent on every decide, and the offer
// identity is the narrowed offer.
func TestOfferedActionIDsNarrowTheOffer(t *testing.T) {
	fixture := routesFixture(t)
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.decision.Algorithm.RaylineARC.PolicyService.OfferedActionIDs = []string{bindings[2].ActionID}
	scorer := newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
	all := []string{bindings[0].ActionID, bindings[1].ActionID, bindings[2].ActionID}
	if offer := scorer.offerOf(all); len(offer) != 1 || offer[0] != bindings[2].ActionID {
		t.Fatalf("offer %v", offer)
	}
	if routeIdentity(scorer, all).OfferSHA256 != routesOfferSHA256(policyTestPackage, []string{bindings[2].ActionID}) {
		t.Fatal("the offer identity is not the narrowed offer")
	}
	armed := fixture.selector.armed.Load()
	armed.scorer = scorer
	fixture.selector.arm(armed)
	_, controls := routesControlsFor(t, fixture, `{}`)
	if _, err := selectWithControls(t, fixture, controls); err != nil {
		t.Fatalf("select: %v", err)
	}
	if sent := fixture.fake.received()[0].Selection.OfferedActionIDs; len(sent) != 1 || sent[0] != bindings[2].ActionID {
		t.Fatalf("offered_action_ids %v", sent)
	}
}

// The echo and identity: pinned members report matched, unpinned stay
// unpinned, worker evidence is copied from the service's cache_trace with
// admission's reasons; every result names its identity.
func TestRoutesExecutionEchoAndIdentity(t *testing.T) {
	routing, _, refusal := readRaylineRoutesRouting(executionBody(strings.Replace(
		executionExample(executionQwen, executionOpus), `"package_sha256": null`, `"package_sha256": "`+policyTestPackage+`"`, 1)), nil)
	if refusal != nil {
		t.Fatal(refusal)
	}
	raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
	read, warm, boundary, mode := true, 0.845, "user_message", routesUserTurnCacheMultiwarm
	result := func(name string, weight float64) routerruntime.RouteResult {
		return routerruntime.RouteResult{
			Name: name, CostWeight: weight, SelectedWorker: "w", WorkerModel: "m/a",
			PolicyActionID: executionQwen, ArmID: "arm-1", ControlID: strings.Repeat("d", 64),
		}
	}
	decision := routerruntime.RouteDecision{
		Results:        []routerruntime.RouteResult{result("swift", 10), result("sage", 1/4.5), result("custom", 1)},
		Identity:       &routerruntime.RouteIdentity{PackageSHA256: policyTestPackage, OfferSHA256: "o"},
		PolicyActionID: executionOpus, ArmID: "arm-2", ExecutionApplied: true,
		CacheTrace: &raylinearc.PolicyCacheTrace{Boundary: &boundary, UserTurnCache: &mode, Actions: []raylinearc.PolicyCacheTraceAction{
			{ActionID: executionQwen, Read: &read, PWarm: &warm},
		}},
	}
	var payload raylineRoutesResponse
	if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
		t.Fatal(err)
	}
	echo := payload.Routing.Execution
	if echo == nil || echo.Status != "applied" || echo.Turn.ID != "t-0007" || echo.Pinned["package_sha256"] != "matched" ||
		echo.Pinned["offer_sha256"] != "unpinned" || *echo.Boundary != "user_message" || len(echo.Workers) != 2 ||
		*echo.Workers[0].PWarm != warm || echo.Workers[1].Read != nil {
		t.Fatalf("echo %+v", echo)
	}
	swift := payload.Presets.Swift.Identity
	if swift == nil || *swift.ActionID != executionQwen || *swift.PolicyActionID != executionQwen || *swift.ArmID != "arm-1" ||
		swift.ThinkingControlsSHA256 != nil || swift.OfferSHA256 != "o" || swift.Prices.Basis != routesBasisCatalog {
		t.Fatalf("swift identity %+v", swift)
	}
	if payload.Identity == nil || *payload.Identity.ActionID != executionOpus || payload.Identity.ControlID != nil {
		t.Fatalf("route identity %+v", payload.Identity)
	}
	// Not applied: the echo says unsupported and lists it.
	decision.ExecutionApplied = false
	payload = raylineRoutesResponse{}
	if err := raylineRoutesAddResults(&payload, routing, decision); err != nil {
		t.Fatal(err)
	}
	if payload.Routing.Execution.Status != "unsupported" || !strings.Contains(strings.Join(payload.Routing.UnappliedKeys, ","), "execution") {
		t.Fatalf("unsupported echo %+v", payload.Routing)
	}
}

// Each package-side refusal reaches the caller with its code and status.
func TestRouteExecutionRefusalContracts(t *testing.T) {
	for class, want := range map[string][2]any{
		arcFailureExecutionUnknownAction:    {"execution_unknown_action", 400},
		arcFailureExecutionIdentityControls: {"execution_identity_mismatch", 409},
		arcFailureExecutionHoldConflict:     {"execution_hold_conflict", 409},
		arcFailureExecutionHoldUnavailable:  {"execution_hold_unavailable", 409},
	} {
		err := selectionFailureError(arcSelectionFailure(class))
		var refusal *routeExecutionRefusal
		if !errors.As(err, &refusal) {
			t.Fatalf("%s: %v", class, err)
		}
		contract := refusal.contract()
		if contract.code != want[0] || contract.status != want[1] || contract.errType != "invalid_request_error" {
			t.Fatalf("%s: %+v", class, contract)
		}
	}
	if !strings.Contains(routeExecutionClasses[arcFailureExecutionIdentityControls].member, "thinking_controls") {
		t.Fatal("identity member")
	}
	refused := &raylinearc.PolicyServiceError{Class: "selection_refused", Detail: map[string]any{"reason": "attribution_out_of_catalog"}}
	if routeExecutionServiceRefusal(refused) != arcFailureExecutionUnknownAction {
		t.Fatal("attribution_out_of_catalog not mapped")
	}
}
