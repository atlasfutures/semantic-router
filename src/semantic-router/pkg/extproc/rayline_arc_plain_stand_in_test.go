//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The plain control: the default base with no instruction.
var plainControlID = thinkingcontrol.Control{Native: "default"}.ID()

// v5GLMPlain is the fixture's GLM at its default base with no instruction:
// an action the fixture does not bind (it binds GLM under "none" and "up").
const v5GLMPlain = "bd202891c3ed5bf9af5250868da17cb3ad68a05184ec5fa757876880257a5b26"

// The stand-ins come from the package: a model bound under "none" and not
// under its plain control gets one (GLM); a model bound under its plain
// control does not (Opus, native only); and once the plain action is bound,
// none is left.
func TestPlainStandInsComeFromThePackage(t *testing.T) {
	if got := v5ActionID(plainControlID, "glm-5.3-flash"); got != v5GLMPlain {
		t.Fatalf("plain GLM action id %s", got)
	}
	if got := v5ActionID(plainControlID, "claude-opus-5"); got != v5OpusAction {
		t.Fatalf("plain Opus action id %s, want the fixture's native Opus", got)
	}
	cfg := v5ConfigWith(t, func(rendered string) string { return rendered })
	policy := cfg.Decisions[0].Algorithm.RaylineARC.PolicyService
	want := map[string]plainStandIn{v5GLMPlain: {standInActionID: v5GLMNone, plainControlID: plainControlID, model: "glm-5.3-flash"}}
	if got := plainStandInsOf(policy); !reflect.DeepEqual(got, want) {
		t.Fatalf("stand-ins %+v", got)
	}
	withPlain := *policy
	withPlain.Bindings = append(append([]config.RaylineARCPolicyBinding{}, policy.Bindings...), config.RaylineARCPolicyBinding{ActionID: v5GLMPlain, Worker: "glm"})
	if got := plainStandInsOf(&withPlain); got != nil {
		t.Fatalf("a package that binds the plain action still stands in: %+v", got)
	}
	if plainStandInsOf(nil) != nil {
		t.Fatal("no policy service, no stand-ins")
	}
}

func TestPlainStandInsInTheEnvelope(t *testing.T) {
	local, opus, sonnet, haikuNone, haikuUp := strings.Repeat("1", 64), strings.Repeat("2", 64), strings.Repeat("3", 64),
		strings.Repeat("4", 64), strings.Repeat("6", 64)
	haikuPlain, other := strings.Repeat("5", 64), strings.Repeat("7", 64)
	scorer := &policyServiceScorer{
		bindings: map[string]policyBinding{
			local: {model: "qwen3.8-27b"}, opus: {model: "claude-opus-5.5"}, sonnet: {model: "claude-sonnet-5.5"},
			haikuNone: {model: "claude-haiku-5.5", level: "none"}, haikuUp: {model: "claude-haiku-5.5", level: "up"},
		},
	}
	standIns := map[string]plainStandIn{haikuPlain: {standInActionID: haikuNone, plainControlID: plainControlID, model: "claude-haiku-5.5"}}
	target := func(actionID, placement, capacity string) string {
		return `{"action_id":"` + actionID + `","worker":"w","placement":"` + placement + `","cache":null,"capacity":` + capacity + `}`
	}
	cloud := func(actionID string) string { return target(actionID, "cloud", "null") }
	overWindow := `{"fits":false,"reason":"context_capacity","required":42733,"available":32768,"available_source":"configured","units":null,"provenance":"caller_bounded"}`
	failed := `{"fits":false,"reason":"runtime_unavailable","required":null,"available":null,"available_source":null,"units":null,"provenance":"provider_reported"}`
	retryOf := func(reconsult string) string {
		if reconsult == "null" {
			return "null"
		}
		return `"rt_0123456789abcdef0123456789abcdef"`
	}
	envelope := func(history, lastServed, reconsult string, targets ...string) string {
		return `{"version":"arc-execution-context.v2","turn":{"id":"sha256:` + strings.Repeat("a", 64) + `","attempt":1,"epoch":"e1","retry_of":` + retryOf(reconsult) + `,"reconsult":` + reconsult + `},` +
			`"event":{"kind":"task_start","basis":"marker_inferred","compaction":null},"last_served":` + lastServed + `,"history":` + history + `,"hold":null,"expect":null,` +
			`"targets":[` + strings.Join(targets, ",") + `]}`
	}
	apply := func(t *testing.T, body string) ([]raylinearc.PolicyReceiptExcluded, *raylinearc.PolicyDecisionRequest, *selection.RaylineARCRouteExecutionV2) {
		t.Helper()
		routing, _, refusal := readRaylineRoutesRouting(executionV2Body(`{"execution":`+body+`}`), nil, config.DefaultRaylineARCValueUSDMax)
		if refusal != nil {
			t.Fatalf("parse refused %s (%s)", refusal.code, refusal.message)
		}
		controls := raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
		request := &raylinearc.PolicyDecisionRequest{Selection: raylinearc.PolicySelection{AvailableActionIDs: []string{local, opus, sonnet, haikuNone, haikuUp}}}
		applyPlainStandIns(controls.ExecutionV2, standIns, "episode")
		unknown, class := applyRoutePolicyAndExecutionV2(request, scorer, controls)
		if class != "" {
			t.Fatalf("refused %s", class)
		}
		return unknown, request, controls.ExecutionV2
	}
	capacityIDs := func(request *raylinearc.PolicyDecisionRequest) []string {
		ids := []string{}
		for _, entry := range request.Capacity {
			ids = append(ids, entry.ActionID)
		}
		return ids
	}
	stoodIn := map[string]selection.RaylineARCPlainStandIn{haikuNone: {PlainActionID: haikuPlain, PlainControlID: plainControlID}}

	// The Chat and Responses shapes: the executor offers Haiku only under
	// its plain control. It is scored as Haiku "none" instead of being left
	// out; with the local model over its window, failed, or absent.
	for name, body := range map[string]string{
		"chat over the window":      envelope(`[]`, "null", "null", target(local, "local", overWindow), cloud(haikuPlain), cloud(sonnet), cloud(opus)),
		"responses without a local": envelope(`[]`, "null", "null", cloud(haikuPlain), cloud(sonnet), cloud(opus)),
		"chat after a local failure": envelope(`[]`, "null", `"capacity"`,
			target(local, "local", failed), cloud(haikuPlain), cloud(sonnet), cloud(opus)),
	} {
		t.Run(name, func(t *testing.T) {
			unknown, request, v2 := apply(t, body)
			if len(unknown) != 0 || !reflect.DeepEqual(v2.StandIns, stoodIn) {
				t.Fatalf("unknown %v, stand-ins %+v", unknown, v2.StandIns)
			}
			ids := capacityIDs(request)
			if !strings.Contains(strings.Join(ids, ","), haikuNone) || strings.Contains(strings.Join(ids, ","), haikuPlain) {
				t.Fatalf("decide capacity %v", ids)
			}
		})
	}

	// A later turn of such a conversation: its history and last_served name
	// the plain action that ran, and are read as the stand-in.
	t.Run("second turn", func(t *testing.T) {
		_, request, v2 := apply(t, envelope(`[{"message":1,"action_id":"`+haikuPlain+`"}]`, `{"action_id":"`+haikuPlain+`","worker":"w","message":1}`, "null",
			cloud(haikuPlain), cloud(sonnet)))
		if len(request.Attribution) != 1 || request.Attribution[0].ActionID != haikuNone ||
			request.Selection.LastServedActionID == nil || *request.Selection.LastServedActionID != haikuNone ||
			!reflect.DeepEqual(v2.HeldStandIns, map[string]string{haikuNone: haikuPlain}) {
			t.Fatalf("attribution %+v, last served %v", request.Attribution, request.Selection.LastServedActionID)
		}
	})

	// A later turn that no longer offers Haiku at all (its backend excluded,
	// say) still names the plain action that ran: read as the stand-in, not
	// refused, and nothing is answered under it.
	t.Run("history without the target", func(t *testing.T) {
		_, request, v2 := apply(t, envelope(`[{"message":1,"action_id":"`+haikuPlain+`"}]`, `{"action_id":"`+haikuPlain+`","worker":"w","message":1}`, "null",
			cloud(sonnet), cloud(opus)))
		if v2.StandIns != nil || request.Attribution[0].ActionID != haikuNone || *request.Selection.LastServedActionID != haikuNone {
			t.Fatalf("stand-ins %+v, attribution %+v", v2.StandIns, request.Attribution)
		}
	})

	// A hold follows its target. Held on a plain target that is stood in
	// for, it holds the stand-in; held on a plain target that is not (the
	// Messages shape, here also marked not to fit), it keeps its own id and
	// is refused as before.
	hold := func(body, actionID string) string {
		return strings.Replace(body, `"hold":null`, `"hold":{"scope":"action","action_id":"`+actionID+`"}`, 1)
	}
	t.Run("hold on a stood-in target", func(t *testing.T) {
		_, request, v2 := apply(t, hold(envelope(`[]`, "null", "null", cloud(haikuPlain), cloud(sonnet)), haikuPlain))
		if request.Selection.HeldActionID == nil || *request.Selection.HeldActionID != haikuNone ||
			!reflect.DeepEqual(v2.HeldStandIns, map[string]string{haikuNone: haikuPlain}) {
			t.Fatalf("held %v, held stand-ins %+v", request.Selection.HeldActionID, v2.HeldStandIns)
		}
	})
	// An envelope the stand-in would make pass a check it fails on its own
	// ids is left as it was, with the refusal it had before.
	raw := func(t *testing.T, body string) (*selection.RaylineARCRouteExecutionV2, string) {
		t.Helper()
		routing, _, refusal := readRaylineRoutesRouting(executionV2Body(`{"execution":`+body+`}`), nil, config.DefaultRaylineARCValueUSDMax)
		if refusal != nil {
			t.Fatalf("parse refused %s (%s)", refusal.code, refusal.message)
		}
		controls := raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
		applyPlainStandIns(controls.ExecutionV2, standIns, "episode")
		request := &raylinearc.PolicyDecisionRequest{Selection: raylinearc.PolicySelection{AvailableActionIDs: []string{haikuNone, haikuUp, opus, sonnet}}}
		_, class := applyRoutePolicyAndExecutionV2(request, scorer, controls)
		return controls.ExecutionV2, class
	}
	t.Run("hold on a target not stood in for", func(t *testing.T) {
		v2, class := raw(t, hold(envelope(`[]`, "null", "null", target(haikuPlain, "cloud", overWindow), cloud(haikuNone), cloud(opus)), haikuPlain))
		if *v2.HoldActionID != haikuPlain || v2.HeldStandIns != nil || class != arcFailureExecutionUnknownAction {
			t.Fatalf("hold %s, held stand-ins %+v, class %q", *v2.HoldActionID, v2.HeldStandIns, class)
		}
	})
	t.Run("hold on the stand-in itself", func(t *testing.T) {
		// Only the plain target is offered, but the caller holds the none
		// action: no target stands in, so the hold cannot be met through the
		// plain action, and the plain target is left out as before.
		v2, _ := raw(t, hold(envelope(`[]`, "null", "null", cloud(haikuPlain), cloud(sonnet)), haikuNone))
		if v2.StandIns != nil || *v2.HoldActionID != haikuNone || slices.Contains(v2.TargetActionIDs, haikuNone) {
			t.Fatalf("stand-ins %+v, hold %s, targets %v", v2.StandIns, *v2.HoldActionID, v2.TargetActionIDs)
		}
	})
	t.Run("history and last_served disagree", func(t *testing.T) {
		v2, class := raw(t, envelope(`[{"message":1,"action_id":"`+haikuPlain+`"}]`, `{"action_id":"`+haikuNone+`","worker":"w","message":1}`, "null",
			cloud(haikuPlain), cloud(sonnet)))
		if v2.StandIns != nil || v2.Attribution[0].ActionID != haikuPlain || class != arcFailureExecutionUnknownAction {
			t.Fatalf("stand-ins %+v, attribution %+v, class %q", v2.StandIns, v2.Attribution, class)
		}
	})

	// The Messages shape offers Haiku under "none" as well: nothing stands
	// in, and the plain target is left out as before, so the decide is the
	// one it was.
	t.Run("messages unchanged", func(t *testing.T) {
		unknown, request, v2 := apply(t, envelope(`[]`, "null", "null", cloud(haikuPlain), cloud(haikuNone), cloud(haikuUp), cloud(opus)))
		if v2.StandIns != nil || len(unknown) != 1 || unknown[0].ActionID != haikuPlain {
			t.Fatalf("unknown %v, stand-ins %+v", unknown, v2.StandIns)
		}
		if ids := capacityIDs(request); !reflect.DeepEqual(ids, []string{haikuNone, haikuUp, opus}) {
			t.Fatalf("decide capacity %v", ids)
		}
	})

	// A bound plain action (Opus, Sonnet) is untouched, and an unknown
	// target that is no model's plain action is still left out.
	t.Run("others unchanged", func(t *testing.T) {
		unknown, _, v2 := apply(t, envelope(`[]`, "null", "null", cloud(opus), cloud(sonnet), cloud(other)))
		want := []raylinearc.PolicyReceiptExcluded{{ActionID: other, Reason: "unknown_action"}}
		if v2.StandIns != nil || !reflect.DeepEqual(unknown, want) {
			t.Fatalf("unknown %v, stand-ins %+v", unknown, v2.StandIns)
		}
	})
}

// The answer names the plain action and control the executor offered, not
// the stand-in, wherever the decision names the stand-in, with no arm or
// steering level; the service's receipt is not changed in place.
func TestPlainStandInsInTheAnswer(t *testing.T) {
	haikuNone, haikuPlain, opus := strings.Repeat("4", 64), strings.Repeat("5", 64), strings.Repeat("2", 64)
	noneControl := strings.Repeat("d", 64)
	result := func(name, actionID string) routerruntime.RouteResult {
		return routerruntime.RouteResult{Name: name, PolicyActionID: actionID, ControlID: noneControl, ArmID: "arm", ThinkingLevel: "none"}
	}
	receipt := &raylinearc.PolicyDecisionReceipt{
		HeldFrom: &haikuNone, CapacityUnknown: []string{haikuNone, opus},
		Excluded: []raylinearc.PolicyReceiptExcluded{{ActionID: haikuNone, Reason: "context_capacity"}, {ActionID: opus, Reason: "context_capacity"}},
	}
	facts := routerruntime.RouteDecision{
		PolicyActionID: haikuNone, ControlID: noneControl, ArmID: "arm", ThinkingLevel: "none", Receipt: receipt,
		Results: []routerruntime.RouteResult{result("swift", haikuNone), result("sage", opus)},
	}
	restorePlainStandIns(&facts, &selection.RaylineARCRouteExecutionV2{
		StandIns:     map[string]selection.RaylineARCPlainStandIn{haikuNone: {PlainActionID: haikuPlain, PlainControlID: plainControlID}},
		HeldStandIns: map[string]string{haikuNone: haikuPlain},
	})
	if facts.PolicyActionID != haikuPlain || facts.ControlID != plainControlID || facts.ArmID != "" || facts.ThinkingLevel != "" {
		t.Fatalf("decision %+v", facts)
	}
	if swift := facts.Results[0]; swift.PolicyActionID != haikuPlain || swift.ControlID != plainControlID || swift.ArmID != "" || swift.ThinkingLevel != "" {
		t.Fatalf("swift %+v", swift)
	}
	if sage := facts.Results[1]; sage.PolicyActionID != opus || sage.ControlID != noneControl || sage.ArmID != "arm" {
		t.Fatalf("sage %+v", sage)
	}
	if got := facts.Receipt; *got.HeldFrom != haikuPlain || !reflect.DeepEqual(got.CapacityUnknown, []string{haikuPlain, opus}) ||
		got.Excluded[0].ActionID != haikuPlain || got.Excluded[1].ActionID != opus {
		t.Fatalf("receipt %+v", got)
	}
	if *receipt.HeldFrom != haikuNone || receipt.Excluded[0].ActionID != haikuNone || receipt.CapacityUnknown[0] != haikuNone {
		t.Fatal("the service's receipt was changed in place")
	}
	unchanged := routerruntime.RouteDecision{PolicyActionID: haikuNone, Receipt: receipt}
	restorePlainStandIns(&unchanged, nil)
	restorePlainStandIns(&unchanged, &selection.RaylineARCRouteExecutionV2{})
	if unchanged.PolicyActionID != haikuNone || unchanged.Receipt != receipt {
		t.Fatal("no stand-ins must change nothing")
	}
	// The "none" action itself last ran: held_from keeps its own id, whatever
	// the targets stood in for.
	ran := routerruntime.RouteDecision{PolicyActionID: haikuNone, Receipt: receipt}
	restorePlainStandIns(&ran, &selection.RaylineARCRouteExecutionV2{
		StandIns: map[string]selection.RaylineARCPlainStandIn{haikuNone: {PlainActionID: haikuPlain, PlainControlID: plainControlID}},
	})
	if *ran.Receipt.HeldFrom != haikuNone || ran.PolicyActionID != haikuPlain {
		t.Fatalf("held_from %s, action %s", *ran.Receipt.HeldFrom, ran.PolicyActionID)
	}
}

// Through the routes API on a v5 cell whose package binds GLM under "none"
// and "up" but not plain: an executor that offers GLM only under its plain
// control (the Chat and Responses shape) gets GLM decided and answered under
// the plain action and control it offered, with a receipt naming it, as the
// executor checks; one that also offers GLM "none" (the Messages shape) is
// answered as before.
func TestPlainStandInThroughTheRoutesAPI(t *testing.T) {
	router, fake := v5Router(t, "api_format: openai")
	arc := router.Config.Decisions[0].Algorithm.RaylineARC
	arc.RoutesAPI.Enabled = true
	arc.RoutesAPI.Routing = &config.RaylineARCRoutesRoutingConfig{}
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return v5GLMNone })
	fake.tierPickModel = func(picked string) string {
		if picked == v5OpusAction {
			return "claude-opus-5"
		}
		return "glm-5.3-flash"
	}
	schedule, preset := "schedule", "arc-policy.hybrid-default.v1"
	fake.receipt = &raylinearc.PolicyDecisionReceipt{
		Policy: &preset, DecidedBy: raylinearc.PolicyReceiptDecidedBy{Model: &schedule, Thinking: "schedule"},
	}
	lookup := func(t *testing.T, targets ...string) map[string]any {
		t.Helper()
		parts := []string{}
		for _, actionID := range targets {
			parts = append(parts, `{"action_id":"`+actionID+`","worker":"w","placement":"cloud","cache":null,"capacity":null}`)
		}
		envelope := `{"version":"arc-execution-context.v2","turn":{"id":"sha256:` + strings.Repeat("a", 64) +
			`","attempt":1,"epoch":"e1","retry_of":null,"reconsult":null},"event":{"kind":"task_start","basis":"marker_inferred","compaction":null},` +
			`"last_served":null,"history":[],"hold":null,"expect":null,"targets":[` + strings.Join(parts, ",") + `]}`
		body := []byte(`{"model":"arc-1.0","max_tokens":16,"messages":[{"role":"user","content":"summarise this"}],` +
			`"routing":{"policy":{"preset":"` + preset + `"},"execution":` + envelope + `}}`)
		response := router.handleRaylineRoutesAPI(body, routesContext(nil))
		if status := immediateStatusCode(t, response); status != 200 {
			t.Fatalf("status %d: %s", status, response.GetImmediateResponse().Body)
		}
		var payload map[string]any
		if err := json.Unmarshal(response.GetImmediateResponse().Body, &payload); err != nil {
			t.Fatal(err)
		}
		return payload
	}

	payload := lookup(t, v5GLMPlain, v5OpusAction)
	identity, _ := payload["identity"].(map[string]any)
	receipt, _ := payload["receipt"].(map[string]any)
	if identity == nil || identity["action_id"] != v5GLMPlain || identity["policy_action_id"] != v5GLMPlain ||
		identity["control_id"] != plainControlID || identity["arm_id"] != nil {
		t.Fatalf("identity %v", identity)
	}
	if receipt == nil || receipt["action_id"] != identity["action_id"] || receipt["control_id"] != identity["control_id"] {
		t.Fatalf("receipt %v", receipt)
	}
	sent := fake.received()
	last := sent[len(sent)-1]
	if ids := capacityActionIDs(last); !reflect.DeepEqual(ids, []string{v5GLMNone, v5OpusAction}) {
		t.Fatalf("decide capacity %v", ids)
	}

	payload = lookup(t, v5GLMPlain, v5GLMNone, v5OpusAction)
	identity, _ = payload["identity"].(map[string]any)
	if identity == nil || identity["action_id"] != v5GLMNone {
		t.Fatalf("messages-shaped identity %v", identity)
	}
	sent = fake.received()
	if ids := capacityActionIDs(sent[len(sent)-1]); !reflect.DeepEqual(ids, []string{v5GLMNone, v5OpusAction}) {
		t.Fatalf("messages-shaped decide capacity %v", ids)
	}
}

func capacityActionIDs(request raylinearc.PolicyDecisionRequest) []string {
	ids := []string{}
	for _, entry := range request.Capacity {
		ids = append(ids, entry.ActionID)
	}
	return ids
}
