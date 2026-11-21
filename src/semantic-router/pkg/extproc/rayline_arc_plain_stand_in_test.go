//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The plain control: the default base with no instruction.
var plainControlID = thinkingcontrol.Control{Native: "default"}.ID()

// v5GLMPlain is the fixture's GLM at its default base with no instruction:
// an action the fixture does not bind (it binds GLM under "none" and "up").
const v5GLMPlain = "bd202891c3ed5bf9af5250868da17cb3ad68a05184ec5fa757876880257a5b26"

// The aliases come from the package: a model bound under "none" and not
// under its plain control gets one (GLM); a model bound under its plain
// control does not (Opus, native only); nor does a plain action the package
// has but the cell leaves unbound, or one that two "none" actions read as.
func TestPlainAliasesComeFromThePackage(t *testing.T) {
	if got := v5ActionID(plainControlID, "glm-5.3-flash"); got != v5GLMPlain {
		t.Fatalf("plain GLM action id %s", got)
	}
	if got := v5ActionID(plainControlID, "claude-opus-5"); got != v5OpusAction {
		t.Fatalf("plain Opus action id %s, want the fixture's native Opus", got)
	}
	// RFC 8785 leaves "&", "<", ">", U+2028 and U+2029 unescaped, as the
	// package's identity does; an HTML-safe encoder would not.
	if got, want := v5ActionID(plainControlID, "a&b<c>\u2028"), sha256Hex([]byte("{\"control_id\":\""+plainControlID+"\",\"model\":\"a&b<c>\u2028\"}")); got != want {
		t.Fatalf("JCS identity %s, want %s", got, want)
	}
	cfg := v5ConfigWith(t, func(rendered string) string { return rendered })
	policy := cfg.Decisions[0].Algorithm.RaylineARC.PolicyService
	want := map[string]plainAlias{v5GLMPlain: {standInActionID: v5GLMNone, plainControlID: plainControlID, model: "glm-5.3-flash"}}
	if got := plainAliasesOf(policy); !reflect.DeepEqual(got, want) {
		t.Fatalf("aliases %+v", got)
	}
	withPlain := *policy
	withPlain.Bindings = append(append([]config.RaylineARCPolicyBinding{}, policy.Bindings...), config.RaylineARCPolicyBinding{ActionID: v5GLMPlain, Worker: "glm"})
	if got := plainAliasesOf(&withPlain); got != nil {
		t.Fatalf("a cell that binds the plain action still aliases it: %+v", got)
	}
	if plainAliasesOf(nil) != nil {
		t.Fatal("no policy service, no aliases")
	}

	none := mustV5Control(t, policy, v5GLMNone)
	actions := map[string]config.RaylineARCPolicyActionV5{v5GLMNone: {ActionID: v5GLMNone, Model: "glm-5.3-flash", Control: none}}
	if got := plainAliasesOfActions(actions, func(id string) bool { return id == v5GLMPlain }); got != nil {
		t.Fatalf("a plain action the package has still aliases: %+v", got)
	}
	neutral := thinkingcontrol.Control{Native: "default", Instruction: &thinkingcontrol.Instruction{
		Level: "none", NeutralText: "Until the next steering instruction, carry on as you were.",
	}}
	twin := v5ActionID(neutral.ID(), "glm-5.3-flash")
	actions[twin] = config.RaylineARCPolicyActionV5{ActionID: twin, Model: "glm-5.3-flash", Control: neutral}
	for range 20 {
		if got := plainAliasesOfActions(actions, func(string) bool { return false }); got != nil {
			t.Fatalf("an ambiguous plain action aliases: %+v", got)
		}
	}
}

func mustV5Control(t *testing.T, policy *config.RaylineARCPolicyServiceConfig, actionID string) thinkingcontrol.Control {
	t.Helper()
	action, ok := policy.PackageV5Action(actionID)
	if !ok {
		t.Fatalf("no v5 action %s", actionID)
	}
	return action.Control
}

// The property the stopgap claims, checked over the envelope's field
// combinations: an envelope that never names the "none" action decides
// exactly as on an oracle cell that binds the plain action as that "none"
// action (same refusal, same decide members with the plain id read as the
// "none" one); one that names it anywhere decides exactly as without the
// stopgap.
func TestPlainAliasMatchesAnOracleCell(t *testing.T) {
	local, sonnet, haikuNone := strings.Repeat("1", 64), strings.Repeat("3", 64), strings.Repeat("4", 64)
	haikuPlain := strings.Repeat("5", 64)
	bindings := map[string]policyBinding{
		local: {model: "qwen3.8-27b"}, sonnet: {model: "claude-sonnet-5.5"}, haikuNone: {model: "claude-haiku-5.5", level: "none"},
	}
	cell := &policyServiceScorer{bindings: bindings}
	oracleBindings := map[string]policyBinding{haikuPlain: {model: "claude-haiku-5.5"}}
	for id, binding := range bindings {
		oracleBindings[id] = binding
	}
	oracle := &policyServiceScorer{bindings: oracleBindings}
	aliases := map[string]plainAlias{haikuPlain: {standInActionID: haikuNone, plainControlID: plainControlID, model: "claude-haiku-5.5"}}
	available := []string{local, sonnet, haikuNone}

	overWindow := `{"fits":false,"reason":"context_capacity","required":42733,"available":32768,"available_source":"configured","units":null,"provenance":"caller_bounded"}`
	target := func(actionID, capacity string) string {
		return `{"action_id":"` + actionID + `","worker":"w","placement":"cloud","cache":null,"capacity":` + capacity + `}`
	}
	type outcome struct {
		class   string
		unknown []raylinearc.PolicyReceiptExcluded
		members string
		aliased bool
	}
	decide := func(t *testing.T, scorer *policyServiceScorer, body string, aliases map[string]plainAlias) (outcome, bool) {
		routing, _, refusal := readRaylineRoutesRouting(executionV2Body(`{"execution":`+body+`}`), nil, config.DefaultRaylineARCValueUSDMax)
		if refusal != nil {
			return outcome{}, false
		}
		controls := raylineRoutesControls(routing, &config.RaylineARCRoutesRoutingConfig{}, time.Now())
		applyPlainAliases(controls.ExecutionV2, aliases)
		offer := available
		if scorer == oracle {
			offer = append(append([]string{}, available...), haikuPlain) // the oracle cell offers what it binds
		}
		request := &raylinearc.PolicyDecisionRequest{Selection: raylinearc.PolicySelection{AvailableActionIDs: offer}}
		unknown, class := applyRoutePolicyAndExecutionV2(request, scorer, controls)
		members, _ := json.Marshal(decideMembers(t, request, nil))
		return outcome{class: class, unknown: unknown, members: string(members), aliased: len(controls.ExecutionV2.StandIns) > 0}, true
	}

	ids := map[string]string{"plain": haikuPlain, "none": haikuNone}
	cases := 0
	for _, targets := range [][]string{{"plain"}, {"none"}, {"plain", "none"}, {}} {
		for _, infeasible := range []bool{false, true} {
			for _, history := range []string{"", "plain", "none", "both"} {
				for _, served := range []string{"", "plain", "none", "plain-no-message"} {
					for _, hold := range []string{"", "plain", "none"} {
						parts := []string{target(sonnet, "null"), target(local, overWindow)}
						for _, name := range targets {
							capacity := "null"
							if name == "plain" && infeasible {
								capacity = overWindow
							}
							parts = append(parts, target(ids[name], capacity))
						}
						attribution := "[]"
						switch history {
						case "plain", "none":
							attribution = `[{"message":1,"action_id":"` + ids[history] + `"}]`
						case "both":
							attribution = `[{"message":1,"action_id":"` + haikuNone + `"},{"message":3,"action_id":"` + haikuPlain + `"}]`
						}
						lastServed := "null"
						switch served {
						case "plain", "none":
							lastServed = `{"action_id":"` + ids[served] + `","worker":"w","message":1}`
						case "plain-no-message":
							lastServed = `{"action_id":"` + haikuPlain + `","worker":"w","message":null}`
						}
						holdJSON := "null"
						if hold != "" {
							holdJSON = `{"scope":"action","action_id":"` + ids[hold] + `"}`
						}
						body := `{"version":"arc-execution-context.v2","turn":{"id":"sha256:` + strings.Repeat("a", 64) + `","attempt":1,"epoch":"e1","retry_of":null,"reconsult":null},` +
							`"event":{"kind":"task_start","basis":"marker_inferred","compaction":null},"last_served":` + lastServed + `,"history":` + attribution +
							`,"hold":` + holdJSON + `,"expect":null,"targets":[` + strings.Join(parts, ",") + `]}`
						name := strings.Join(targets, "+") + "/inf=" + map[bool]string{true: "y", false: "n"}[infeasible] + "/h=" + history + "/ls=" + served + "/hold=" + hold
						got, parsed := decide(t, cell, body, aliases)
						if !parsed {
							continue
						}
						cases++
						namesNone := strings.Contains(body, haikuNone)
						namesPlain := strings.Contains(body, haikuPlain)
						if namesNone || !namesPlain {
							want, _ := decide(t, cell, body, nil)
							if !reflect.DeepEqual(got, want) {
								t.Errorf("%s: with the stopgap %+v, without %+v", name, got, want)
							}
							continue
						}
						want, _ := decide(t, oracle, body, nil)
						want.members = strings.ReplaceAll(want.members, haikuPlain, haikuNone)
						want.aliased = true
						if !reflect.DeepEqual(got, want) {
							t.Errorf("%s: with the stopgap %+v, oracle %+v", name, got, want)
						}
					}
				}
			}
		}
	}
	if cases < 100 {
		t.Fatalf("only %d envelopes parsed", cases)
	}
}

// Each lookup that used an alias logs it once, after its decision, with how
// it ended, also when only its history named the plain action.
func TestPlainAliasIsLoggedOncePerLookup(t *testing.T) {
	haikuNone, haikuPlain := strings.Repeat("4", 64), strings.Repeat("5", 64)
	v2 := &selection.RaylineARCRouteExecutionV2{
		Attribution: []raylinearc.PolicyAttribution{{Message: 1, ActionID: haikuPlain}}, TargetActionIDs: []string{strings.Repeat("3", 64)},
	}
	applyPlainAliases(v2, map[string]plainAlias{haikuPlain: {standInActionID: haikuNone, plainControlID: plainControlID, model: "claude-haiku-5.5"}})
	if v2.Attribution[0].ActionID != haikuNone {
		t.Fatal("the history was not read as the stand-in")
	}
	logs := captureLogs(t)
	logPlainAliases(v2, "episode", errors.New("refused"))
	logPlainAliases(&selection.RaylineARCRouteExecutionV2{}, "episode", nil)
	lines := 0
	for _, entry := range logs.All() {
		if entry.Message == "rayline_arc_route_target_stand_in" {
			lines++
			if fields := entry.ContextMap(); fields["plain_action_id"] != haikuPlain || fields["stand_in_action_id"] != haikuNone || fields["outcome"] != "failed" {
				t.Fatalf("logged %v", fields)
			}
		}
	}
	if lines != 1 {
		t.Fatalf("%d lines, want one", lines)
	}
	if encoded, _ := json.Marshal(v2); strings.Contains(string(encoded), "PlainActionID") {
		t.Fatal("the bookkeeping is encoded with the envelope")
	}
}

// The answer names only what the executor offered: every occurrence of the
// stand-in id, in any member, and the control, arm and steering level of an
// object that named it as its action.
func TestPlainAliasesInTheAnswer(t *testing.T) {
	haikuNone, haikuPlain, opus := strings.Repeat("4", 64), strings.Repeat("5", 64), strings.Repeat("2", 64)
	noneControl := strings.Repeat("d", 64)
	answer := `{"policy_action_id":"` + haikuNone + `","thinking_level":"none",` +
		`"identity":{"action_id":"` + haikuNone + `","policy_action_id":"` + haikuNone + `","control_id":"` + noneControl + `","arm_id":"arm"},` +
		`"alternatives":[{"action_id":"` + haikuNone + `","thinking_level":"none","model":"m"},{"action_id":"` + opus + `","thinking_level":"none"}],` +
		`"receipt":{"action_id":"` + haikuNone + `","control_id":"` + noneControl + `","held_from":"` + haikuNone + `","excluded":[{"action_id":"` + haikuNone + `","reason":"r"}],"capacity_unknown":["` + haikuNone + `"]},` +
		`"candidates":{"rows":[["` + haikuNone + `"]]}}`
	renamed, err := renamePlainAliasesInAnswer([]byte(answer), map[string]selection.RaylineARCPlainStandIn{
		haikuNone: {PlainActionID: haikuPlain, PlainControlID: plainControlID, Model: "claude-haiku-5.5"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(renamed), haikuNone) {
		t.Fatalf("the answer still names the stand-in: %s", renamed)
	}
	var got map[string]any
	_ = json.Unmarshal(renamed, &got)
	identity := got["identity"].(map[string]any)
	if identity["control_id"] != plainControlID || identity["arm_id"] != nil || got["thinking_level"] != nil {
		t.Fatalf("identity %v, level %v", identity, got["thinking_level"])
	}
	alternatives := got["alternatives"].([]any)
	if first := alternatives[0].(map[string]any); first["thinking_level"] != nil || first["model"] != "m" {
		t.Fatalf("alternative %v", first)
	}
	if second := alternatives[1].(map[string]any); second["action_id"] != opus || second["thinking_level"] != "none" {
		t.Fatalf("another action's alternative changed: %v", second)
	}
	if same, _ := renamePlainAliasesInAnswer([]byte(answer), nil); string(same) != answer {
		t.Fatal("an answer without stand-ins must be unchanged")
	}
}

// Through the routes API on a v5 cell whose package binds GLM under "none"
// and "up" but not plain: an executor that offers GLM only under its plain
// control (the Chat and Responses shape) gets GLM decided and answered under
// the plain action and control it offered, as the executor checks, and the
// answer names the stand-in nowhere; one that also offers GLM "none" (the
// Messages shape) is answered as before.
func TestPlainAliasThroughTheRoutesAPI(t *testing.T) {
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
	lookup := func(t *testing.T, targets ...string) (map[string]any, string) {
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
		raw := response.GetImmediateResponse().Body
		var payload map[string]any
		if err := json.Unmarshal(raw, &payload); err != nil {
			t.Fatal(err)
		}
		return payload, string(raw)
	}

	payload, raw := lookup(t, v5GLMPlain, v5OpusAction)
	identity, _ := payload["identity"].(map[string]any)
	receipt, _ := payload["receipt"].(map[string]any)
	if identity == nil || identity["action_id"] != v5GLMPlain || identity["policy_action_id"] != v5GLMPlain ||
		identity["control_id"] != plainControlID || identity["arm_id"] != nil || payload["policy_action_id"] != v5GLMPlain {
		t.Fatalf("identity %v", identity)
	}
	if receipt == nil || receipt["action_id"] != identity["action_id"] || receipt["control_id"] != identity["control_id"] {
		t.Fatalf("receipt %v", receipt)
	}
	if strings.Contains(raw, v5GLMNone) {
		t.Fatalf("the answer names the stand-in: %s", raw)
	}
	sent := fake.received()
	if ids := capacityActionIDs(sent[len(sent)-1]); !reflect.DeepEqual(ids, []string{v5GLMNone, v5OpusAction}) {
		t.Fatalf("decide capacity %v", ids)
	}

	payload, _ = lookup(t, v5GLMPlain, v5GLMNone, v5OpusAction)
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
