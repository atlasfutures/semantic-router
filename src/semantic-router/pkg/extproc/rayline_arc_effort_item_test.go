//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Opus 5.5's actions at the high base: the native-only control, and the
// effort-item controls (ladder opus-effort-item-high-v1) at level none and low.
const (
	v5OpusHigh              = "7a3c5e9b1d2f4a6c8e0b2d4f6a8c0e1b3d5f7a9c2e4b6d8f0a1c3e5b7d9f2a4c"
	v5OpusHighControl       = "8c4faa90467d7151b03810424e673723a62bdface213d36659862037aaf4dc2d"
	v5OpusEffortNone        = "5e1f0a3b7c9d2e4f6a8b0c1d3e5f7a9b2c4d6e8f0a1b3c5d7e9f2a4b6c8d0e1f"
	v5OpusEffortNoneControl = "382b5a7fd7543311f73cc135fad21a303c4e301fb363a3696146ac707406d1bc"
	v5OpusEffortLow         = "6f2a1b4c8d0e3f5a7b9c1d2e4f6a8b0c3d5e7f9a1b2c4d6e8f0a3b5c7d9e1f2a"
	v5OpusEffortLowControl  = "4d90550d326886b21d6bed20a0dc15a63a1c590765c3e769e48646a47c3cb7e7"
)

// withOpusEffortItemActions adds the three actions, copies of the fixture's
// Opus action carrying the registry's controls.
func withOpusEffortItemActions(manifest []byte) ([]byte, []string) {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(manifest, &doc); err != nil {
		panic(err)
	}
	var actions []map[string]any
	if err := json.Unmarshal(doc["actions"], &actions); err != nil {
		panic(err)
	}
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		panic(err)
	}
	for _, action := range actions {
		if action["action_id"] != v5OpusAction {
			continue
		}
		for _, add := range []struct{ id, control, level string }{
			{v5OpusHigh, v5OpusHighControl, ""},
			{v5OpusEffortNone, v5OpusEffortNoneControl, "none"},
			{v5OpusEffortLow, v5OpusEffortLowControl, "low"},
		} {
			control, ok := registry.Control(add.control)
			if !ok {
				panic("no registry control " + add.control)
			}
			encoded, _ := json.Marshal(action)
			var copied map[string]any
			_ = json.Unmarshal(encoded, &copied)
			copied["action_id"], copied["control_id"] = add.id, add.control
			if add.level == "" {
				copied["control"] = map[string]any{
					"base": map[string]any{"native": control.Native}, "budget_tokens": nil, "instruction": nil,
				}
				actions = append(actions, copied)
				continue
			}
			copied["level"] = add.level
			copied["control"] = map[string]any{
				"base": map[string]any{"native": control.Native}, "budget_tokens": nil,
				"instruction": map[string]any{
					"level": add.level, "text": control.Instruction.Text, "neutral_text": control.Instruction.NeutralText,
					"rules": map[string]any{
						"emit": "on_change_v1", "placement": "after_new_input_v1", "replay": "ledger_v1",
						"unit": thinkingcontrol.UnitMessagesEffortItem,
					},
				},
			}
			actions = append(actions, copied)
		}
		break
	}
	encoded, err := json.Marshal(actions)
	if err != nil {
		panic(err)
	}
	doc["actions"] = encoded
	out, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		panic(err)
	}
	return out, []string{v5OpusHigh, v5OpusEffortNone, v5OpusEffortLow}
}

// onOpus55WithEffortItems serves the Opus worker as Opus 5.5, the cell with
// the effort-item ladder, and binds its high-base actions.
func onOpus55WithEffortItems(config string) string {
	config = strings.Replace(config, "provider_model_id: anthropic/claude-opus-5\n",
		"provider_model_id: anthropic/claude-opus-5.5\n", 1)
	opus := "              - action_id: " + v5OpusAction + "\n                worker: opus\n"
	return strings.Replace(config, opus, opus+
		"              - action_id: "+v5OpusHigh+"\n                worker: opus\n"+
		"              - action_id: "+v5OpusEffortNone+"\n                worker: opus\n"+
		"              - action_id: "+v5OpusEffortLow+"\n                worker: opus\n", 1)
}

// An Opus effort-item episode reaches the provider as pathfinder places it
// (ADR 0143, 0147): a tool_result-only tail gains the new-input block and the
// item after it, later calls (a native-only one included) replay both at their
// anchor, so the request-level effort and the cached prefix never move; every
// call sends the beta. Claude
// Code's own per-turn output_config is stripped before placement, so only
// placement's item reaches the provider.
func TestV5EffortItemPlacesReplaysAndDeclaresItsBeta(t *testing.T) {
	router, fake := v5RouterWithManifest(t, "anthropic", onOpus55WithEffortItems, withOpusEffortItemActions)
	router.Config.KeepUnservedClaudeThinking = true
	const episode = "v5-effort-item"
	start := `{"role":"user","content":"Start.","output_config":{"effort":"max"}}`
	toolTurn := func(id string) string {
		return `{"role":"assistant","content":[{"type":"tool_use","id":"` + id + `","name":"Bash","input":{"command":"ls"}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + id + `","content":"a"}]}`
	}
	body := func(units ...string) string {
		return `{"model":"auto","max_tokens":1024,"messages":[` + strings.Join(units, ",") + `]}`
	}
	item := `{"role":"system","content":[],"output_config":{"effort":"low"}}`
	continued := `{"type":"text","text":"Continue."}`
	for _, call := range []struct {
		action, client string
		items          int
	}{
		{v5OpusEffortNone, body(start), 0},
		{v5OpusEffortLow, body(start, toolTurn("c1")), 1},
		{v5OpusEffortLow, body(start, toolTurn("c1"), toolTurn("c2")), 1},
		// A native-only action on the episode replays the item, so it too
		// declares the beta.
		{v5OpusHigh, body(start, toolTurn("c1"), toolTurn("c2"), toolTurn("c3")), 1},
	} {
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return call.action })
		dispatched, ctx := v5Dispatch(t, router, episode, call.client)
		sent := string(dispatched.GetBodyMutation().GetBody())
		if got := strings.Count(sent, item); got != call.items {
			t.Fatalf("%s: %d effort items, want %d:\n%s", call.action[:8], got, call.items, sent)
		}
		if call.items > 0 && !strings.Contains(sent, continued+`]},`+item) {
			t.Fatalf("%s: the item does not follow the new-input block:\n%s", call.action[:8], sent)
		}
		if strings.Contains(sent, `"max"`) {
			t.Fatalf("%s: the client's per-turn output_config reached the provider:\n%s", call.action[:8], sent)
		}
		if !strings.Contains(sent, `"output_config":{"effort":"high"}`) {
			t.Fatalf("%s: the request-level effort moved off the episode's high base:\n%s", call.action[:8], sent)
		}
		beta := ""
		for _, header := range dispatched.GetHeaderMutation().GetSetHeaders() {
			if strings.EqualFold(header.GetHeader().GetKey(), anthropicBetaHeader) {
				beta = string(header.GetHeader().GetRawValue()) + header.GetHeader().GetValue()
			}
		}
		if !strings.Contains(beta, thinkingcontrol.EffortItemBeta) {
			t.Fatalf("%s: anthropic-beta %q does not declare the effort item", call.action[:8], beta)
		}
		if trace := ctx.RaylineARCThinking; trace == nil || trace.Lever != thinkingControlLever ||
			trace.EffortItemHeader != thinkingcontrol.EffortItemBeta {
			t.Fatalf("%s: trace = %+v", call.action[:8], trace)
		}
	}
}
