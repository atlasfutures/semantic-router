/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package thinkingcontrol

import (
	"strings"
	"testing"
)

// ADR 0131, as pathfinder#4120 states it: on a cell whose row states
// text_tool_tail {placement: append_to_tool}, a unit after a Chat tool run is
// a text part of the run's last tool message.

func TestOnlyKimiK3ChatSpellsATextToolTailAsAppendToTool(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	spelled := 0
	for _, format := range []string{FormatMessages, FormatChat, FormatResponses} {
		for _, model := range []string{"moonshotai/kimi-k3", "z-ai/glm-5.3-flash"} {
			cell, err := reg.Cell(model, "openrouter", format)
			if err != nil {
				continue
			}
			want := model == "moonshotai/kimi-k3" && format == FormatChat
			if cell.TextToolTailAppendToTool != want {
				t.Errorf("%s openrouter x %s: append_to_tool %v, want %v", model, format, cell.TextToolTailAppendToTool, want)
			}
			if cell.TextToolTailAppendToTool {
				spelled++
			}
		}
	}
	if spelled != 1 {
		t.Fatalf("%d cells spell append_to_tool, want kimi-k3 chat alone", spelled)
	}
}

// A stated spelling compiles only as {placement: append_to_tool,
// task_fidelity: [evidence]} on a chat cell; anything else is refused, never
// ignored.
func TestTheRegistryRefusesATextToolTailItCannotSpell(t *testing.T) {
	for _, tc := range []struct {
		name, format, stated string
		ok                   bool
	}{
		{"chat", FormatChat, `{"placement":"append_to_tool","task_fidelity":["registry:x"]}`, true},
		{"messages", FormatMessages, `{"placement":"append_to_tool","task_fidelity":["registry:x"]}`, false},
		{"another placement", FormatChat, `{"placement":"insert_after","task_fidelity":["registry:x"]}`, false},
		{"no evidence", FormatChat, `{"placement":"append_to_tool","task_fidelity":[]}`, false},
		{"another key", FormatChat, `{"placement":"append_to_tool","task_fidelity":["registry:x"],"x":1}`, false},
		{"no placement", FormatChat, `{"task_fidelity":["registry:x"]}`, false},
	} {
		parsed, err := parseJSON(embeddedArtifact)
		if err != nil {
			t.Fatal(err)
		}
		stated, err := parseJSON([]byte(tc.stated))
		if err != nil {
			t.Fatal(err)
		}
		var target *value
		for _, cell := range parsed.get("cells").items {
			if cell.get("model").str == "z-ai/glm-5.3-flash" && cell.get("provider").str == "openrouter" &&
				cell.get("format").str == tc.format {
				target = cell
			}
		}
		if target == nil {
			t.Fatalf("%s: no glm-5.3-flash openrouter cell", tc.name)
		}
		target.set("text_tool_tail", stated)
		tampered, err := jcs(parsed)
		if err != nil {
			t.Fatal(err)
		}
		reg, err := Load(tampered)
		if !tc.ok {
			if err == nil || !strings.Contains(err.Error(), "text_tool_tail") {
				t.Errorf("%s: loaded: %v", tc.name, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		cell, err := reg.Cell("z-ai/glm-5.3-flash", "openrouter", FormatChat)
		if err != nil || !cell.TextToolTailAppendToTool {
			t.Errorf("%s: cell spells append_to_tool %v, err %v", tc.name, cell != nil && cell.TextToolTailAppendToTool, err)
		}
	}
}

func placeSpelled(t *testing.T, placer *Placer, raw string, control *Control, rules tailRules) (string, Receipt, error) {
	t.Helper()
	body, err := parseJSON([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	placed, receipt, err := placer.place(body, control, rules)
	if err != nil {
		return "", receipt, err
	}
	out, err := dump(placed)
	if err != nil {
		t.Fatal(err)
	}
	return string(out), receipt, nil
}

// The spelling moves only a unit after a tool message: a user-message tail,
// and the hoisted image message a Chat image run ends in, keep their rows.
func TestAppendToToolSpellsOnlyATextToolTail(t *testing.T) {
	up := registryControl(t, glmChatUp)
	spelled := tailRules{imageToolTailAdmitted: true, appendToTool: true}
	for _, tc := range []struct {
		name, run, want string
	}{
		{"text tool run", chatTextRun, `{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"README.md"},{"type":"text","text":"` + up.Instruction.Text + `"}]}]}`},
		{"hoisted image run", chatImageRun, `{"type":"image_url","image_url":{"url":"` + tailPNG + `"}},{"type":"text","text":"` + up.Instruction.Text + `"}]}]}`},
		{"user tail", `{"role":"user","content":"Go on."}`, `{"role":"user","content":[{"type":"text","text":"Go on."},{"type":"text","text":"` + up.Instruction.Text + `"}]}]}`},
	} {
		placer, err := NewPlacer(FormatChat)
		if err != nil {
			t.Fatal(err)
		}
		raw := `{"model":"m","messages":[{"role":"user","content":"Start."},` + tc.run + `]}`
		out, receipt, err := placeSpelled(t, placer, raw, up, spelled)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if receipt.Written == nil || *receipt.Written != WrittenInstruction || !strings.HasSuffix(out, tc.want) {
			t.Errorf("%s: wrote %v:\n got %s\nwant suffix %s", tc.name, receipt.Written, out, tc.want)
		}
		ledger := placer.State().Ledger
		wantPlacement := PlacementAppend
		if tc.name == "text tool run" {
			wantPlacement = PlacementAppendToTool
		}
		if len(ledger) != 1 || ledger[0].Placement != wantPlacement {
			t.Errorf("%s: ledger %+v, want placement %s", tc.name, ledger, wantPlacement)
		}
	}
}

// A stored append_to_tool item resumes, and a non-Chat placer refuses the
// spelling.
func TestAnAppendToToolLedgerResumesOnlyOnChat(t *testing.T) {
	up := registryControl(t, glmChatUp)
	placer, err := NewPlacer(FormatChat)
	if err != nil {
		t.Fatal(err)
	}
	raw := `{"model":"m","messages":[{"role":"user","content":"Start."},` + chatTextRun + `]}`
	if _, _, placeErr := placeSpelled(t, placer, raw, up, tailRules{appendToTool: true}); placeErr != nil {
		t.Fatal(placeErr)
	}
	resumed, err := ResumePlacer(placer.State())
	if err != nil {
		t.Fatalf("an append_to_tool ledger did not resume: %v", err)
	}
	again, _, err := placeSpelled(t, resumed, raw, up, tailRules{appendToTool: true})
	if err != nil || !strings.Contains(again, `"content":[{"type":"text","text":"README.md"},{"type":"text"`) {
		t.Fatalf("the replay did not keep the tool message spelling: %v\n%s", err, again)
	}
	messages, err := NewPlacer(FormatMessages)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = placeSpelled(t, messages, `{"model":"m","messages":[{"role":"user","content":"Start."}]}`, up,
		tailRules{appendToTool: true})
	if err == nil || !strings.Contains(err.Error(), "append_to_tool is a Chat tool-tail spelling") {
		t.Fatalf("a messages placer took the spelling: %v", err)
	}
}

// The in-band system fold and the spelling do not compose (pathfinder
// refuses both on one call); no cell admits both.
func TestAppendToToolRefusesTheInbandSystemFold(t *testing.T) {
	control := *registryControl(t, glmChatUp)
	instruction := *control.Instruction
	instruction.InbandSystem = InbandSystemFold
	control.Instruction = &instruction
	placer, err := NewPlacer(FormatChat)
	if err != nil {
		t.Fatal(err)
	}
	raw := `{"model":"m","messages":[{"role":"user","content":"Start."},` + chatTextRun + `]}`
	_, _, err = placeSpelled(t, placer, raw, &control, tailRules{appendToTool: true})
	want := "call 0: append_to_tool and the in-band system fold do not compose; no cell admits both (ADR 0131)"
	if err == nil || err.Error() != want {
		t.Fatalf("refusal = %v, want %q", err, want)
	}
}
