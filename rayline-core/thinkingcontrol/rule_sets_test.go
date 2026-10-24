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

// The Messages effort item (pathfinder ADR 0143, placed after_new_input_v1
// per ADR 0147) and the every_turn_v1 emit rule are rule sets of their own.
// VSR reads them so the registry and pathfinder's ids load, and refuses them
// at admission and placement, where rendering them as text would write the
// wrong bytes at the wrong turns.
var unimplementedRuleSetCells = []struct {
	model, format string
	ids           []string
}{
	{"anthropic/claude-opus-5.5", FormatMessages, []string{
		"382b5a7fd7543311f73cc135fad21a303c4e301fb363a3696146ac707406d1bc",
		"4d90550d326886b21d6bed20a0dc15a63a1c590765c3e769e48646a47c3cb7e7",
		"e5f9971d3095eb5f5997be15b1c5c7d72947d37c75ccb07ddd4daca8107d66ba",
	}},
	{"openai/gpt-6.1-sol", FormatMessages, []string{
		"78da4ab5bf28d53e78dbf14e60353c60e77ff8b8b8775cfa7955d95b84d43b58",
		"8becd4c8d1ab1ecf86c667212ae543be644c6335d25d3034ae4b98e675b06aa0",
		"f7c965779815ce7eb2092cd91ccb81b05ba456dd4eec5863980f1873b83735c9",
	}},
	{"meta/muse-spark-1.3-contributor", FormatMessages, []string{
		"5fb4c9692b6953a0cecd9702ffd17bfa5b0bc2e5dccef60d383d058929cffeec",
		"897bab23ec521e0acf1563a25aad9a326e8e680359811cad807d2b9059a2766e",
		"ba63c111149c30c1e05f20b420ba9b8154d25ee903ae2061597928f748efce5f",
	}},
	{"meta/muse-spark-1.3-contributor", FormatChat, []string{
		"5fb4c9692b6953a0cecd9702ffd17bfa5b0bc2e5dccef60d383d058929cffeec",
		"897bab23ec521e0acf1563a25aad9a326e8e680359811cad807d2b9059a2766e",
		"ba63c111149c30c1e05f20b420ba9b8154d25ee903ae2061597928f748efce5f",
	}},
}

func TestEffortItemAndEveryTurnControlsLoadUnderPathfindersIDsAndAreNotAdmitted(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range unimplementedRuleSetCells {
		cell, err := reg.Cell(tc.model, "openrouter", tc.format)
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range tc.ids {
			control, ok := reg.Control(id)
			if !ok {
				t.Fatalf("control %s is not in the registry", prefix12(id))
			}
			if control.ID() != id {
				t.Fatalf("control %s re-hashes to %s", prefix12(id), prefix12(control.ID()))
			}
			if !cell.Admits(id) {
				t.Fatalf("%s %s does not list control %s", tc.model, tc.format, prefix12(id))
			}
			if control.Instruction == nil || control.Instruction.Implemented() {
				t.Fatalf("control %s reads as a rule set VSR places", prefix12(id))
			}
			if _, err := reg.Admit(tc.model, "openrouter", tc.format, control, true); err == nil ||
				!strings.Contains(err.Error(), "which VSR does not implement") {
				t.Fatalf("control %s was admitted on %s %s: %v", prefix12(id), tc.model, tc.format, err)
			}
			placer, err := NewPlacer(tc.format)
			if err != nil {
				t.Fatal(err)
			}
			if err := placer.check(&control); err == nil || !strings.Contains(err.Error(), "VSR does not place") {
				t.Fatalf("the placer took control %s: %v", prefix12(id), err)
			}
		}
	}
}

func TestTheEffortItemSpellsItsOwnPlacementAndEveryTurnOnlyTheTextRules(t *testing.T) {
	item := Control{Native: "high", Instruction: &Instruction{
		Level: "low", Text: "low", NeutralText: "high", Unit: UnitMessagesEffortItem,
	}}
	if got := item.ID(); got != "4d90550d326886b21d6bed20a0dc15a63a1c590765c3e769e48646a47c3cb7e7" {
		t.Fatalf("an effort item control hashes to %s, not pathfinder's id", prefix12(got))
	}
	const text = "Until the next steering instruction, use your normal judgement."
	everyTurn := Control{Native: "default", Instruction: &Instruction{Level: "none", NeutralText: text, Emit: EmitEveryTurn}}
	if got := everyTurn.ID(); got != "ba63c111149c30c1e05f20b420ba9b8154d25ee903ae2061597928f748efce5f" {
		t.Fatalf("an every_turn control hashes to %s, not pathfinder's id", prefix12(got))
	}
	// Any other combination is no rule set pathfinder names.
	for _, rules := range []string{
		`{"emit":"every_turn_v1","placement":"turn_tail_v2","replay":"ledger_v1","unit":"configuration_update_v1"}`,
		`{"emit":"every_turn_v1","inband_system":"fold_before_unit_v1","placement":"turn_tail_v2","replay":"ledger_v1"}`,
		`{"emit":"on_change_v1","placement":"turn_tail_v2","replay":"ledger_v1","unit":"messages_effort_item_v1"}`,
		`{"emit":"on_change_v1","placement":"after_new_input_v1","replay":"ledger_v1"}`,
		`{"emit":"every_turn_v1","placement":"after_new_input_v1","replay":"ledger_v1","unit":"messages_effort_item_v1"}`,
		`{"emit":"every_other_turn_v1","placement":"turn_tail_v2","replay":"ledger_v1"}`,
	} {
		raw := `{"base":{"native":"high"},"budget_tokens":null,"instruction":` +
			`{"level":"low","neutral_text":"high","rules":` + rules + `,"text":"low"}}`
		if _, err := ParseControl([]byte(raw)); err == nil {
			t.Fatalf("rules %s parsed", rules)
		}
	}
}

// pathfinder f24b141c28 / f2cf0f4361: gemini-4-argon's six cells admit its
// not-sent default base and no instruction.
func TestGemini4ArgonAdmitsItsDefaultBaseOnEveryCell(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	native := Control{Native: "default"}
	for _, at := range [][2]string{
		{"anthropic", FormatMessages}, {"openai", FormatChat}, {"openai", FormatResponses},
		{"openrouter", FormatMessages}, {"openrouter", FormatChat}, {"openrouter", FormatResponses},
	} {
		cell, err := reg.Admit("google/gemini-4-argon", at[0], at[1], native, false)
		if err != nil {
			t.Fatalf("%s x %s: %v", at[0], at[1], err)
		}
		if cell.Instruction != AdmissionRefused {
			t.Fatalf("%s x %s admits instructions: %s", at[0], at[1], cell.Instruction)
		}
	}
}

// pathfinder native_bases: gpt-5.6-terra's openrouter x messages cell admits
// its native default control (628d285537ca) again, beside steer3 at xhigh.
// Steer3 did not separate at default, so no instruction control rides it.
func TestGPT56TerraAdmitsItsNativeDefaultBesideSteer3AtXhigh(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	const model = "openai/gpt-5.6-terra"
	native := Control{Native: "default"}
	if id := native.ID(); id != "628d285537ca7cdb52c4bd3433cea7a1ec36005a14dbc67ffa1c40487d7b8ad0" {
		t.Fatalf("the native default control is %s", id)
	}
	cell, err := reg.Admit(model, "openrouter", FormatMessages, native, false)
	if err != nil {
		t.Fatal(err)
	}
	if wire, ok := cell.baseWire["default"]; !ok || wire.kind != kindObject || len(wire.members) != 0 {
		t.Fatalf("the default base must write no wire, got %v", wire)
	}
	steers := map[string]int{}
	for id := range cell.controls {
		control, _ := reg.Control(id)
		if control.Instruction != nil {
			steers[control.Native]++
		}
	}
	if len(steers) != 1 || steers["xhigh"] != 3 {
		t.Fatalf("steer3 must ride xhigh alone, got %v", steers)
	}
	if _, err := reg.Admit(model, "openrouter", FormatMessages, Control{Native: "xhigh"}, false); err != nil {
		t.Fatalf("the native xhigh control: %v", err)
	}
}
