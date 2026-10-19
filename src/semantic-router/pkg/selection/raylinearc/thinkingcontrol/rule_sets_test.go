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

import "testing"

// The Messages effort item (pathfinder ADR 0143, placed after_new_input_v1
// per ADR 0147) and the every_turn_v1 emit rule are rule sets of their own.
// VSR reads them so the registry and pathfinder's ids load, and refuses them
// at admission and placement, where rendering them as text would write the
// wrong bytes at the wrong turns.
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
