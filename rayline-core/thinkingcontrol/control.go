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

// Package thinkingcontrol is VSR's implementation of pathfinder's thinking
// controls (ADR 0109): the compiled registry, control identity, admission per
// (model, provider, format), and the turn_tail_v2 / on_change_v1 / ledger_v1
// placement rules that render a control onto a client request body.
//
// It is a port of pathfinder's reference renderer
// (serving/thinking_control_render.py over sandbox/thinking_control_placement.py)
// and is held to its golden corpora byte for byte (testdata/golden). It has
// no dependency on the ARC runtime.
package thinkingcontrol

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// InstructionNone is the instruction level that means no steering
// instruction in force.
const InstructionNone = "none"

// Rules are the text rules, which every other rule set extends or overrides.
// An instruction's rules are exactly one of pathfinder's
// INSTRUCTION_RULE_SETS (Instruction.rules); the compiled registry refuses
// any other.
var Rules = map[string]string{"placement": "turn_tail_v2", "emit": "on_change_v1", "replay": "ledger_v1"}

// UnitMessagesEffortItem is pathfinder's Messages mid-conversation effort
// item unit (ADR 0143). Its rule set is the text rules with the unit and
// placement PlacementAfterNewInput (ADR 0147). VSR reads controls that use
// it, so that the registry and their ids load, and refuses to admit or place
// them: it does not implement the unit.
const UnitMessagesEffortItem = "messages_effort_item_v1"

// PlacementAfterNewInput is the placement the effort item's rule set names
// in place of turn_tail_v2 (ADR 0147).
const PlacementAfterNewInput = "after_new_input_v1"

// EmitEveryTurn is pathfinder's every_turn_v1 emit rule: the text rules with
// a steered level re-written at every governed turn. VSR reads controls that
// use it and refuses to admit or place them: it does not implement the rule.
const EmitEveryTurn = "every_turn_v1"

// UnitConfigurationUpdate is an instruction unit besides text that a
// rule set may name (pathfinder ADR 0114): a Responses configuration_update
// input item that sets the reasoning effort. VSR reads controls that use it,
// so that the registry and their ids load, and refuses to admit them: per
// the ADR it does not implement the unit until it serves a Responses cell.
const UnitConfigurationUpdate = "configuration_update_v1"

// InbandSystemFold is the in-band system rule a rule set may name under the
// key "inband_system" (pathfinder's FOLD_RULES): each in-band system or
// developer message is folded into the nearest preceding governed turn,
// after its own content and before any steering unit (foldInbandSystem). It
// is a rule set of its own, so a fold control has its own id; it never
// combines with a unit.
const InbandSystemFold = "fold_before_unit_v1"

// Control is what an action names: a per-session base, an optional token
// budget, and an optional steering instruction. It carries no provider,
// format or wire bytes.
type Control struct {
	Native       string
	BudgetTokens *int64
	Instruction  *Instruction
}

// Instruction is a control's steering instruction.
type Instruction struct {
	Level       string
	Text        string
	NeutralText string
	// Unit is the instruction's unit when it is not text: empty,
	// UnitConfigurationUpdate or UnitMessagesEffortItem.
	Unit string
	// InbandSystem is the in-band system rule: empty, or InbandSystemFold.
	InbandSystem string
	// Emit is the emit rule when it is not on_change_v1: empty, or
	// EmitEveryTurn.
	Emit string
}

// rules is the instruction's rule set as its control object spells it.
func (in *Instruction) rules() map[string]string {
	rules := make(map[string]string, len(Rules)+1)
	for key, val := range Rules {
		rules[key] = val
	}
	if in.Unit != "" {
		rules["unit"] = in.Unit
	}
	if in.Unit == UnitMessagesEffortItem {
		rules["placement"] = PlacementAfterNewInput
	}
	if in.InbandSystem != "" {
		rules["inband_system"] = in.InbandSystem
	}
	if in.Emit != "" {
		rules["emit"] = in.Emit
	}
	return rules
}

// Implemented reports whether VSR places this instruction's rule set: the
// text rules, alone or with the in-band system fold. The configuration_update
// unit (ADR 0114), the Messages effort item (ADR 0143, 0147) and the
// every_turn_v1 emit rule load, and are refused at admission and placement.
func (in *Instruction) Implemented() bool {
	return in.Unit == "" && in.Emit == ""
}

// Error is a control, artifact, admission or placement that cannot be used
// without guessing.
type Error struct{ msg string }

func (e *Error) Error() string { return e.msg }

func refuse(format string, args ...any) error { return &Error{msg: fmt.Sprintf(format, args...)} }

// json is the control as a JSON object with all three members present.
func (c Control) json() *value {
	budget := nullValue()
	if c.BudgetTokens != nil {
		budget = intValue(*c.BudgetTokens)
	}
	instruction := nullValue()
	if c.Instruction != nil {
		rules := objectValue()
		// jcs sorts members, so the map's order does not reach the id.
		for key, val := range c.Instruction.rules() {
			rules.set(key, stringValue(val))
		}
		instruction = objectValue(
			member{"level", stringValue(c.Instruction.Level)},
			member{"text", stringValue(c.Instruction.Text)},
			member{"neutral_text", stringValue(c.Instruction.NeutralText)},
			member{"rules", rules},
		)
	}
	return objectValue(
		member{"base", objectValue(member{"native", stringValue(c.Native)})},
		member{"budget_tokens", budget},
		member{"instruction", instruction},
	)
}

// Canonical is the control's RFC 8785 JSON, the form ParseControl reads and
// ID hashes.
func (c Control) Canonical() ([]byte, error) { return jcs(c.json()) }

// ID is sha256 of the control's RFC 8785 form, every string as raw UTF-8.
func (c Control) ID() string {
	canonical, err := jcs(c.json())
	if err != nil {
		// A budget beyond 2^53 is refused when the control is read; a
		// constructed one is a programming error.
		panic(err)
	}
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:])
}

// ParseControl reads a control from JSON with exactly the keys and types a
// control has, or refuses it. The manifest of a v5 package and the compiled
// registry both carry controls in this form.
func ParseControl(raw []byte) (Control, error) {
	v, err := parseJSON(raw)
	if err != nil {
		return Control{}, refuse("not a thinking control: %v", err)
	}
	return requireControl(v, "control")
}

// ControlIDOf is sha256 of the RFC 8785 form of a control exactly as raw
// holds it. A v5 manifest's control_id is this over its control object; it
// equals Control.ID for any control ParseControl accepts.
func ControlIDOf(raw []byte) (string, error) {
	v, err := parseJSON(raw)
	if err != nil {
		return "", err
	}
	canonical, err := jcs(v)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:]), nil
}

func requireControl(v *value, where string) (Control, error) {
	bad := refuse("%s: not a thinking control {base: {native}, budget_tokens, instruction}", where)
	if v == nil || v.kind != kindObject || !hasExactly(v, "base", "budget_tokens", "instruction") {
		return Control{}, bad
	}
	base := v.get("base")
	if base.kind != kindObject || !hasExactly(base, "native") || !base.get("native").isString() {
		return Control{}, bad
	}
	control := Control{Native: base.get("native").str}
	switch budget := v.get("budget_tokens"); budget.kind {
	case kindNull:
	case kindInt:
		if budget.integer.Sign() <= 0 || !budget.integer.IsInt64() || budget.integer.Cmp(maxExactInt) > 0 {
			return Control{}, bad
		}
		n := budget.integer.Int64()
		control.BudgetTokens = &n
	default:
		return Control{}, bad
	}
	switch instruction := v.get("instruction"); instruction.kind {
	case kindNull:
	case kindObject:
		if !hasExactly(instruction, "level", "text", "neutral_text", "rules") {
			return Control{}, bad
		}
		for _, key := range []string{"level", "text", "neutral_text"} {
			if !instruction.get(key).isString() {
				return Control{}, bad
			}
		}
		rules := instruction.get("rules")
		if rules.kind != kindObject {
			return Control{}, bad
		}
		// The rules are exactly one of pathfinder's INSTRUCTION_RULE_SETS:
		// the text rules, alone or with one of the configuration_update unit,
		// the in-band system fold or the every_turn_v1 emit rule; or the
		// effort item's set, with its own unit and placement.
		parsed := &Instruction{}
		if got := rules.get("unit"); got != nil {
			if !got.isString() || (got.str != UnitConfigurationUpdate && got.str != UnitMessagesEffortItem) {
				return Control{}, bad
			}
			parsed.Unit = got.str
		}
		if got := rules.get("inband_system"); got != nil {
			if !got.isString() || got.str != InbandSystemFold || parsed.Unit != "" {
				return Control{}, bad
			}
			parsed.InbandSystem = InbandSystemFold
		}
		if got := rules.get("emit"); got != nil && got.isString() && got.str == EmitEveryTurn {
			if parsed.Unit != "" || parsed.InbandSystem != "" {
				return Control{}, bad
			}
			parsed.Emit = EmitEveryTurn
		}
		want := parsed.rules()
		if len(rules.members) != len(want) {
			return Control{}, bad
		}
		for key, val := range want {
			if got := rules.get(key); !got.isString() || got.str != val {
				return Control{}, bad
			}
		}
		control.Instruction = &Instruction{
			Level:        instruction.get("level").str,
			Text:         instruction.get("text").str,
			NeutralText:  instruction.get("neutral_text").str,
			Unit:         parsed.Unit,
			InbandSystem: parsed.InbandSystem,
			Emit:         parsed.Emit,
		}
	default:
		return Control{}, bad
	}
	return control, nil
}

func hasExactly(obj *value, keys ...string) bool {
	if len(obj.members) != len(keys) {
		return false
	}
	for _, key := range keys {
		if !obj.has(key) {
			return false
		}
	}
	return true
}
