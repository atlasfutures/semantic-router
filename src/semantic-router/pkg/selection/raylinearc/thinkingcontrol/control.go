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

// Rules are the placement rules every instruction names; the compiled
// registry refuses any other.
var Rules = map[string]string{"placement": "turn_tail_v2", "emit": "on_change_v1", "replay": "ledger_v1"}

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
	// Unit is the instruction's rules.unit when the registry names one
	// (pathfinder's configuration_update_v1: a Responses item setting the
	// native effort). This router renders text instructions only, so a
	// control with a unit is read, kept under its own id, and never admitted
	// (Registry.Admit) or rendered.
	Unit string
}

// UnitRule is the one optional rules member; its value names a rendering
// unit other than the text instruction.
const UnitRule = "unit"

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
		for _, key := range []string{"placement", "emit", "replay"} {
			rules.set(key, stringValue(Rules[key]))
		}
		if c.Instruction.Unit != "" {
			rules.set(UnitRule, stringValue(c.Instruction.Unit))
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
		unit := rules.get(UnitRule)
		if want := len(Rules); len(rules.members) != want && !(unit != nil && len(rules.members) == want+1) {
			return Control{}, bad
		}
		for key, want := range Rules {
			if got := rules.get(key); !got.isString() || got.str != want {
				return Control{}, bad
			}
		}
		if unit != nil && (!unit.isString() || unit.str == "") {
			return Control{}, bad
		}
		control.Instruction = &Instruction{
			Level:       instruction.get("level").str,
			Text:        instruction.get("text").str,
			NeutralText: instruction.get("neutral_text").str,
		}
		if unit != nil {
			control.Instruction.Unit = unit.str
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
