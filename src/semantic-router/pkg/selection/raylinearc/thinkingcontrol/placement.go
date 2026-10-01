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
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"unicode"
)

// Placements of an instruction relative to its anchor unit.
const (
	PlacementAppend      = "append"
	PlacementInsertAfter = "insert_after"
)

// EpochResetTranscriptRewrite is the reason an epoch restarts when a ledger
// item's anchor no longer matches the client's transcript.
const EpochResetTranscriptRewrite = "transcript_rewrite"

// Instruction states: which instruction the model can see.
const (
	StateNever         = "never"
	StateNeutralMarker = "neutral_marker"
	StateSteered       = "steered"
)

// What a call added to the ledger.
const (
	WrittenInstruction   = "instruction"
	WrittenNeutralMarker = "neutral_marker"
)

// Receipt is what one call's placement did; the field names are the golden
// corpora's.
type Receipt struct {
	Call      int    `json:"call"`
	ControlID string `json:"control_id"`
	// LevelInForce is this call's instruction level; nil for a native-only
	// episode.
	LevelInForce     *string `json:"level_in_force"`
	InstructionState *string `json:"instruction_state"`
	// Written is "instruction", "neutral_marker", or nil when nothing was
	// written.
	Written          *string `json:"written"`
	Epoch            int     `json:"epoch"`
	EpochResetReason *string `json:"epoch_reset_reason"`
	Retry            bool    `json:"retry"`
}

// LedgerItem is one written instruction: where it is anchored, and its bytes.
type LedgerItem struct {
	Anchor       int    `json:"anchor"`
	PrefixDigest string `json:"prefix_digest"`
	Placement    string `json:"placement"`
	Text         string `json:"text"`
	Kind         string `json:"kind"`
}

// Anchor is the governed tail a body is placed at: its unit index and the
// digest of the client units through it.
type Anchor struct {
	Index  int    `json:"index"`
	Digest string `json:"digest"`
}

// PlacerState is everything Place reads, so a placer resumes the same
// episode in another process; it is what the episode store keeps.
type PlacerState struct {
	Format          string       `json:"format"`
	First           *Control     `json:"first,omitempty"`
	Ledger          []LedgerItem `json:"ledger"`
	Epoch           int          `json:"epoch"`
	InForce         *string      `json:"in_force"`
	PreviousAnchor  *Anchor      `json:"previous_anchor"`
	PreviousControl *string      `json:"previous_control"`
	Calls           int          `json:"calls"`
}

// Placer is one episode's placement state on one wire format, advanced one
// call at a time. Base, budget and lever presence are fixed for the episode.
//
// A refused call leaves the state as it was. (Pathfinder's placer keeps an
// epoch reset it made before refusing; no corpus case reaches that.)
type Placer struct {
	state PlacerState
}

// NewPlacer starts an episode on a wire format.
func NewPlacer(format string) (*Placer, error) {
	if !knownFormat(format) {
		return nil, refuse("unknown wire format '%s'", format)
	}
	return &Placer{state: PlacerState{Format: format, Ledger: []LedgerItem{}}}, nil
}

// ResumePlacer continues the episode a state describes; a malformed state is
// refused, never repaired.
func ResumePlacer(state PlacerState) (*Placer, error) {
	malformed := refuse("the placement state is not one a placer wrote")
	if !knownFormat(state.Format) || state.Epoch < 0 || state.Calls < 0 {
		return nil, malformed
	}
	// The first call fixes the episode's base, budget and lever presence, and
	// every call records its anchor and control; a state that has calls
	// without them, or them without calls, would let the next call re-found
	// the episode.
	if state.Calls == 0 {
		if state.First != nil || len(state.Ledger) > 0 || state.Epoch != 0 || state.InForce != nil ||
			state.PreviousAnchor != nil || state.PreviousControl != nil {
			return nil, malformed
		}
	} else if state.First == nil || state.First.Native == "" || state.PreviousAnchor == nil ||
		state.PreviousControl == nil {
		return nil, malformed
	}
	// Only a lever episode writes instructions; a ledger or a level in force
	// on a native-only episode would put text on its provider requests.
	if (state.InForce != nil || len(state.Ledger) > 0) && (state.First == nil || state.First.Instruction == nil) {
		return nil, malformed
	}
	// A steered level is in force exactly when the last written item is an
	// instruction: a hold writes nothing, and a return to none writes the
	// marker and clears it. Anything else would report a steer the body
	// does not carry, or carry one the receipt does not report.
	lastIsInstruction := len(state.Ledger) > 0 && state.Ledger[len(state.Ledger)-1].Kind == WrittenInstruction
	if (state.InForce != nil) != lastIsInstruction {
		return nil, malformed
	}
	for _, item := range state.Ledger {
		if item.Anchor < 0 || (item.Placement != PlacementAppend && item.Placement != PlacementInsertAfter) ||
			(item.Kind != WrittenInstruction && item.Kind != WrittenNeutralMarker) {
			return nil, malformed
		}
	}
	state.Ledger = append([]LedgerItem{}, state.Ledger...)
	return &Placer{state: state}, nil
}

// State is the placer's state, for the episode store.
func (p *Placer) State() PlacerState {
	out := p.state
	out.Ledger = append([]LedgerItem{}, p.state.Ledger...)
	return out
}

func knownFormat(format string) bool {
	return format == FormatMessages || format == FormatChat || format == FormatResponses
}

// check refuses a missing control, or one that changes the episode's base,
// budget or lever presence.
func (p *Placer) check(control *Control) error {
	index := p.state.Calls
	if control == nil {
		return refuse("call %d: no control; absence is not the 'none' level", index)
	}
	first := p.state.First
	if first == nil {
		first = control
	}
	if control.Native != first.Native || !sameBudget(control.BudgetTokens, first.BudgetTokens) ||
		(control.Instruction == nil) != (first.Instruction == nil) {
		return refuse("call %d: base, budget and lever presence are fixed per episode", index)
	}
	return nil
}

func sameBudget(a, b *int64) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// place applies the ledger to body for this call's control and returns the
// body (thinking fields and model untouched) and the call's receipt. body is
// not modified.
func (p *Placer) place(body *value, control *Control) (*value, Receipt, error) {
	if err := p.check(control); err != nil {
		return nil, Receipt{}, err
	}
	next := p.State()
	index := next.Calls
	format := next.Format
	if next.First == nil {
		first := *control
		next.First = &first
	}
	units, err := unitsOf(body, format)
	if err != nil {
		return nil, Receipt{}, err
	}
	tail, err := governedTail(units, format)
	if err != nil {
		return nil, Receipt{}, err
	}
	anchor := Anchor{Index: tail, Digest: prefixDigest(units, tail)}
	id := control.ID()
	retry := next.PreviousAnchor != nil && *next.PreviousAnchor == anchor
	if retry && (next.PreviousControl == nil || *next.PreviousControl != id) {
		return nil, Receipt{}, refuse("call %d: a retried turn carries the control it was first sent with", index)
	}
	var reset *string
	for _, item := range next.Ledger {
		if item.Anchor >= len(units) || prefixDigest(units, item.Anchor) != item.PrefixDigest {
			next.Ledger, next.Epoch, next.InForce = []LedgerItem{}, next.Epoch+1, nil
			reason := EpochResetTranscriptRewrite
			reset = &reason
			break
		}
	}
	var written *string
	instruction := control.Instruction
	if instruction != nil && !retry {
		level := instruction.Level
		write := func(kind, text string) error {
			placement, err := placementFor(units[tail], format)
			if err != nil {
				return err
			}
			next.Ledger = append(next.Ledger, LedgerItem{
				Anchor: anchor.Index, PrefixDigest: anchor.Digest, Placement: placement, Text: text, Kind: kind,
			})
			written = &kind
			return nil
		}
		switch {
		case level != InstructionNone && (next.InForce == nil || *next.InForce != level):
			if err := write(WrittenInstruction, instruction.Text); err != nil {
				return nil, Receipt{}, err
			}
		case level == InstructionNone && next.InForce != nil:
			if err := write(WrittenNeutralMarker, instruction.NeutralText); err != nil {
				return nil, Receipt{}, err
			}
		}
		if level == InstructionNone {
			next.InForce = nil
		} else {
			inForce := level
			next.InForce = &inForce
		}
	}
	applied, err := applyLedger(units, next.Ledger, format)
	if err != nil {
		return nil, Receipt{}, err
	}
	placed := body.clone()
	placed.set(unitsKey(format), arrayValue(applied...))
	receipt := Receipt{Call: index, ControlID: id, Written: written, Epoch: next.Epoch, EpochResetReason: reset, Retry: retry}
	if instruction != nil {
		level := instruction.Level
		receipt.LevelInForce = &level
		state := StateNever
		switch {
		case next.InForce != nil:
			state = StateSteered
		case len(next.Ledger) > 0 && next.Ledger[len(next.Ledger)-1].Kind == WrittenNeutralMarker:
			state = StateNeutralMarker
		}
		receipt.InstructionState = &state
	}
	next.PreviousAnchor, next.PreviousControl = &anchor, &id
	next.Calls++
	p.state = next
	return placed, receipt, nil
}

// --- the anchor projection ---------------------------------------------------

const (
	ambientOpen  = "<system-reminder>"
	ambientClose = "</system-reminder>"
)

// isAmbientReminderText is one whole injected Claude Code reminder block,
// and nothing else.
func isAmbientReminderText(block *value) bool {
	if block == nil || block.kind != kindObject {
		return false
	}
	if t := block.get("type"); !t.isString() || t.str != "text" {
		return false
	}
	text := block.get("text")
	if !text.isString() {
		return false
	}
	stripped := pythonStrip(text.str)
	return strings.HasPrefix(stripped, ambientOpen) && strings.HasSuffix(stripped, ambientClose) &&
		strings.Count(stripped, ambientOpen) == 1 && strings.Count(stripped, ambientClose) == 1
}

// pythonStrip is Python's str.strip(): it trims what str.isspace() accepts,
// which is Go's unicode.IsSpace plus the four ASCII separators.
func pythonStrip(s string) string {
	return strings.TrimFunc(s, func(r rune) bool {
		return (r >= 0x1c && r <= 0x1f) || unicode.IsSpace(r)
	})
}

// projectedMessage is one message's conversation content minus a harness's
// per-request churn: whole injected <system-reminder> blocks, cache_control
// hints, and the stream-only caller {"type": "direct"} annotation.
func projectedMessage(message *value) *value {
	if message == nil || message.kind != kindObject {
		return message
	}
	direct := objectValue(member{"type", stringValue("direct")})
	projected := objectValue()
	for _, m := range message.members {
		if m.key == "cache_control" {
			continue
		}
		if m.key == "content" && m.val.kind == kindArray {
			blocks := arrayValue()
			for _, block := range m.val.items {
				if isAmbientReminderText(block) {
					continue
				}
				if block.kind == kindObject {
					cleaned := objectValue()
					for _, field := range block.members {
						if field.key == "cache_control" || (field.key == "caller" && field.val.equal(direct)) {
							continue
						}
						cleaned.set(field.key, field.val)
					}
					blocks.items = append(blocks.items, cleaned)
				} else {
					blocks.items = append(blocks.items, block)
				}
			}
			projected.set(m.key, blocks)
			continue
		}
		projected.set(m.key, m.val)
	}
	return projected
}

// prefixDigest is the anchor digest: client units 0..through, minus a
// harness's per-request churn.
func prefixDigest(units []*value, through int) string {
	projected := arrayValue()
	for _, unit := range units[:through+1] {
		projected.items = append(projected.items, projectedMessage(unit))
	}
	canonical, err := dumpSorted(projected)
	if err != nil {
		// Units come from a parsed body, which holds no non-finite number.
		panic(err)
	}
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:])
}

// --- per-format units -----------------------------------------------------------

func unitsKey(format string) string {
	if format == FormatResponses {
		return "input"
	}
	return "messages"
}

func unitsOf(body *value, format string) ([]*value, error) {
	key := unitsKey(format)
	units := body.get(key)
	if units == nil || units.kind != kindArray || len(units.items) == 0 {
		return nil, refuse("the %s body has no %s list of objects to govern", format, key)
	}
	for _, unit := range units.items {
		if unit.kind != kindObject {
			return nil, refuse("the %s body has no %s list of objects to govern", format, key)
		}
	}
	return units.items, nil
}

func stringField(unit *value, key, fallback string) (string, bool) {
	v := unit.get(key)
	if v == nil {
		return fallback, true
	}
	if !v.isString() {
		return "", false
	}
	return v.str, true
}

func isSystem(unit *value) bool {
	role, ok := stringField(unit, "role", "")
	typ, typed := stringField(unit, "type", "message")
	return ok && (role == "system" || role == "developer") && unit.has("role") && typed && typ == "message"
}

// governedTail is the last unit before any trailing in-band system messages.
func governedTail(units []*value, format string) (int, error) {
	position := len(units) - 1
	for position >= 0 && isSystem(units[position]) {
		position--
	}
	if position < 0 {
		return 0, refuse("the %s body has no conversational unit to govern", format)
	}
	return position, nil
}

func isUserMessage(unit *value, format string) bool {
	role, ok := stringField(unit, "role", "")
	if !ok || !unit.has("role") || role != "user" {
		return false
	}
	if format != FormatResponses {
		return true
	}
	typ, typed := stringField(unit, "type", "message")
	return typed && typ == "message"
}

// placementFor is where this turn's instruction goes, relative to the tail
// client unit.
func placementFor(unit *value, format string) (string, error) {
	if isUserMessage(unit, format) {
		return PlacementAppend, nil
	}
	role, _ := stringField(unit, "role", "")
	typ, _ := stringField(unit, "type", "")
	if (format == FormatChat && unit.has("role") && role == "tool") ||
		(format == FormatResponses && unit.has("type") && typ == "function_call_output") {
		return PlacementInsertAfter, nil
	}
	return "", refuse("the %s tail is not a governed turn's input (a user message or tool results)", format)
}

func textUnit(format, text string) *value {
	typ := "text"
	if format == FormatResponses {
		typ = "input_text"
	}
	return objectValue(member{"type", stringValue(typ)}, member{"text", stringValue(text)})
}

func appended(unit *value, format, text string) (*value, error) {
	out := unit.clone()
	content := out.get("content")
	var parts []*value
	switch {
	case content.isString():
		parts = []*value{textUnit(format, content.str)}
	case content != nil && content.kind == kindArray:
		parts = append([]*value{}, content.items...)
	default:
		return nil, refuse("the tail %s message has no string or list content to append to", format)
	}
	out.set("content", arrayValue(append(parts, textUnit(format, text))...))
	return out, nil
}

func inserted(format, text string) *value {
	if format == FormatResponses {
		return objectValue(
			member{"type", stringValue("message")},
			member{"role", stringValue("user")},
			member{"content", arrayValue(textUnit(format, text))},
		)
	}
	return objectValue(member{"role", stringValue("user")}, member{"content", arrayValue(textUnit(format, text))})
}

func applyLedger(units []*value, ledger []LedgerItem, format string) ([]*value, error) {
	byAnchor := map[int][]LedgerItem{}
	for _, item := range ledger {
		byAnchor[item.Anchor] = append(byAnchor[item.Anchor], item)
	}
	out := make([]*value, 0, len(units)+len(ledger))
	for position, unit := range units {
		items := byAnchor[position]
		if len(items) > 1 {
			return nil, refuse("two instructions anchor at unit %d; one turn carries one", position)
		}
		if len(items) == 1 && items[0].Placement == PlacementAppend {
			placed, err := appended(unit, format, items[0].Text)
			if err != nil {
				return nil, err
			}
			out = append(out, placed)
			continue
		}
		out = append(out, unit.clone())
		if len(items) == 1 {
			out = append(out, inserted(format, items[0].Text))
		}
	}
	return out, nil
}
