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
	"fmt"
	"sort"
	"strconv"
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
	for index, item := range state.Ledger {
		// The placer appends at a later tail each time, so anchors strictly
		// increase; only then is the last item the one in force.
		if index > 0 && item.Anchor <= state.Ledger[index-1].Anchor {
			return nil, malformed
		}
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
		(control.Instruction == nil) != (first.Instruction == nil) ||
		(control.Instruction != nil && !sameRules(control.Instruction, first.Instruction)) {
		return refuse("call %d: base, budget, lever presence and rules are fixed per episode", index)
	}
	return nil
}

// sameRules reports whether two instructions name the same rule set: the
// text rules every instruction carries, plus its unit and in-band system rule.
func sameRules(a, b *Instruction) bool {
	return a.Unit == b.Unit && a.InbandSystem == b.InbandSystem
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
	rendered, err := renderedUnits(units, next.Ledger, format)
	if err != nil {
		return nil, Receipt{}, err
	}
	var applied []*value
	if instruction != nil && instruction.InbandSystem == InbandSystemFold {
		if applied, err = foldInbandSystem(units, rendered, format); err != nil {
			return nil, Receipt{}, err
		}
	} else {
		for _, group := range rendered {
			applied = append(applied, group...)
		}
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
	// codex answers its freeform tools (apply_patch) with
	// custom_tool_call_output: a tool result like function_call_output.
	if isToolResult(unit, format) {
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

// renderedUnits is, per client unit, what the ledger renders for it: the
// unit (with any appended instruction), then any unit inserted after it.
func renderedUnits(units []*value, ledger []LedgerItem, format string) ([][]*value, error) {
	byAnchor := map[int][]LedgerItem{}
	for _, item := range ledger {
		byAnchor[item.Anchor] = append(byAnchor[item.Anchor], item)
	}
	out := make([][]*value, 0, len(units))
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
			out = append(out, []*value{placed})
			continue
		}
		group := []*value{unit.clone()}
		if len(items) == 1 {
			group = append(group, inserted(format, items[0].Text))
		}
		out = append(out, group)
	}
	return out, nil
}

// --- the in-band system fold (fold_before_unit_v1) -------------------------------

// isToolResult is a tool-result unit: a Chat tool message, or a Responses
// function or custom tool call output.
func isToolResult(unit *value, format string) bool {
	role, _ := stringField(unit, "role", "")
	typ, _ := stringField(unit, "type", "")
	return (format == FormatChat && unit.has("role") && role == "tool") ||
		(format == FormatResponses && unit.has("type") && (typ == "function_call_output" || typ == "custom_tool_call_output"))
}

// foldedUnits is an in-band system row's text as the text units it folds
// into: its own blocks, byte for byte.
func foldedUnits(row *value, format string) ([]*value, error) {
	var extra []string
	for _, m := range row.members {
		if m.key != "role" && m.key != "content" && m.key != "type" {
			extra = append(extra, m.key)
		}
	}
	if len(extra) > 0 {
		return nil, refuse("an in-band system message carries %s; only its content can be folded", pythonStrList(extra))
	}
	content := row.get("content")
	if content.isString() {
		return []*value{textUnit(format, content.str)}, nil
	}
	textType := textUnit(format, "").get("type").str
	refused := refuse("an in-band system message's content is not %s units; it cannot be folded", textType)
	if content == nil || content.kind != kindArray || len(content.items) == 0 {
		return nil, refused
	}
	out := make([]*value, 0, len(content.items))
	for _, block := range content.items {
		if block.kind != kindObject {
			return nil, refused
		}
		if t := block.get("type"); !t.isString() || t.str != textType {
			return nil, refused
		}
		if !block.get("text").isString() {
			return nil, refused
		}
		out = append(out, block.clone())
	}
	return out, nil
}

// pythonStrList is Python's repr of a sorted list of strings, as the
// reference renderer's refusal text writes it.
func pythonStrList(keys []string) string {
	sorted := append([]string{}, keys...)
	sort.Strings(sorted)
	quoted := make([]string, len(sorted))
	for i, key := range sorted {
		quoted[i] = pythonStrRepr(key)
	}
	return "[" + strings.Join(quoted, ", ") + "]"
}

// pythonStrRepr is Python's repr of a printable string: single-quoted
// unless it holds a single quote and no double quote.
func pythonStrRepr(s string) string {
	quote := "'"
	if strings.Contains(s, "'") && !strings.Contains(s, `"`) {
		quote = `"`
	}
	var b strings.Builder
	b.WriteString(quote)
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == '\'' && quote == "'":
			b.WriteString(`\'`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r < 0x20 || r == 0x7f || (r >= 0x80 && r < 0xa0):
			fmt.Fprintf(&b, `\x%02x`, r)
		case !pythonPrintable(r):
			switch {
			case r < 0x100:
				fmt.Fprintf(&b, `\x%02x`, r)
			case r < 0x10000:
				fmt.Fprintf(&b, `\u%04x`, r)
			default:
				fmt.Fprintf(&b, `\U%08x`, r)
			}
		default:
			b.WriteRune(r)
		}
	}
	b.WriteString(quote)
	return b.String()
}

// pythonPrintable approximates str.isprintable for one character: Unicode
// letters, marks, numbers, punctuation, symbols and the ASCII space.
func pythonPrintable(r rune) bool {
	return r == ' ' || unicode.In(r, unicode.L, unicode.M, unicode.N, unicode.P, unicode.S)
}

// foldIntoMessage is placed with folded after the client's own content and
// before anything placement appended.
func foldIntoMessage(client, placed *value, format string, folded []*value) (*value, error) {
	out := placed.clone()
	own, content := client.get("content"), placed.get("content")
	if own.isString() && content.isString() {
		// The client's string, extended by a joined steering suffix or not at
		// all: the client's text, the folded units, then the suffix (less the
		// blank-line join) as its own unit.
		if !strings.HasPrefix(content.str, own.str) ||
			(content.str != own.str && !strings.HasPrefix(content.str[len(own.str):], "\n\n")) {
			return nil, refuse("the placed message does not extend the client's own text")
		}
		parts := append([]*value{textUnit(format, own.str)}, folded...)
		if content.str != own.str {
			parts = append(parts, textUnit(format, content.str[len(own.str)+2:]))
		}
		out.set("content", arrayValue(parts...))
		return out, nil
	}
	ownCount := -1
	switch {
	case own.isString():
		ownCount = 1
	case own != nil && own.kind == kindArray:
		ownCount = len(own.items)
	}
	if ownCount < 0 || content == nil || content.kind != kindArray || len(content.items) < ownCount {
		return nil, refuse("the governed %s message has no string or list content to fold into", format)
	}
	parts := append([]*value{}, content.items[:ownCount]...)
	parts = append(parts, folded...)
	parts = append(parts, content.items[ownCount:]...)
	out.set("content", arrayValue(parts...))
	return out, nil
}

// foldInbandSystem is the fold_before_unit_v1 rule over one call's placed
// units: each in-band system or developer row moves into the nearest
// preceding governed turn (a user message, or a tool-result run), after that
// turn's own content and before any steering unit, and is removed where it
// stood. Rows before the first governed turn stay. A row after an assistant
// unit, or inside a tool-result run, is refused. Deterministic in its inputs,
// so a call extending the previous call's units renders them unchanged.
func foldInbandSystem(units []*value, rendered [][]*value, format string) ([]*value, error) {
	if len(rendered) != len(units) {
		return nil, refuse("the rendered units do not align with the client's")
	}
	for _, group := range rendered {
		if len(group) == 0 {
			return nil, refuse("the rendered units do not align with the client's")
		}
	}
	folded := map[int][]*value{}
	dropped := map[int]bool{}
	target, previous := -1, -1
	for position, unit := range units {
		if isSystem(unit) {
			if target < 0 {
				continue // before the first governed turn: stays where it is
			}
			if previous != target {
				return nil, refuse("in-band system unit %d follows unit %s, not a governed turn; "+
					"folding it would rewrite an answered turn", position, pythonOptionalInt(previous))
			}
			var following *value
			for _, later := range units[position+1:] {
				if !isSystem(later) {
					following = later
					break
				}
			}
			if isToolResult(units[target], format) && following != nil && isToolResult(following, format) {
				return nil, refuse("in-band system unit %d sits inside a tool-result run; "+
					"folding it would split the run", position)
			}
			if len(rendered[position]) != 1 {
				return nil, refuse("placement wrote at in-band system unit %d", position)
			}
			rows, err := foldedUnits(unit, format)
			if err != nil {
				return nil, err
			}
			folded[target] = append(folded[target], rows...)
			dropped[position] = true
			continue
		}
		if isUserMessage(unit, format) || isToolResult(unit, format) {
			target = position
		}
		previous = position
	}
	out := make([]*value, 0, len(units)+len(rendered))
	for position, group := range rendered {
		if dropped[position] {
			continue
		}
		head := group[0].clone()
		insertedUnits := make([]*value, 0, len(group)-1)
		for _, unit := range group[1:] {
			insertedUnits = append(insertedUnits, unit.clone())
		}
		rows, isTarget := folded[position]
		switch {
		case isTarget && isUserMessage(units[position], format):
			merged, err := foldIntoMessage(units[position], head, format, rows)
			if err != nil {
				return nil, err
			}
			out = append(out, merged)
			out = append(out, insertedUnits...)
		case isTarget:
			// After a tool run: the folded text is a user message of its own,
			// ahead of any instruction placement inserted after the run.
			message := inserted(format, "")
			message.set("content", arrayValue(rows...))
			out = append(out, head, message)
			out = append(out, insertedUnits...)
		default:
			out = append(out, head)
			out = append(out, insertedUnits...)
		}
	}
	return out, nil
}

// pythonOptionalInt is how the reference renderer's refusal text writes an
// optional index: the number, or None.
func pythonOptionalInt(n int) string {
	if n < 0 {
		return "None"
	}
	return strconv.Itoa(n)
}
