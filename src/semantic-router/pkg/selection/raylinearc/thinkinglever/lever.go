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

// Package thinkinglever plans and replays the items a thinking lever writes
// into an ARC worker's provider-bound transcript. It has no dependency on the
// ARC runtime, so configuration validation and the episode store can share
// one definition of a binding and a ledger.
package thinkinglever

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// A thinking lever moves a worker's reasoning depth for one episode without
// changing the request-level reasoning fields, which every measured provider
// renders into the prompt-cache key. Both levers work the same way: the
// router inserts an item into the provider-bound transcript that the client
// never sees, so every later turn must put every earlier item back, byte for
// byte and at the same position, or the provider cache misses from there on.
// The level is abstract; the binding says which bytes realise it.
type Lever string

const (
	// LeverSteeringSuffix appends an instruction at the transcript
	// tail. Its text is advice the model may ignore.
	LeverSteeringSuffix Lever = "prompt_steering_suffix"
	// LeverPerTurnEffort inserts a content-less system message that
	// carries only a reasoning effort, which providers that document it
	// apply from that point on without invalidating the cached prefix.
	LeverPerTurnEffort Lever = "per_turn_effort"
)

// EmitMode says when a lever item is written.
type EmitMode string

const (
	// EmitEveryTurn states the level in force on every governed
	// turn, so a provider's persistence semantics cannot matter.
	EmitEveryTurn EmitMode = "every_turn"
	// EmitOnChange writes an item only when the level changes.
	EmitOnChange EmitMode = "on_change"
)

// Placement records where an item sits relative to the client
// message it was anchored to. Admission is per placement: a lever measured
// cache-safe at one position says nothing about another.
type Placement string

const (
	// PlaceAppendTailUserText appends a text part to the tail user
	// message.
	PlaceAppendTailUserText Placement = "append_tail_user_text"
	// PlaceUserAfterToolRun inserts a user message after the tail
	// run of tool results.
	PlaceUserAfterToolRun Placement = "insert_user_after_tool_run"
	// PlaceSystemBeforeTurn inserts the item before the tail user
	// message.
	PlaceSystemBeforeTurn Placement = "system_before_governed_turn"
	// PlaceSystemAfterToolRun inserts the item after the tail run of
	// tool results. Chat Completions requires tool messages to follow their
	// call directly, so nothing can go before them.
	PlaceSystemAfterToolRun Placement = "system_after_tool_run"
)

// ResetTranscriptRewrite is the only reason an epoch ends: the client
// rewrote its transcript, so the anchors of earlier items are gone.
const ResetTranscriptRewrite = "transcript_rewrite"

// Skip reasons explain a governed turn that wrote no item.
const (
	SkipTailNotSteerable     = "tail_not_steerable"
	SkipPlacementRefused     = "placement_not_admitted"
	SkipNeutralInexpressible = "neutral_inexpressible"
	SkipChangeTooSoon        = "change_too_soon"
	// SkipLedgerFull holds the level in force rather than drop items to make
	// room, which would edit history the client never rewrote.
	SkipLedgerFull = "ledger_full"
)

const (
	DigestBytes  = 16
	maxLevels    = 16
	MaxLevelName = 32
	// MaxLedgerLength and MaxPayloads keep a full ledger, with every payload
	// at MaxSuffixBytes, well inside the 64 KiB episode-state limit.
	MaxLedgerLength      = 384
	MaxPayloads          = 8
	MaxSuffixBytes       = 1024
	systemReminderPrefix = "<system-reminder>"
)

// Level is one rung. Exactly one of Suffix or Effort carries the
// level's bytes, matching the binding's lever. An empty Suffix is a level
// that writes nothing.
type Level struct {
	Name   string `json:"level"`
	Rank   int    `json:"rank"`
	Suffix string `json:"suffix,omitempty"`
	Effort string `json:"effort,omitempty"`
}

// Binding is the lever admitted for one worker, as compiled from the
// shared thinking-level registry. The router never invents bytes: every item
// it writes comes from a binding.
type Binding struct {
	Lever Lever    `json:"lever"`
	Emit  EmitMode `json:"emit"`
	// Neutral names the level that restores default depth. Empty means the
	// ladder has none, and a return to default cannot be expressed once an
	// instruction is in force.
	Neutral    string      `json:"neutral_level,omitempty"`
	Placements []Placement `json:"placements"`
	Levels     []Level     `json:"levels"`
}

// Validate refuses a binding the ledger could not replay deterministically.
func (binding Binding) Validate() error {
	switch binding.Lever {
	case LeverSteeringSuffix, LeverPerTurnEffort:
	default:
		return fmt.Errorf("unknown thinking lever %q", binding.Lever)
	}
	switch binding.Emit {
	case EmitEveryTurn, EmitOnChange:
	default:
		return fmt.Errorf("unknown thinking emit mode %q", binding.Emit)
	}
	if len(binding.Levels) < 2 || len(binding.Levels) > maxLevels {
		return fmt.Errorf("a thinking binding needs 2 to %d levels", maxLevels)
	}
	seen := make(map[string]bool, len(binding.Levels))
	for _, level := range binding.Levels {
		if err := binding.validateLevel(level); err != nil {
			return err
		}
		if seen[level.Name] {
			return fmt.Errorf("thinking level %q is declared twice", level.Name)
		}
		seen[level.Name] = true
	}
	if binding.Neutral != "" && !seen[binding.Neutral] {
		return fmt.Errorf("neutral thinking level %q is not declared", binding.Neutral)
	}
	return binding.validatePlacements()
}

func (binding Binding) validatePlacements() error {
	if len(binding.Placements) == 0 {
		return errors.New("a thinking binding admits at least one placement")
	}
	for _, placement := range binding.Placements {
		if !placementFitsLever(binding.Lever, placement) {
			return fmt.Errorf("placement %q does not fit lever %q", placement, binding.Lever)
		}
	}
	return nil
}

func placementFitsLever(lever Lever, placement Placement) bool {
	switch placement {
	case PlaceAppendTailUserText, PlaceUserAfterToolRun:
		return lever == LeverSteeringSuffix
	case PlaceSystemBeforeTurn, PlaceSystemAfterToolRun:
		return lever == LeverPerTurnEffort
	default:
		return false
	}
}

func (binding Binding) admits(placement Placement) bool {
	for _, admitted := range binding.Placements {
		if admitted == placement {
			return true
		}
	}
	return false
}

func (binding Binding) validateLevel(level Level) error {
	if strings.TrimSpace(level.Name) == "" || level.Name != strings.TrimSpace(level.Name) ||
		len(level.Name) > MaxLevelName {
		return fmt.Errorf("a thinking level needs a trimmed, nonblank name of at most %d bytes", MaxLevelName)
	}
	switch binding.Lever {
	case LeverSteeringSuffix:
		if level.Effort != "" {
			return fmt.Errorf("suffix level %q carries an effort", level.Name)
		}
		if level.Suffix != "" && strings.TrimSpace(level.Suffix) == "" {
			return fmt.Errorf("suffix level %q is blank but not empty", level.Name)
		}
		if len(level.Suffix) > MaxSuffixBytes {
			return fmt.Errorf("suffix level %q exceeds %d bytes", level.Name, MaxSuffixBytes)
		}
	case LeverPerTurnEffort:
		if level.Suffix != "" || !plainEffortName(level.Effort) {
			return fmt.Errorf("effort level %q needs a plain effort name and no suffix", level.Name)
		}
	}
	return nil
}

func plainEffortName(effort string) bool {
	if effort == "" {
		return false
	}
	for _, character := range effort {
		if (character < 'a' || character > 'z') && character != '_' {
			return false
		}
	}
	return true
}

// Level returns the named rung.
func (binding Binding) Level(name string) (Level, bool) {
	for _, level := range binding.Levels {
		if level.Name == name {
			return level, true
		}
	}
	return Level{}, false
}

// ControlSHA256 is the digest a trained artifact records for the action it
// chose, and the identity the router and the exporter must agree on.
func (binding Binding) ControlSHA256(level Level) string {
	return binding.payloadFor(level).ControlSHA256()
}

func (binding Binding) payloadFor(level Level) Payload {
	if binding.Lever == LeverPerTurnEffort {
		return Payload{Lever: binding.Lever, Effort: level.Effort}
	}
	return Payload{Lever: binding.Lever, Suffix: level.Suffix}
}

// Payload is the exact content of one item, independent of any binding. The
// ledger keeps payloads rather than level names so an item written under one
// binding replays byte for byte under the next: dropping or rewriting it
// would edit history the client never rewrote.
type Payload struct {
	Lever  Lever
	Suffix string
	Effort string
}

// ControlSHA256 hashes a text item's UTF-8 text, or the RFC 8785 canonical
// form of an effort item's value. For the one member an effort item carries,
// sorted compact encoding of a plain [a-z_] effort name is that form, and
// Validate refuses any other name.
func (payload Payload) ControlSHA256() string {
	bytes := []byte(payload.Suffix)
	if payload.Lever == LeverPerTurnEffort {
		bytes = []byte(`{"reasoning":{"effort":"` + payload.Effort + `"}}`)
	}
	sum := sha256.Sum256(bytes)
	return hex.EncodeToString(sum[:])
}

// writes reports whether a payload produces an item at all. A per-turn
// effort always does; a suffix with no text does not.
func (payload Payload) writes() bool {
	return payload.Lever == LeverPerTurnEffort || payload.Suffix != ""
}

// Ledger is the per-episode record of every item the router wrote. It is
// committed with the rest of the episode state, so an attempt that never
// reached a 2xx response leaves nothing behind.
type Ledger struct {
	Epoch    uint32
	Payloads []Payload
	Entries  []LedgerEntry
	// InForce is, per lever, the item last written; a worker only ever
	// sees its own lever's items, so each lever has its own level in force.
	InForce []LeverState
}

// LeverState is what one lever has in force in this epoch.
type LeverState struct {
	Lever          Lever
	Payload        int
	Level          string
	LastChangeTurn uint64
}

// LedgerEntry locates one item by the client message it is anchored to.
// Index and Digest refer to the client's transcript, which is the only one
// that survives between turns; Payload indexes Ledger.Payloads.
type LedgerEntry struct {
	Index     uint32
	Placement Placement
	Digest    string
	Payload   int
	Turn      uint64
}

// Clone returns a deep copy; a nil ledger clones to nil.
func (ledger *Ledger) Clone() *Ledger {
	if ledger == nil {
		return nil
	}
	cloned := *ledger
	cloned.Payloads = append([]Payload(nil), ledger.Payloads...)
	cloned.Entries = append([]LedgerEntry(nil), ledger.Entries...)
	cloned.InForce = append([]LeverState(nil), ledger.InForce...)
	return &cloned
}

func (ledger *Ledger) state(lever Lever) (LeverState, bool) {
	for _, state := range ledger.InForce {
		if state.Lever == lever {
			return state, true
		}
	}
	return LeverState{Lever: lever, Payload: -1}, false
}

func (ledger *Ledger) setState(next LeverState) {
	for index, state := range ledger.InForce {
		if state.Lever == next.Lever {
			ledger.InForce[index] = next
			return
		}
	}
	ledger.InForce = append(ledger.InForce, next)
}

func (ledger *Ledger) payloadIndex(payload Payload) int {
	for index, known := range ledger.Payloads {
		if known == payload {
			return index
		}
	}
	return -1
}

// ValidateLedger refuses a persisted ledger no planner could have written.
func ValidateLedger(ledger *Ledger) error {
	if ledger == nil {
		return nil
	}
	if len(ledger.Entries) > MaxLedgerLength || len(ledger.Payloads) > MaxPayloads ||
		len(ledger.InForce) > 2 {
		return errors.New("thinking ledger exceeds its limits")
	}
	for _, payload := range ledger.Payloads {
		if !validPayload(payload) {
			return errors.New("thinking ledger payload is invalid")
		}
	}
	previous := -1
	for _, entry := range ledger.Entries {
		if int(entry.Index) <= previous {
			return errors.New("thinking ledger anchors are not ascending")
		}
		previous = int(entry.Index)
		if entry.Payload < 0 || entry.Payload >= len(ledger.Payloads) ||
			len(entry.Digest) != DigestBytes*2 || !isLowerHex(entry.Digest) ||
			!placementFitsLever(ledger.Payloads[entry.Payload].Lever, entry.Placement) {
			return errors.New("thinking ledger entry is invalid")
		}
	}
	for _, state := range ledger.InForce {
		if state.Payload < -1 || state.Payload >= len(ledger.Payloads) ||
			len(state.Level) > MaxLevelName ||
			(state.Payload >= 0 && ledger.Payloads[state.Payload].Lever != state.Lever) {
			return errors.New("thinking ledger lever state is invalid")
		}
	}
	return nil
}

func validPayload(payload Payload) bool {
	switch payload.Lever {
	case LeverSteeringSuffix:
		return payload.Effort == "" && len(payload.Suffix) <= MaxSuffixBytes
	case LeverPerTurnEffort:
		return payload.Suffix == "" && plainEffortName(payload.Effort)
	default:
		return false
	}
}

func isLowerHex(value string) bool {
	for _, character := range value {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

// Turn is everything the planner needs for one request.
type Turn struct {
	Binding Binding
	Ledger  *Ledger
	// Messages is the client's transcript, before any lever item.
	Messages  []Message
	TurnIndex uint64
	// Requested is the level the policy asks for on this turn.
	Requested              string
	MinTurnsBetweenChanges uint64
	// MaxEntries caps the ledger below MaxLedgerLength; zero means the
	// maximum.
	MaxEntries int
}

// Message is the part of a neutral message the planner reads: its role,
// and a digest that ignores what clients rewrite between turns.
type Message struct {
	Role   string
	Digest string
}

// Plan is the planner's decision. Next is staged; the caller commits it
// only with the episode state.
type Plan struct {
	Next Ledger
	// LevelInForce and ControlInForce describe the binding's lever after
	// this turn: what the worker is actually being asked, which is not
	// always what the policy requested.
	LevelInForce   string
	ControlInForce string
	Emitted        bool
	Retry          bool
	Placement      Placement
	ResetReason    string
	Skipped        string
	Replayed       int
}

// PlanTurn verifies the ledger against the client transcript, decides
// whether this turn writes an item, and returns the staged ledger.
//
// Only a client rewrite -- an anchor that no longer matches, as after
// compaction or /clear -- starts a new epoch, and the items it drops sat in
// messages the client itself removed. Nothing else ever removes an item: a
// changed binding appends, and a full ledger stops writing.
func PlanTurn(turn Turn) (plan Plan, err error) {
	if err := turn.Binding.Validate(); err != nil {
		return Plan{}, err
	}
	requested, ok := turn.Binding.Level(turn.Requested)
	if !ok {
		return Plan{}, fmt.Errorf("requested thinking level %q is not in the binding", turn.Requested)
	}
	plan = Plan{Next: startingLedger(turn.Ledger)}
	if !anchorsHold(plan.Next, turn.Messages) {
		plan.ResetReason = ResetTranscriptRewrite
		plan.Next = Ledger{Epoch: plan.Next.Epoch + 1}
	}
	plan.Replayed = countLever(plan.Next, turn.Binding.Lever)
	// Every return below reports the lever's state after this turn.
	defer plan.describe(turn.Binding.Lever)
	if entry, ok := retriedEntry(plan.Next, turn.Messages); ok {
		plan.Retry = true
		plan.Placement = entry.Placement
		return plan, nil
	}
	placement, steerable := placementFor(turn.Binding.Lever, turn.Messages)
	if !steerable || !turn.Binding.admits(placement) {
		plan.Skipped = SkipTailNotSteerable
		if steerable {
			plan.Skipped = SkipPlacementRefused
		}
		return plan, nil
	}
	payload := turn.Binding.payloadFor(requested)
	emit, skipped := shouldEmit(turn, plan.Next, requested, payload, plan.ResetReason != "")
	plan.Skipped = skipped
	if !emit {
		return plan, nil
	}
	if skip := plan.Next.append(turn, placement, payload, requested.Name); skip != "" {
		plan.Skipped = skip
		return plan, nil
	}
	plan.Emitted = true
	plan.Placement = placement
	return plan, nil
}

func (plan *Plan) describe(lever Lever) {
	state, _ := plan.Next.state(lever)
	plan.LevelInForce = state.Level
	if state.Payload >= 0 {
		plan.ControlInForce = plan.Next.Payloads[state.Payload].ControlSHA256()
	}
}

func (ledger *Ledger) append(turn Turn, placement Placement, payload Payload, level string) string {
	limit := MaxLedgerLength
	if turn.MaxEntries > 0 && turn.MaxEntries < limit {
		limit = turn.MaxEntries
	}
	index := ledger.payloadIndex(payload)
	if len(ledger.Entries) >= limit || (index < 0 && len(ledger.Payloads) >= MaxPayloads) {
		return SkipLedgerFull
	}
	if index < 0 {
		ledger.Payloads = append(ledger.Payloads, payload)
		index = len(ledger.Payloads) - 1
	}
	tail := len(turn.Messages) - 1
	ledger.Entries = append(ledger.Entries, LedgerEntry{
		Index: uint32(tail), Placement: placement, Digest: turn.Messages[tail].Digest,
		Payload: index, Turn: turn.TurnIndex,
	})
	state, _ := ledger.state(payload.Lever)
	if state.Payload != index {
		state.LastChangeTurn = turn.TurnIndex
	}
	state.Payload, state.Level = index, level
	ledger.setState(state)
	return ""
}

func startingLedger(ledger *Ledger) Ledger {
	if ledger == nil {
		return Ledger{}
	}
	return *ledger.Clone()
}

func anchorsHold(ledger Ledger, messages []Message) bool {
	for _, entry := range ledger.Entries {
		index := int(entry.Index)
		if index >= len(messages) || messages[index].Digest != entry.Digest {
			return false
		}
	}
	return true
}

func countLever(ledger Ledger, lever Lever) int {
	count := 0
	for _, entry := range ledger.Entries {
		if ledger.Payloads[entry.Payload].Lever == lever {
			count++
		}
	}
	return count
}

// retriedEntry detects a client retrying a turn whose attempt already
// committed: the tail is the message the last item was anchored to. Reusing
// that item keeps the bytes, and the cache, identical.
func retriedEntry(ledger Ledger, messages []Message) (LedgerEntry, bool) {
	if len(ledger.Entries) == 0 || len(messages) == 0 {
		return LedgerEntry{}, false
	}
	last := ledger.Entries[len(ledger.Entries)-1]
	tail := len(messages) - 1
	return last, int(last.Index) == tail && messages[tail].Digest == last.Digest
}

// placementFor places an item at the transcript tail so that no message the
// provider has already seen changes.
func placementFor(lever Lever, messages []Message) (Placement, bool) {
	if len(messages) == 0 {
		return "", false
	}
	switch messages[len(messages)-1].Role {
	case "user":
		if lever == LeverPerTurnEffort {
			return PlaceSystemBeforeTurn, true
		}
		return PlaceAppendTailUserText, true
	case "tool":
		if lever == LeverPerTurnEffort {
			return PlaceSystemAfterToolRun, true
		}
		return PlaceUserAfterToolRun, true
	default:
		return "", false
	}
}

func shouldEmit(
	turn Turn,
	ledger Ledger,
	requested Level,
	payload Payload,
	reset bool,
) (bool, string) {
	if turn.Binding.Emit == EmitEveryTurn {
		return payload.writes(), ""
	}
	state, _ := ledger.state(turn.Binding.Lever)
	// Nothing written yet in this epoch is the neutral level in force. A
	// change of bytes under the same level name, after a binding change, is
	// a change and is re-asserted.
	if state.Payload < 0 {
		if requested.Name == turn.Binding.Neutral || !payload.writes() {
			return false, ""
		}
		return true, ""
	}
	if ledger.Payloads[state.Payload] == payload {
		return false, ""
	}
	if !reset && turn.TurnIndex-state.LastChangeTurn < turn.MinTurnsBetweenChanges {
		return false, SkipChangeTooSoon
	}
	if !payload.writes() {
		// Writing nothing cannot cancel an instruction already in force.
		return false, SkipNeutralInexpressible
	}
	return true, ""
}
