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
	"encoding/json"
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

// Reset reasons are a closed set so a reader can count them.
const (
	ResetBindingChanged    = "binding_changed"
	ResetTranscriptRewrite = "transcript_rewrite"
	ResetOverflow          = "ledger_overflow"
)

// Skip reasons explain a governed turn that wrote no item.
const (
	SkipTailNotSteerable     = "tail_not_steerable"
	SkipPlacementRefused     = "placement_not_admitted"
	SkipNeutralInexpressible = "neutral_inexpressible"
	SkipChangeTooSoon        = "change_too_soon"
)

const (
	DigestBytes  = 16
	maxLevels    = 16
	MaxLevelName = 32
	// MaxLedgerLength keeps a full ledger of the longest level names
	// well inside the 64 KiB episode-state limit.
	MaxLedgerLength      = 384
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

// SHA256 pins the binding's bytes. The ledger stores level names only, so a
// ledger written under one binding must never replay under another.
func (binding Binding) SHA256() string {
	payload, _ := json.Marshal(binding)
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])
}

// ControlSHA256 is the digest a trained artifact records for the action it
// chose, and the identity the router and the exporter must agree on. A text
// lever hashes its UTF-8 text. A JSON item hashes its RFC 8785 canonical
// form; for the one member an effort item carries, sorted compact encoding
// of a plain-ASCII effort name is that form, and Validate refuses anything
// else.
func (binding Binding) ControlSHA256(level Level) string {
	payload := []byte(level.Suffix)
	if binding.Lever == LeverPerTurnEffort {
		payload = []byte(`{"reasoning":{"effort":"` + level.Effort + `"}}`)
	}
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])
}

// writes reports whether a level produces an item at all. A per-turn effort
// level always does; a suffix level with no text does not.
func (binding Binding) writes(level Level) bool {
	return binding.Lever == LeverPerTurnEffort || level.Suffix != ""
}

// Ledger is the per-episode record of every item the router wrote.
// It is committed with the rest of the episode state, so an attempt that
// never reached a 2xx response leaves nothing behind.
type Ledger struct {
	BindingSHA256  string
	Epoch          uint32
	LevelInForce   string
	LastChangeTurn uint64
	Entries        []LedgerEntry
}

// LedgerEntry locates one item by the client message it is anchored
// to. Index and Digest refer to the client's transcript, which is the only
// one that survives between turns.
type LedgerEntry struct {
	Index     uint32
	Placement Placement
	Digest    string
	Level     string
	Turn      uint64
}

// Clone returns a deep copy; a nil ledger clones to nil.
func (ledger *Ledger) Clone() *Ledger {
	if ledger == nil {
		return nil
	}
	cloned := *ledger
	cloned.Entries = append([]LedgerEntry(nil), ledger.Entries...)
	return &cloned
}

// ValidateLedger refuses a persisted ledger no planner could have
// written. Binding membership is checked per turn, against the binding in
// force then; this checks only the shape.
func ValidateLedger(ledger *Ledger) error {
	if ledger == nil {
		return nil
	}
	if len(ledger.BindingSHA256) != sha256.Size*2 || !isLowerHex(ledger.BindingSHA256) {
		return errors.New("thinking ledger binding digest is invalid")
	}
	if len(ledger.LevelInForce) > MaxLevelName ||
		len(ledger.Entries) > MaxLedgerLength {
		return errors.New("thinking ledger exceeds its limits")
	}
	previous := -1
	for _, entry := range ledger.Entries {
		if int(entry.Index) <= previous {
			return errors.New("thinking ledger anchors are not ascending")
		}
		previous = int(entry.Index)
		if entry.Level == "" || len(entry.Level) > MaxLevelName ||
			len(entry.Digest) != DigestBytes*2 || !isLowerHex(entry.Digest) ||
			!knownPlacement(entry.Placement) {
			return errors.New("thinking ledger entry is invalid")
		}
	}
	return nil
}

func knownPlacement(placement Placement) bool {
	return placementFitsLever(LeverSteeringSuffix, placement) ||
		placementFitsLever(LeverPerTurnEffort, placement)
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
}

// Message is the part of a neutral message the planner reads: its
// role, and a digest that ignores what clients rewrite between turns.
type Message struct {
	Role   string
	Digest string
}

// Plan is the planner's decision. Next is staged; the caller commits
// it only with the episode state.
type Plan struct {
	Next         Ledger
	LevelInForce string
	Emitted      bool
	Retry        bool
	Placement    Placement
	ResetReason  string
	Skipped      string
	Replayed     int
}

// PlanTurn verifies the ledger against the client transcript,
// decides whether this turn writes an item, and returns the staged ledger.
// It never fails a turn for a rewritten transcript: that starts a new epoch
// and costs one cache miss, which is what the rewrite already cost.
func PlanTurn(turn Turn) (Plan, error) {
	if err := turn.Binding.Validate(); err != nil {
		return Plan{}, err
	}
	requested, ok := turn.Binding.Level(turn.Requested)
	if !ok {
		return Plan{}, fmt.Errorf("requested thinking level %q is not in the binding", turn.Requested)
	}
	plan := Plan{Next: startingLedger(turn)}
	plan.ResetReason = verifyLedger(&plan.Next, turn)
	if plan.ResetReason != "" {
		plan.Next = freshLedger(turn, plan.Next.Epoch+1)
	}
	plan.Replayed = len(plan.Next.Entries)
	tail := len(turn.Messages) - 1
	if entry, ok := retriedEntry(plan.Next, turn.Messages); ok {
		plan.Retry = true
		plan.LevelInForce = plan.Next.LevelInForce
		plan.Placement = entry.Placement
		return plan, nil
	}
	placement, steerable := placementFor(turn.Binding.Lever, turn.Messages)
	if !steerable || !turn.Binding.admits(placement) {
		plan.Skipped = SkipTailNotSteerable
		if steerable {
			plan.Skipped = SkipPlacementRefused
		}
		plan.LevelInForce = plan.Next.LevelInForce
		return plan, nil
	}
	emit, skipped := shouldEmit(turn, plan.Next, requested, plan.ResetReason != "")
	plan.Skipped = skipped
	if !emit {
		plan.LevelInForce = plan.Next.LevelInForce
		return plan, nil
	}
	if len(plan.Next.Entries) >= MaxLedgerLength {
		plan.ResetReason = ResetOverflow
		plan.Next = freshLedger(turn, plan.Next.Epoch+1)
		plan.Replayed = 0
	}
	if plan.Next.LevelInForce != requested.Name {
		plan.Next.LastChangeTurn = turn.TurnIndex
	}
	plan.Next.LevelInForce = requested.Name
	plan.Next.Entries = append(plan.Next.Entries, LedgerEntry{
		Index:     uint32(tail),
		Placement: placement,
		Digest:    turn.Messages[tail].Digest,
		Level:     requested.Name,
		Turn:      turn.TurnIndex,
	})
	plan.Emitted = true
	plan.Placement = placement
	plan.LevelInForce = requested.Name
	return plan, nil
}

func startingLedger(turn Turn) Ledger {
	if turn.Ledger == nil {
		return freshLedger(turn, 0)
	}
	return *turn.Ledger.Clone()
}

func freshLedger(turn Turn, epoch uint32) Ledger {
	return Ledger{
		BindingSHA256: turn.Binding.SHA256(),
		Epoch:         epoch,
		LevelInForce:  turn.Binding.Neutral,
	}
}

func verifyLedger(ledger *Ledger, turn Turn) string {
	if ledger.BindingSHA256 != turn.Binding.SHA256() {
		return ResetBindingChanged
	}
	previous := -1
	for _, entry := range ledger.Entries {
		index := int(entry.Index)
		if index <= previous || index >= len(turn.Messages) ||
			turn.Messages[index].Digest != entry.Digest {
			return ResetTranscriptRewrite
		}
		if _, ok := turn.Binding.Level(entry.Level); !ok {
			return ResetBindingChanged
		}
		previous = index
	}
	return ""
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

// placementFor places an item at the transcript tail so that no
// message the provider has already seen changes.
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
	reset bool,
) (bool, string) {
	writes := turn.Binding.writes(requested)
	if turn.Binding.Emit == EmitEveryTurn {
		return writes, ""
	}
	// A reset puts the neutral level in force, so a non-neutral request after
	// one is a change and is re-asserted at the new tail.
	changed := requested.Name != ledger.LevelInForce
	if !changed {
		return false, ""
	}
	if changed && !reset && len(ledger.Entries) > 0 &&
		turn.TurnIndex-ledger.LastChangeTurn < turn.MinTurnsBetweenChanges {
		return false, SkipChangeTooSoon
	}
	if !writes {
		// Writing nothing cannot cancel an instruction already in force.
		if len(ledger.Entries) > 0 && !reset {
			return false, SkipNeutralInexpressible
		}
		return false, ""
	}
	return true, ""
}
