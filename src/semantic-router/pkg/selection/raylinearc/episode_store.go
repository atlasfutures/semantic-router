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

package raylinearc

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkingcontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkinglever"
)

const (
	episodeStateSchemaV1 = "rayline.arc.episode-state.v1"
	episodeStateSchemaV2 = "rayline.arc.episode-state.v2"
	// episodeStateSchema adds the thinking-lever ledger, the upstream prefix
	// records, the policy-service state and the reasoning issuers. It is
	// written only for an episode that carries one of them, so an episode none of them touched keeps its
	// v2 bytes and an older router can still read it -- and an older router
	// refuses a v3 record by its schema rather than by a field it does not
	// know.
	episodeStateSchema   = "rayline.arc.episode-state.v3"
	maxFutureClockSkew   = 5 * time.Minute
	episodeOwnerBytes    = 24
	maxEpisodeStateBytes = 64 * 1024
)

var (
	ErrEpisodeLeaseLost = errors.New("ARC episode lease lost")
	// ErrEpisodeConflict is a relaxed commit that lost to another turn: the
	// episode changed, or a strict lease holds it, since it was read.
	ErrEpisodeConflict = errors.New("ARC episode changed since it was read")
	ErrEpisodeCapacity = errors.New("ARC episode store capacity reached")
	// ErrEpisodeLeaseHeld is joined with the context error when Prepare ran
	// out of time AFTER observing another owner's lease. A timeout without it
	// may be the store itself stalling, which is not contention.
	ErrEpisodeLeaseHeld = errors.New("ARC episode lease held by another request")
)

// Lease is an opaque fenced ownership grant. Its owner token and key are kept
// private so callers cannot serialize or log them accidentally.
type Lease struct {
	episodeIDHash string
	ownerToken    string
	version       uint64
}

func (lease Lease) Version() uint64 {
	return lease.version
}

type EpisodeStore interface {
	Prepare(
		context.Context,
		string,
		int,
	) (Lease, *EpisodeState, error)
	Commit(
		context.Context,
		Lease,
		uint64,
		*EpisodeState,
	) error
	Abort(context.Context, Lease) error
}

// EpisodeSnapshotStore serves relaxed episodes: a read that takes no lease, and
// a commit that succeeds only if the episode is still at the version read.
// A relaxed turn never waits on another; when two collide, one commit loses
// with ErrEpisodeConflict and only that turn's state update is dropped.
type EpisodeSnapshotStore interface {
	Snapshot(
		ctx context.Context,
		episodeIDHash string,
		workerCount int,
	) (*EpisodeState, EpisodeReadToken, error)
	CommitIfUnchanged(
		ctx context.Context,
		episodeIDHash string,
		read EpisodeReadToken,
		state *EpisodeState,
	) error
}

// EpisodeReadToken names exactly what a relaxed read saw, and when. A version
// alone is not enough: an episode can expire or be evicted and be recreated
// back to the same version, and a stale turn must still lose. So the token
// carries something recreation cannot repeat, a digest of the stored state
// (Redis) or the entry's store-wide generation (memory), and the time of the
// read: a commit whose read is older than the idle TTL is refused, because
// only then could a newer incarnation of the episode have come and gone,
// including one created after a read that found the episode absent.
type EpisodeReadToken struct {
	version uint64
	tag     string
	readAt  time.Time
}

// staleRead reports whether a relaxed read is too old to commit: a newer
// incarnation created after it could have expired by now.
//
// Accepted residual risk (operator decision, router-infra#55): on Redis the
// bound is checked before the commit's round trip, not inside the script. A
// read that found the episode absent can therefore still commit if, within
// that round trip and at exactly the idle-TTL boundary, a newer incarnation
// was created after the read, idled a full TTL and expired. The stale turn's
// state then lands on an episode that had already expired from disuse, so no
// live state is lost; relaxed episode state is best effort by contract.
func staleRead(read EpisodeReadToken, now time.Time, idleTTL time.Duration) bool {
	return read.readAt.IsZero() || now.Sub(read.readAt) >= idleTTL
}

// Version is the episode version the read saw.
func (token EpisodeReadToken) Version() uint64 {
	return token.version
}

// EpisodeStateStager writes state under a held lease without releasing it or
// advancing the version, so the lease's own Commit still lands and an Abort
// leaves the staged state in place. It persists what a turn must keep even if
// the turn fails, such as a boundary decision taken before dispatch.
type EpisodeStateStager interface {
	Stage(context.Context, Lease, *EpisodeState) error
}

type EpisodeLeaseRenewer interface {
	Renew(context.Context, Lease) error
}

type EpisodeStoreReadiness interface {
	Ready(context.Context) error
}

type episodeStateWire struct {
	SchemaVersion        string                  `json:"schema_version"`
	Version              uint64                  `json:"version"`
	PreviousArm          *int                    `json:"previous_arm"`
	TurnIndex            uint64                  `json:"turn_index"`
	Warmth               []*episodeWarmthWire    `json:"warmth"`
	EncoderOwner         *string                 `json:"encoder_owner,omitempty"`
	EncoderVisitedOwners *[]string               `json:"encoder_visited_owners,omitempty"`
	Thinking             *episodeThinkingWire    `json:"thinking,omitempty"`
	Upstream             []episodeUpstreamWire   `json:"upstream,omitempty"`
	Policy               *PolicyEpisodeState     `json:"policy,omitempty"`
	Controls             []episodeControlWire    `json:"controls,omitempty"`
	ReasoningIssuers     []string                `json:"reasoning_issuers,omitempty"`
	PolicyBoundary       *PolicyBoundaryDecision `json:"policy_boundary,omitempty"`
}

// episodeControlWire is one thinking-control placer. Its ledger names each
// instruction by an index into Texts: a ladder has a handful of texts, and a
// long episode writes them many times.
type episodeControlWire struct {
	Key             string                   `json:"key"`
	Format          string                   `json:"format"`
	First           *episodeControlFirstWire `json:"first,omitempty"`
	Texts           []string                 `json:"texts"`
	Ledger          []episodeControlItemWire `json:"ledger"`
	Epoch           int                      `json:"epoch"`
	InForce         *string                  `json:"in_force,omitempty"`
	PreviousAnchor  *thinkingcontrol.Anchor  `json:"previous_anchor,omitempty"`
	PreviousControl *string                  `json:"previous_control,omitempty"`
	Calls           int                      `json:"calls"`
}

// episodeControlFirstWire is what the placer checks of the episode's first
// control: its base, budget and whether it has a lever.
type episodeControlFirstWire struct {
	Native       string `json:"native"`
	BudgetTokens *int64 `json:"budget_tokens,omitempty"`
	Lever        bool   `json:"lever"`
}

type episodeControlItemWire struct {
	Anchor    int    `json:"a"`
	Digest    string `json:"d"`
	Placement string `json:"p"`
	Text      int    `json:"t"`
	Kind      string `json:"k"`
}

var (
	controlPlacementCodes = map[string]string{
		thinkingcontrol.PlacementAppend: "a", thinkingcontrol.PlacementInsertAfter: "i",
	}
	controlKindCodes = map[string]string{
		thinkingcontrol.WrittenInstruction: "i", thinkingcontrol.WrittenNeutralMarker: "m",
	}
)

func controlPlacementsToWire(placements []ControlPlacement) []episodeControlWire {
	out := make([]episodeControlWire, 0, len(placements))
	for _, placement := range placements {
		state := placement.State
		wire := episodeControlWire{
			Key: placement.Key, Format: state.Format, Texts: []string{}, Ledger: []episodeControlItemWire{},
			Epoch: state.Epoch, InForce: state.InForce, PreviousAnchor: state.PreviousAnchor,
			PreviousControl: state.PreviousControl, Calls: state.Calls,
		}
		if state.First != nil {
			wire.First = &episodeControlFirstWire{
				Native: state.First.Native, BudgetTokens: state.First.BudgetTokens, Lever: state.First.Instruction != nil,
			}
		}
		texts := map[string]int{}
		for _, item := range state.Ledger {
			index, ok := texts[item.Text]
			if !ok {
				index = len(wire.Texts)
				texts[item.Text] = index
				wire.Texts = append(wire.Texts, item.Text)
			}
			wire.Ledger = append(wire.Ledger, episodeControlItemWire{
				Anchor: item.Anchor, Digest: item.PrefixDigest, Placement: controlPlacementCodes[item.Placement],
				Text: index, Kind: controlKindCodes[item.Kind],
			})
		}
		out = append(out, wire)
	}
	return out
}

func controlPlacementsFromWire(wires []episodeControlWire) ([]ControlPlacement, error) {
	if len(wires) == 0 {
		return nil, nil
	}
	if len(wires) > MaxControlPlacements {
		return nil, errors.New("ARC episode state holds too many control placements")
	}
	out := make([]ControlPlacement, 0, len(wires))
	seen := make(map[string]bool, len(wires))
	for _, wire := range wires {
		if wire.Key == "" || seen[wire.Key] {
			return nil, errors.New("ARC episode control placement key is empty or repeated")
		}
		seen[wire.Key] = true
		state := thinkingcontrol.PlacerState{
			Format: wire.Format, Ledger: []thinkingcontrol.LedgerItem{}, Epoch: wire.Epoch, InForce: wire.InForce,
			PreviousAnchor: wire.PreviousAnchor, PreviousControl: wire.PreviousControl, Calls: wire.Calls,
		}
		if wire.First != nil {
			first := thinkingcontrol.Control{Native: wire.First.Native, BudgetTokens: wire.First.BudgetTokens}
			if wire.First.Lever {
				first.Instruction = &thinkingcontrol.Instruction{}
			}
			state.First = &first
		}
		for _, item := range wire.Ledger {
			placement, kind := codeName(controlPlacementCodes, item.Placement), codeName(controlKindCodes, item.Kind)
			if placement == "" || kind == "" || item.Text < 0 || item.Text >= len(wire.Texts) {
				return nil, errors.New("ARC episode control ledger is malformed")
			}
			state.Ledger = append(state.Ledger, thinkingcontrol.LedgerItem{
				Anchor: item.Anchor, PrefixDigest: item.Digest, Placement: placement, Text: wire.Texts[item.Text], Kind: kind,
			})
		}
		if _, err := thinkingcontrol.ResumePlacer(state); err != nil {
			return nil, errors.New("ARC episode control placement is malformed")
		}
		out = append(out, ControlPlacement{Key: wire.Key, State: state})
	}
	return out, nil
}

func codeName(codes map[string]string, code string) string {
	for name, candidate := range codes {
		if candidate == code {
			return name
		}
	}
	return ""
}

func cloneControlPlacements(placements []ControlPlacement) []ControlPlacement {
	if placements == nil {
		return nil
	}
	out := make([]ControlPlacement, len(placements))
	for index, placement := range placements {
		state := placement.State
		state.Ledger = append([]thinkingcontrol.LedgerItem(nil), state.Ledger...)
		out[index] = ControlPlacement{Key: placement.Key, State: state}
	}
	return out
}

// CloneControlPlacements is a copy the caller may extend.
func CloneControlPlacements(placements []ControlPlacement) []ControlPlacement {
	return cloneControlPlacements(placements)
}

type episodeUpstreamWire struct {
	Worker   string `json:"w"`
	Messages int    `json:"n"`
	Digest   string `json:"d"`
}

type episodeThinkingWire struct {
	Epoch    uint32                       `json:"epoch"`
	Payloads []episodeThinkingPayloadWire `json:"payloads"`
	Entries  []episodeThinkingEntryWire   `json:"entries"`
	InForce  []episodeThinkingStateWire   `json:"in_force"`
}

type episodeThinkingPayloadWire struct {
	Lever  string `json:"lever"`
	Suffix string `json:"suffix,omitempty"`
	Effort string `json:"effort,omitempty"`
}

// episodeThinkingEntryWire uses short keys and placement codes: a long
// every-turn ledger is the largest thing an episode record carries.
type episodeThinkingEntryWire struct {
	Index     uint32 `json:"i"`
	Placement string `json:"p"`
	Digest    string `json:"d"`
	Payload   int    `json:"k"`
	Turn      uint64 `json:"t"`
}

type episodeThinkingStateWire struct {
	Lever          string `json:"lever"`
	Payload        int    `json:"payload"`
	Level          string `json:"level"`
	LastChangeTurn uint64 `json:"last_change_turn"`
	// Marker is written only when true, so a record without a neutral
	// marker keeps the shape every earlier build decodes.
	Marker bool `json:"marker,omitempty"`
}

var thinkingPlacementCodes = map[thinkinglever.Placement]string{
	thinkinglever.PlaceAppendTailUserText: "a",
	thinkinglever.PlaceUserAfterToolRun:   "u",
	thinkinglever.PlaceSystemBeforeTurn:   "sb",
	thinkinglever.PlaceSystemAfterToolRun: "sa",
}

type episodeWarmthWire struct {
	LastUsedUnixMS  int64 `json:"last_used_unix_ms"`
	LastInputTokens int   `json:"last_input_tokens"`
}

func newEpisodeOwnerToken() (string, error) {
	raw := make([]byte, episodeOwnerBytes)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("create ARC episode owner token: %w", err)
	}
	return hex.EncodeToString(raw), nil
}

func cloneEpisodeState(state *EpisodeState) *EpisodeState {
	if state == nil {
		return nil
	}
	cloned := &EpisodeState{
		PreviousArm:          cloneEpisodeArm(state.PreviousArm),
		TurnIndex:            state.TurnIndex,
		Warmth:               make([]*WorkerWarmth, len(state.Warmth)),
		EncoderOwner:         state.EncoderOwner,
		EncoderVisitedOwners: append([]string(nil), state.EncoderVisitedOwners...),
		Policy:               state.Policy.Clone(),
		Thinking:             state.Thinking.Clone(),
		Upstream:             append([]UpstreamPrefix(nil), state.Upstream...),
		Controls:             cloneControlPlacements(state.Controls),
		ReasoningIssuers:     append([]string(nil), state.ReasoningIssuers...),
		PolicyBoundary:       clonePolicyBoundary(state.PolicyBoundary),
	}
	for index, warmth := range state.Warmth {
		if warmth == nil {
			continue
		}
		value := *warmth
		cloned.Warmth[index] = &value
	}
	return cloned
}

func clonePolicyBoundary(boundary *PolicyBoundaryDecision) *PolicyBoundaryDecision {
	if boundary == nil {
		return nil
	}
	cloned := *boundary
	return &cloned
}

// ClonePolicyBoundary is a copy the caller may change.
func ClonePolicyBoundary(boundary *PolicyBoundaryDecision) *PolicyBoundaryDecision {
	return clonePolicyBoundary(boundary)
}

func cloneEpisodeArm(value *int) *int {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func marshalEpisodeState(
	state *EpisodeState,
	version uint64,
	now time.Time,
) ([]byte, error) {
	if err := validatePersistedEpisodeState(state, now); err != nil {
		return nil, err
	}
	wire := episodeStateWire{
		SchemaVersion: episodeStateSchemaV2,
		Version:       version,
		PreviousArm:   cloneEpisodeArm(state.PreviousArm),
		TurnIndex:     state.TurnIndex,
		Warmth:        make([]*episodeWarmthWire, len(state.Warmth)),
		Policy:        state.Policy.Clone(),
	}
	if state.Policy != nil {
		wire.SchemaVersion = episodeStateSchema
	}
	if state.Thinking != nil {
		wire.SchemaVersion = episodeStateSchema
		wire.Thinking = thinkingLedgerToWire(state.Thinking)
	}
	if len(state.Upstream) > 0 {
		wire.SchemaVersion = episodeStateSchema
		for _, prefix := range state.Upstream {
			wire.Upstream = append(wire.Upstream, episodeUpstreamWire(prefix))
		}
	}
	if len(state.Controls) > 0 {
		wire.SchemaVersion = episodeStateSchema
		wire.Controls = controlPlacementsToWire(state.Controls)
	}
	if len(state.ReasoningIssuers) > 0 {
		wire.SchemaVersion = episodeStateSchema
		wire.ReasoningIssuers = append([]string(nil), state.ReasoningIssuers...)
	}
	if state.PolicyBoundary != nil {
		wire.SchemaVersion = episodeStateSchema
		wire.PolicyBoundary = clonePolicyBoundary(state.PolicyBoundary)
	}
	owner := state.EncoderOwner
	visited := append([]string{}, state.EncoderVisitedOwners...)
	wire.EncoderOwner = &owner
	wire.EncoderVisitedOwners = &visited
	for index, warmth := range state.Warmth {
		if warmth == nil {
			continue
		}
		wire.Warmth[index] = &episodeWarmthWire{
			LastUsedUnixMS:  warmth.LastUsed.UTC().UnixMilli(),
			LastInputTokens: warmth.LastInputTokens,
		}
	}
	payload, err := json.Marshal(wire)
	if err != nil {
		return nil, errors.New("marshal ARC episode state")
	}
	if len(payload) > maxEpisodeStateBytes {
		return nil, errors.New("ARC episode state exceeds size limit")
	}
	return payload, nil
}

func unmarshalEpisodeState(
	payload []byte,
	workerCount int,
	now time.Time,
) (*EpisodeState, uint64, error) {
	if len(payload) == 0 {
		state, err := NewEpisodeState(workerCount)
		return state, 0, err
	}
	if len(payload) > maxEpisodeStateBytes {
		return nil, 0, errors.New("ARC episode state exceeds size limit")
	}
	var wire episodeStateWire
	if err := decodeStrictJSON(payload, &wire); err != nil {
		return nil, 0, errors.New("decode ARC episode state")
	}
	if len(wire.Warmth) != workerCount {
		return nil, 0, errors.New("ARC episode state contract mismatch")
	}
	owner, visited, err := decodeEpisodeStateAffinity(wire)
	if err != nil {
		return nil, 0, err
	}
	state := episodeStateFromWire(wire, workerCount, owner, visited)
	if state.Controls, err = controlPlacementsFromWire(wire.Controls); err != nil {
		return nil, 0, err
	}
	if err := validatePersistedEpisodeState(state, now); err != nil {
		return nil, 0, err
	}
	return state, wire.Version, nil
}

func decodeEpisodeStateAffinity(
	wire episodeStateWire,
) (string, []string, error) {
	if (wire.Thinking != nil || len(wire.Upstream) > 0 || wire.Policy != nil || len(wire.Controls) > 0 ||
		len(wire.ReasoningIssuers) > 0 || wire.PolicyBoundary != nil) !=
		(wire.SchemaVersion == episodeStateSchema) {
		return "", nil, errors.New("ARC episode state contract mismatch")
	}
	switch wire.SchemaVersion {
	case episodeStateSchemaV1:
		if wire.EncoderOwner != nil || wire.EncoderVisitedOwners != nil {
			return "", nil, errors.New("ARC episode state contract mismatch")
		}
		return "", nil, nil
	case episodeStateSchemaV2, episodeStateSchema:
		if wire.EncoderOwner == nil || wire.EncoderVisitedOwners == nil ||
			*wire.EncoderVisitedOwners == nil {
			return "", nil, errors.New("ARC episode state contract mismatch")
		}
		return *wire.EncoderOwner,
			append([]string(nil), (*wire.EncoderVisitedOwners)...), nil
	default:
		return "", nil, errors.New("ARC episode state contract mismatch")
	}
}

func episodeStateFromWire(
	wire episodeStateWire,
	workerCount int,
	owner string,
	visited []string,
) *EpisodeState {
	state := &EpisodeState{
		PreviousArm:          cloneEpisodeArm(wire.PreviousArm),
		TurnIndex:            wire.TurnIndex,
		Warmth:               make([]*WorkerWarmth, workerCount),
		EncoderOwner:         owner,
		EncoderVisitedOwners: visited,
		Policy:               wire.Policy.Clone(),
		Thinking:             thinkingLedgerFromWire(wire.Thinking),
		PolicyBoundary:       clonePolicyBoundary(wire.PolicyBoundary),
	}
	if len(wire.ReasoningIssuers) > 0 {
		state.ReasoningIssuers = append([]string(nil), wire.ReasoningIssuers...)
	}
	for _, prefix := range wire.Upstream {
		state.Upstream = append(state.Upstream, UpstreamPrefix(prefix))
	}
	for index, warmth := range wire.Warmth {
		if warmth == nil {
			continue
		}
		state.Warmth[index] = &WorkerWarmth{
			LastUsed:        time.UnixMilli(warmth.LastUsedUnixMS).UTC(),
			LastInputTokens: warmth.LastInputTokens,
		}
	}
	return state
}

func validatePersistedEpisodeState(
	state *EpisodeState,
	now time.Time,
) error {
	if now.IsZero() {
		return errors.New("ARC episode store clock is required")
	}
	if state == nil {
		return errors.New("ARC episode state is required")
	}
	if err := validateEpisodeState(state, len(state.Warmth)); err != nil {
		return err
	}
	if err := state.Policy.Validate(); err != nil {
		return err
	}
	if err := state.PolicyBoundary.Validate(len(state.Warmth)); err != nil {
		return err
	}
	if len(state.Controls) > MaxControlPlacements {
		return errors.New("ARC episode state holds too many control placements")
	}
	for _, placement := range state.Controls {
		if _, err := thinkingcontrol.ResumePlacer(placement.State); err != nil || placement.Key == "" {
			return errors.New("ARC episode control placement is malformed")
		}
	}
	futureLimit := now.Add(maxFutureClockSkew)
	for _, warmth := range state.Warmth {
		if warmth != nil && warmth.LastUsed.After(futureLimit) {
			return errors.New("ARC episode warmth is implausibly future-dated")
		}
	}
	return nil
}

func thinkingLedgerToWire(ledger *thinkinglever.Ledger) *episodeThinkingWire {
	wire := &episodeThinkingWire{
		Epoch:    ledger.Epoch,
		Payloads: make([]episodeThinkingPayloadWire, len(ledger.Payloads)),
		Entries:  make([]episodeThinkingEntryWire, len(ledger.Entries)),
		InForce:  make([]episodeThinkingStateWire, len(ledger.InForce)),
	}
	for index, payload := range ledger.Payloads {
		wire.Payloads[index] = episodeThinkingPayloadWire{
			Lever: string(payload.Lever), Suffix: payload.Suffix, Effort: payload.Effort,
		}
	}
	for index, entry := range ledger.Entries {
		wire.Entries[index] = episodeThinkingEntryWire{
			Index: entry.Index, Placement: thinkingPlacementCodes[entry.Placement],
			Digest: entry.Digest, Payload: entry.Payload, Turn: entry.Turn,
		}
	}
	for index, state := range ledger.InForce {
		wire.InForce[index] = episodeThinkingStateWire{
			Lever: string(state.Lever), Payload: state.Payload,
			Level: state.Level, LastChangeTurn: state.LastChangeTurn, Marker: state.Marker,
		}
	}
	return wire
}

// thinkingLedgerFromWire decodes the ledger; an unknown placement code
// decodes to an empty placement, which ValidateLedger then refuses.
func thinkingLedgerFromWire(wire *episodeThinkingWire) *thinkinglever.Ledger {
	if wire == nil {
		return nil
	}
	placements := make(map[string]thinkinglever.Placement, len(thinkingPlacementCodes))
	for placement, code := range thinkingPlacementCodes {
		placements[code] = placement
	}
	ledger := &thinkinglever.Ledger{
		Epoch:    wire.Epoch,
		Payloads: make([]thinkinglever.Payload, len(wire.Payloads)),
		Entries:  make([]thinkinglever.LedgerEntry, len(wire.Entries)),
		InForce:  make([]thinkinglever.LeverState, len(wire.InForce)),
	}
	for index, payload := range wire.Payloads {
		ledger.Payloads[index] = thinkinglever.Payload{
			Lever: thinkinglever.Lever(payload.Lever), Suffix: payload.Suffix, Effort: payload.Effort,
		}
	}
	for index, entry := range wire.Entries {
		ledger.Entries[index] = thinkinglever.LedgerEntry{
			Index: entry.Index, Placement: placements[entry.Placement],
			Digest: entry.Digest, Payload: entry.Payload, Turn: entry.Turn,
		}
	}
	for index, state := range wire.InForce {
		ledger.InForce[index] = thinkinglever.LeverState{
			Lever: thinkinglever.Lever(state.Lever), Payload: state.Payload,
			Level: state.Level, LastChangeTurn: state.LastChangeTurn, Marker: state.Marker,
		}
	}
	return ledger
}
