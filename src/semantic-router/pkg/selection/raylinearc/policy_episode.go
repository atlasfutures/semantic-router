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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strconv"
)

// MaxPolicyLedgerEntries bounds the attribution ledger so episode state stays
// under its size limit; older turns fall off and are sent unattributed.
const MaxPolicyLedgerEntries = 256

// PolicyLedgerEntry records which action produced one assistant message.
type PolicyLedgerEntry struct {
	Message  int    `json:"message"`
	ActionID string `json:"action_id"`
	ArmID    string `json:"arm_id"`
}

// PolicyEpisodeState is the policy-service mode's per-episode memory: the
// attribution ledger, the context epoch, the request prefix the ledger's
// message indices refer to, and the compaction clock the model schedule
// reads.
//
// Epoch (the policy service's context_epoch) and EpochStartTurn are separate
// clocks. Epoch changes whenever the transcript stops extending the recorded
// prefix, because the ledger's indices no longer mean anything. EpochStartTurn
// moves only on an explicit compaction signal, as pathfinder's
// ModelRoutingState never infers compaction from a transcript change.
type PolicyEpisodeState struct {
	Epoch          int    `json:"epoch"`
	EpochStartTurn uint64 `json:"epoch_start_turn"`
	// CompactionCount is the number of compactions the episode has been told
	// of; it is the ordinal the next explicit compaction must exceed.
	CompactionCount int `json:"compaction_count,omitempty"`
	// CompactionSummary is the digest of the summary in the last recognised
	// compaction continuation. Every request of that context still opens
	// with it, so only a different summary is a new compaction.
	CompactionSummary string              `json:"compaction_summary,omitempty"`
	PrefixLen         int                 `json:"prefix_len"`
	PrefixDigest      string              `json:"prefix_digest"`
	Ledger            []PolicyLedgerEntry `json:"ledger"`
}

func (state *PolicyEpisodeState) Clone() *PolicyEpisodeState {
	if state == nil {
		return nil
	}
	cloned := *state
	cloned.Ledger = append([]PolicyLedgerEntry(nil), state.Ledger...)
	return &cloned
}

// maxPolicyArmIDBytes bounds a trained arm id; pathfinder's are 64 hex.
const maxPolicyArmIDBytes = 128

// Validate refuses policy state that could not have been written by Next: a
// persisted record is read back from a shared store, and a negative prefix
// or message index would otherwise panic when slicing the request.
func (state *PolicyEpisodeState) Validate() error {
	if state == nil {
		return nil
	}
	if state.Epoch < 0 || state.PrefixLen < 0 || state.CompactionCount < 0 {
		return errors.New("ARC policy episode state has a negative epoch, prefix or compaction count")
	}
	if state.CompactionSummary != "" && !isLowerHex64(state.CompactionSummary) {
		return errors.New("ARC policy episode compaction summary digest is malformed")
	}
	if (state.PrefixDigest != "" || state.PrefixLen != 0) && !isLowerHex64(state.PrefixDigest) {
		return errors.New("ARC policy episode prefix digest is malformed")
	}
	if len(state.Ledger) > MaxPolicyLedgerEntries {
		return errors.New("ARC policy episode ledger exceeds its bound")
	}
	for _, entry := range state.Ledger {
		if entry.Message < 0 || entry.Message > state.PrefixLen ||
			!isLowerHex64(entry.ActionID) || len(entry.ArmID) > maxPolicyArmIDBytes {
			return errors.New("ARC policy episode ledger entry is malformed")
		}
	}
	return nil
}

func isLowerHex64(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, character := range value {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

// ContextEpoch is the epoch as the policy service's context_epoch.
func (state *PolicyEpisodeState) ContextEpoch() string {
	return strconv.Itoa(state.Epoch)
}

// MessagesDigest hashes the first n raw messages exactly as received.
func MessagesDigest(messages []json.RawMessage, n int) string {
	hash := sha256.New()
	for _, message := range messages[:n] {
		hash.Write([]byte(strconv.Itoa(len(message))))
		hash.Write([]byte{':'})
		hash.Write(message)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// PolicyCallKind is what a request is to the episode's main conversation.
type PolicyCallKind string

const (
	// PolicyCallUnknown is a request no signal classified. It is served as a
	// main turn, which is what every request was before calls were
	// classified, and is reported as unknown rather than as a turn.
	PolicyCallUnknown PolicyCallKind = "unknown"
	// PolicyCallMain is a request explicitly marked as a main turn.
	PolicyCallMain PolicyCallKind = "main"
	// PolicyCallSide is a call outside the main conversation: a compaction
	// helper, a title generation, a subagent turn. It keeps the held arm and
	// leaves the episode exactly as it found it.
	PolicyCallSide PolicyCallKind = "side"
)

// PolicyCompactionSignal is an explicit statement that this request is the
// first of a compacted context. Ordinal is the compaction's ordinal from an
// explicit header (0 when none was sent); SummaryDigest is the digest of a
// recognised harness continuation summary ("" when none was found).
type PolicyCompactionSignal struct {
	Ordinal       int
	SummaryDigest string
}

// PolicyTurnSignals is what the request says about itself, read before the
// episode is consulted.
type PolicyTurnSignals struct {
	CallKind PolicyCallKind
	// CallKindSource names the signal that set CallKind, empty for unknown.
	CallKindSource string
	// Compaction is nil when the request carries no compaction signal.
	Compaction *PolicyCompactionSignal
}

// PolicyTurn transitions, as reported for one request.
const (
	PolicyTransitionStart       = "start"
	PolicyTransitionExtends     = "extends"
	PolicyTransitionPrefixBreak = "prefix_break"
	PolicyTransitionCompaction  = "compaction"
	PolicyTransitionSideCall    = "side_call"
)

// PolicyTurn resolves this request against the stored episode: the epoch it
// belongs to, the attribution for its assistant messages, and which
// transition that was.
//
//   - A side call changes nothing: it is attributed against the current
//     ledger if it extends the recorded prefix, and the state it returns is
//     the stored one.
//   - A new explicit compaction starts a new epoch with an empty ledger and
//     moves the compaction boundary (EpochStartTurn) to this turn.
//   - A request that does not extend the recorded prefix byte for byte
//     (a /clear, a rewritten history, an unrecognised compaction) starts a
//     new epoch with an empty ledger. It is not a compaction: the boundary
//     and the compaction count stay where they were.
func PolicyTurn(
	previous *PolicyEpisodeState,
	messages []json.RawMessage,
	roles []string,
	turnIndex uint64,
	signals PolicyTurnSignals,
) (*PolicyEpisodeState, []PolicyAttribution, string) {
	extends := previous != nil && previous.PrefixLen <= len(messages) &&
		MessagesDigest(messages, previous.PrefixLen) == previous.PrefixDigest
	if signals.CallKind == PolicyCallSide {
		if previous == nil {
			return &PolicyEpisodeState{}, []PolicyAttribution{}, PolicyTransitionSideCall
		}
		attribution := []PolicyAttribution{}
		if extends {
			attribution = previous.attribution(messages, roles)
		}
		return previous.Clone(), attribution, PolicyTransitionSideCall
	}
	if ordinal, summary, ok := newCompaction(previous, signals.Compaction); ok {
		return &PolicyEpisodeState{
			Epoch:             nextPolicyEpoch(previous),
			EpochStartTurn:    turnIndex,
			CompactionCount:   ordinal,
			CompactionSummary: summary,
		}, []PolicyAttribution{}, PolicyTransitionCompaction
	}
	if !extends {
		next := &PolicyEpisodeState{Epoch: nextPolicyEpoch(previous)}
		if previous == nil {
			return next, []PolicyAttribution{}, PolicyTransitionStart
		}
		next.EpochStartTurn = previous.EpochStartTurn
		next.CompactionCount = previous.CompactionCount
		next.CompactionSummary = previous.CompactionSummary
		return next, []PolicyAttribution{}, PolicyTransitionPrefixBreak
	}
	return previous.Clone(), previous.attribution(messages, roles), PolicyTransitionExtends
}

func nextPolicyEpoch(previous *PolicyEpisodeState) int {
	if previous == nil {
		return 0
	}
	return previous.Epoch + 1
}

// newCompaction reports whether signal is a compaction the episode has not
// yet applied, with the ordinal and summary digest to record. An explicit
// ordinal is authoritative and idempotent: one the episode already counted is
// a repeat, not a new context. Without one, a recognised summary is new only
// if it differs from the last one recorded, because every request of a
// compacted context still opens with its summary.
func newCompaction(previous *PolicyEpisodeState, signal *PolicyCompactionSignal) (int, string, bool) {
	if signal == nil {
		return 0, "", false
	}
	count, summary := 0, ""
	if previous != nil {
		count, summary = previous.CompactionCount, previous.CompactionSummary
	}
	if signal.SummaryDigest != "" {
		summary = signal.SummaryDigest
	}
	if signal.Ordinal > 0 {
		return signal.Ordinal, summary, signal.Ordinal > count
	}
	if signal.SummaryDigest == "" || previous != nil && signal.SummaryDigest == previous.CompactionSummary {
		return 0, "", false
	}
	return count + 1, summary, true
}

func (state *PolicyEpisodeState) attribution(messages []json.RawMessage, roles []string) []PolicyAttribution {
	attribution := make([]PolicyAttribution, 0, len(state.Ledger))
	for _, entry := range state.Ledger {
		if entry.Message >= len(messages) || roles[entry.Message] != "assistant" {
			continue
		}
		armID := entry.ArmID
		attribution = append(attribution, PolicyAttribution{
			Message: entry.Message, ActionID: entry.ActionID, ArmID: &armID,
		})
	}
	return attribution
}

// Next is the state to commit once this turn's reply is accepted: the reply
// will be message len(messages) of the next request.
func (state *PolicyEpisodeState) Next(
	messages []json.RawMessage,
	actionID string,
	armID string,
) *PolicyEpisodeState {
	next := state.Clone()
	next.Ledger = append(next.Ledger, PolicyLedgerEntry{Message: len(messages), ActionID: actionID, ArmID: armID})
	if len(next.Ledger) > MaxPolicyLedgerEntries {
		next.Ledger = next.Ledger[len(next.Ledger)-MaxPolicyLedgerEntries:]
	}
	next.PrefixLen = len(messages)
	next.PrefixDigest = MessagesDigest(messages, len(messages))
	return next
}

// ModelChangeAllowed is pathfinder's task_turn_compaction_v1 schedule
// (training/sdk/routing_schedule.py, RoutingSchedule.decision and
// ModelRoutingState.decision): the model may change at the session start,
// after five completed turns, at a compaction boundary, and five turns after
// one. Between those, the model is held and only its level varies.
//
// A counted compaction also marks a boundary at turn 0 or at the same turn as
// an earlier one, which an epoch start alone cannot express.
func ModelChangeAllowed(completedTurns uint64, epochStartTurn uint64, compactionCount int) bool {
	if completedTurns == 0 || completedTurns == 5 {
		return true
	}
	if (epochStartTurn > 0 || compactionCount > 0) &&
		(completedTurns == epochStartTurn || completedTurns == epochStartTurn+5) {
		return true
	}
	return false
}
