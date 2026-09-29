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
// attribution ledger, the context epoch, and the request prefix the ledger's
// message indices refer to.
type PolicyEpisodeState struct {
	Epoch          int                 `json:"epoch"`
	EpochStartTurn uint64              `json:"epoch_start_turn"`
	PrefixLen      int                 `json:"prefix_len"`
	PrefixDigest   string              `json:"prefix_digest"`
	Ledger         []PolicyLedgerEntry `json:"ledger"`
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
	if state.Epoch < 0 || state.PrefixLen < 0 {
		return errors.New("ARC policy episode state has a negative epoch or prefix")
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

// PolicyTurn resolves this request against the stored episode: the epoch it
// belongs to and the attribution for its assistant messages. A request that
// does not extend the recorded prefix byte for byte starts a new epoch with an
// empty ledger (compaction, /clear, or a client that rewrote history).
func PolicyTurn(
	previous *PolicyEpisodeState,
	messages []json.RawMessage,
	roles []string,
	turnIndex uint64,
) (*PolicyEpisodeState, []PolicyAttribution) {
	extends := previous != nil && previous.PrefixLen <= len(messages) &&
		MessagesDigest(messages, previous.PrefixLen) == previous.PrefixDigest
	if !extends {
		epoch := 0
		if previous != nil {
			epoch = previous.Epoch + 1
		}
		return &PolicyEpisodeState{Epoch: epoch, EpochStartTurn: turnIndex}, []PolicyAttribution{}
	}
	current := previous.Clone()
	attribution := make([]PolicyAttribution, 0, len(current.Ledger))
	for _, entry := range current.Ledger {
		if entry.Message >= len(messages) || roles[entry.Message] != "assistant" {
			continue
		}
		armID := entry.ArmID
		attribution = append(attribution, PolicyAttribution{
			Message: entry.Message, ActionID: entry.ActionID, ArmID: &armID,
		})
	}
	return current, attribution
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
// (training/sdk/routing_schedule.py): the model may change at the session
// start, after five completed turns, at a compaction boundary, and five turns
// after one. Between those, the model is held and only its level varies.
func ModelChangeAllowed(completedTurns uint64, epochStartTurn uint64) bool {
	if completedTurns == 0 || completedTurns == 5 {
		return true
	}
	if epochStartTurn > 0 && (completedTurns == epochStartTurn || completedTurns == epochStartTurn+5) {
		return true
	}
	return false
}
