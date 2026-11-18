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
	"errors"
	"time"
)

// PolicyLedgerBasisInferred marks a ledger entry the cell recorded by
// assuming a pending decision ran: the conversation's next turn arrived
// carrying a reply after the message the decision was for, and nothing said
// which model wrote it. An entry without a basis is one the cell served.
const PolicyLedgerBasisInferred = "inferred"

// maxPendingRouteIDBytes bounds the route id a pending decision keeps; a
// longer caller-minted id is not kept.
const maxPendingRouteIDBytes = 128

// PendingDecision is a route lookup's decision that nothing has executed as
// far as the cell knows. It is stored beside the trajectory, which it leaves
// as it was: the next request of the conversation resolves it before
// anything reads the episode (ResolvePending).
type PendingDecision struct {
	// RouteID is the lookup's route id, so a later report can name the turn.
	RouteID string `json:"route_id,omitempty"`
	Arm     int    `json:"arm"`
	// Model is the decided action's model, the identity the served-model
	// record keeps.
	Model string `json:"model,omitempty"`
	// TurnIndex is the completed-turn count the decision was made at.
	TurnIndex     uint64 `json:"turn_index"`
	DecidedUnixMS int64  `json:"decided_unix_ms"`
	// InputTokens is what the arm's warmth records if the decision is
	// assumed to have run, as a committed turn's does.
	InputTokens int `json:"input_tokens"`
	// PrefixLen and PrefixDigest are the decision request's messages: the
	// reply to the decision is message PrefixLen of the next turn.
	PrefixLen    int    `json:"prefix_len"`
	PrefixDigest string `json:"prefix_digest"`
	// Policy is the policy context the decision was made in and its ledger
	// entry; nil when it recorded none.
	Policy *PendingPolicy `json:"policy,omitempty"`
}

// PendingPolicy is a pending decision's policy context and ledger entry.
// The context is the one the decision's request resolved to: the stored one
// when its epoch is the stored epoch, a new one (prefix break, compaction)
// otherwise. Only the new entry is kept, not a copy of the stored ledger.
type PendingPolicy struct {
	Epoch             int                      `json:"epoch"`
	EpochStartTurn    uint64                   `json:"epoch_start_turn"`
	CompactionCount   int                      `json:"compaction_count,omitempty"`
	CompactionSummary string                   `json:"compaction_summary,omitempty"`
	Exploration       *PolicyExplorationOrigin `json:"exploration,omitempty"`
	Entry             PolicyLedgerEntry        `json:"entry"`
}

// NewPendingDecision is the pending form of a decision for a request of
// digests: arm, its model, the policy state the decision would commit
// (nil when it records none) and the encoded tokens, at turnIndex completed
// turns. A route id longer than the stored bound is not kept.
func NewPendingDecision(
	routeID string,
	arm int,
	model string,
	turnIndex uint64,
	inputTokens int,
	digests *PrefixDigests,
	next *PolicyEpisodeState,
	now time.Time,
) *PendingDecision {
	count := len(digests.Messages())
	pending := &PendingDecision{
		Arm: arm, Model: model, TurnIndex: turnIndex, DecidedUnixMS: now.UTC().UnixMilli(),
		InputTokens: inputTokens, PrefixLen: count, PrefixDigest: digests.Digest(count),
	}
	if len(routeID) <= maxPendingRouteIDBytes {
		pending.RouteID = routeID
	}
	if next != nil && len(next.Ledger) > 0 {
		entry := next.Ledger[len(next.Ledger)-1]
		pending.Policy = &PendingPolicy{
			Epoch: next.Epoch, EpochStartTurn: next.EpochStartTurn, CompactionCount: next.CompactionCount,
			CompactionSummary: next.CompactionSummary, Exploration: next.Exploration.Clone(), Entry: entry,
		}
		if entry.Prices != nil {
			prices := *entry.Prices
			pending.Policy.Entry.Prices = &prices
		}
	}
	return pending
}

// ClonePendingDecision is a copy the caller may change.
func ClonePendingDecision(pending *PendingDecision) *PendingDecision {
	if pending == nil {
		return nil
	}
	cloned := *pending
	if pending.Policy != nil {
		policy := *pending.Policy
		policy.Exploration = pending.Policy.Exploration.Clone()
		if pending.Policy.Entry.Prices != nil {
			prices := *pending.Policy.Entry.Prices
			policy.Entry.Prices = &prices
		}
		cloned.Policy = &policy
	}
	return &cloned
}

// Validate refuses a stored pending decision NewPendingDecision could not
// have produced.
func (pending *PendingDecision) Validate(workerCount int) error {
	if pending == nil {
		return nil
	}
	malformed := errors.New("ARC episode pending decision is malformed")
	if pending.Arm < 0 || pending.Arm >= workerCount || pending.PrefixLen < 1 || pending.InputTokens < 0 ||
		pending.DecidedUnixMS <= 0 || !isLowerHex64(pending.PrefixDigest) ||
		len(pending.RouteID) > maxPendingRouteIDBytes || len(pending.Model) > MaxPolicyExclusionModelBytes {
		return malformed
	}
	policy := pending.Policy
	if policy == nil {
		return nil
	}
	entry := policy.Entry
	if policy.Epoch < 0 || policy.CompactionCount < 0 || policy.EpochStartTurn > pending.TurnIndex ||
		(policy.CompactionSummary != "" && !isLowerHex64(policy.CompactionSummary)) ||
		entry.Message != pending.PrefixLen || !isLowerHex64(entry.ActionID) ||
		len(entry.ArmID) > maxPolicyArmIDBytes || entry.Basis != "" {
		return malformed
	}
	if err := entry.Prices.validate(); err != nil {
		return err
	}
	return policy.Exploration.validate()
}

// Pending resolutions, as ResolvePending reports them.
const (
	// PendingNone: the episode held no pending decision.
	PendingNone = ""
	// PendingInferred: the request is the decision's next turn; the
	// decision was recorded as a turn that ran.
	PendingInferred = "inferred"
	// PendingDropped: the request is the decision's turn asked again
	// (identical, edited, or with a message appended beside it), or departs
	// from it; the decision was dropped unrecorded.
	PendingDropped = "dropped"
	// PendingStale: a turn committed after the decision, so it no longer
	// describes the episode; it was dropped.
	PendingStale = "stale"
)

// ResolvePending is the episode a request of digests and roles is decided
// on, given the pending decision the episode holds:
//
//   - the request's next turn -- it extends the decision's messages and
//     carries an assistant message right after them -- records the decision
//     as a turn that ran, marked inferred;
//   - any other request drops it: the same turn asked again (identical, or
//     with its last message edited, or with a message appended beside it
//     without a reply), or one that departs from it.
//
// The resolved state holds no pending decision. state itself is never
// changed; it is returned as is when it holds none.
func ResolvePending(state *EpisodeState, digests *PrefixDigests, roles []string) (*EpisodeState, string) {
	if state == nil || state.Pending == nil {
		return state, PendingNone
	}
	pending := state.Pending
	resolved := cloneEpisodeState(state)
	resolved.Pending = nil
	if pending.TurnIndex != state.TurnIndex {
		return resolved, PendingStale
	}
	messages := digests.Messages()
	if pending.PrefixLen >= len(messages) || pending.PrefixLen >= len(roles) ||
		roles[pending.PrefixLen] != "assistant" || digests.Digest(pending.PrefixLen) != pending.PrefixDigest {
		return resolved, PendingDropped
	}
	if pending.Policy != nil {
		resolved.Policy = pending.inferredPolicy(state.Policy)
	}
	// The arm's last call record outlives the inferred turn, which made no
	// call the cell saw: Commit replaces the arm's warmth.
	previous := resolved.Warmth[pending.Arm]
	if err := resolved.Commit(pending.Arm, pending.InputTokens, time.UnixMilli(pending.DecidedUnixMS).UTC()); err != nil {
		return resolved, PendingDropped
	}
	if previous != nil {
		resolved.Warmth[pending.Arm].Call = previous.Call
	}
	resolved.ServedModel = pending.Model
	return resolved, PendingInferred
}

// inferredPolicy is stored with the pending decision's ledger entry
// recorded, marked inferred, in the context the decision was made in.
func (pending *PendingDecision) inferredPolicy(stored *PolicyEpisodeState) *PolicyEpisodeState {
	policy := pending.Policy
	var base *PolicyEpisodeState
	if stored != nil && stored.Epoch == policy.Epoch {
		base = stored.Clone()
	} else {
		// The decision opened a new context (a prefix break or a
		// compaction), which starts with an empty ledger and the full offer.
		base = &PolicyEpisodeState{Epoch: policy.Epoch}
	}
	base.EpochStartTurn = policy.EpochStartTurn
	base.CompactionCount = policy.CompactionCount
	base.CompactionSummary = policy.CompactionSummary
	entry := policy.Entry
	entry.Basis = PolicyLedgerBasisInferred
	if entry.Prices != nil {
		prices := *entry.Prices
		entry.Prices = &prices
	}
	next := base.withLedgerEntry(entry, pending.PrefixLen, pending.PrefixDigest)
	next.Exploration = policy.Exploration.Clone()
	return next
}
