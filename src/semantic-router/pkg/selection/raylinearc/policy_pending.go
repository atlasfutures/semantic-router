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
	// PrefillBaseDigest is, when the decision's request ended with an
	// assistant message (a prefill the model continues), the digest of the
	// messages before it: the next turn carries the completed reply in that
	// message's place. Empty otherwise.
	PrefillBaseDigest string `json:"prefill_base_digest,omitempty"`
	// KeptBoundary is the boundary decision a failed generation turn had
	// stored for these messages when the lookup wrote this decision and kept
	// (it answered that arm); nil when there was none. A stored boundary for
	// the same messages that is not this one was taken after the decision.
	KeptBoundary *PolicyBoundaryDecision `json:"kept_boundary,omitempty"`
	// PriorServedModel is the served-model record as the decision read it,
	// before the decision overwrote it with its own model; empty when it
	// read none. A request that drops the decision unexecuted reads this
	// instead of the record, which now names the dropped decision.
	PriorServedModel string `json:"prior_served_model,omitempty"`
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
// digests and roles: arm, its model, the policy state the decision would commit
// (nil when it records none) and the encoded tokens, at turnIndex completed
// turns. A route id longer than the stored bound is not kept.
func NewPendingDecision(
	routeID string,
	arm int,
	model string,
	turnIndex uint64,
	inputTokens int,
	digests *PrefixDigests,
	roles []string,
	next *PolicyEpisodeState,
	now time.Time,
) *PendingDecision {
	count := len(digests.Messages())
	pending := &PendingDecision{
		Arm: arm, Model: model, TurnIndex: turnIndex, DecidedUnixMS: now.UTC().UnixMilli(),
		InputTokens: inputTokens, PrefixLen: count, PrefixDigest: digests.Digest(count),
	}
	if count > 0 && len(roles) == count && roles[count-1] == "assistant" {
		pending.PrefillBaseDigest = digests.Digest(count - 1)
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
	cloned.KeptBoundary = clonePolicyBoundary(pending.KeptBoundary)
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
		len(pending.RouteID) > maxPendingRouteIDBytes ||
		(pending.PrefillBaseDigest != "" && !isLowerHex64(pending.PrefillBaseDigest)) {
		return malformed
	}
	if err := pending.KeptBoundary.Validate(workerCount); err != nil {
		return err
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
	// PendingSuperseded: a generation call for the decision's own messages
	// decided after it and stored its boundary decision before it ran; the
	// cell cannot tell the lookup's decision ran, so it was dropped.
	PendingSuperseded = "superseded"
)

// ResolvePending is the episode a request of digests and roles is decided
// on, given the pending decision the episode holds:
//
//   - the request's next turn records the decision as a turn that ran,
//     marked inferred: it extends the decision's messages, carries an
//     assistant message right after them and something after that reply
//     (a request ending at an assistant message is a prefill of the same
//     turn); or, for a decision whose request ended with a prefill, it
//     carries the completed reply in the prefill's place and something
//     after it;
//   - any other request drops it: the same turn asked again (identical, or
//     with its last message edited, with a message appended beside it
//     without a reply, or with a prefill added), or one that departs from
//     it.
//
// roles are the request's, with every assistant output item (of any kind)
// read as "assistant".
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
	if boundary := state.PolicyBoundary; boundary != nil && boundary.TurnIndex == state.TurnIndex &&
		boundary.PrefixLen == pending.PrefixLen && boundary.PrefixDigest == pending.PrefixDigest &&
		!boundary.Same(pending.KeptBoundary) {
		// A boundary decision only names an arm, and several actions (one
		// per thinking level) may share it: any decision a generation call
		// took after the lookup supersedes the lookup's.
		return resolved, PendingSuperseded
	}
	reply, ok := pending.replyIndex(digests, roles)
	if !ok {
		return resolved, PendingDropped
	}
	if pending.Policy != nil {
		resolved.Policy = pending.inferredPolicy(state.Policy, digests, reply)
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

// replyIndex is where the request carries the reply to the decision, and
// false when it carries none (ResolvePending).
func (pending *PendingDecision) replyIndex(digests *PrefixDigests, roles []string) (int, bool) {
	decided, count := pending.PrefixLen, len(digests.Messages())
	if len(roles) != count || decided >= count {
		return 0, false
	}
	if decided+1 < count && roles[decided] == "assistant" && digests.Digest(decided) == pending.PrefixDigest {
		return decided, true
	}
	if pending.PrefillBaseDigest != "" && roles[decided-1] == "assistant" && roles[count-1] != "assistant" &&
		digests.Digest(decided-1) == pending.PrefillBaseDigest {
		return decided - 1, true
	}
	return 0, false
}

// inferredPolicy is stored with the pending decision's ledger entry
// recorded, marked inferred, in the context the decision was made in.
func (pending *PendingDecision) inferredPolicy(stored *PolicyEpisodeState, digests *PrefixDigests, reply int) *PolicyEpisodeState {
	policy := pending.Policy
	var base *PolicyEpisodeState
	if stored != nil && stored.Epoch == policy.Epoch && stored.PrefixLen <= pending.PrefixLen &&
		digests.Digest(stored.PrefixLen) == stored.PrefixDigest {
		base = stored.Clone()
	} else {
		// The decision opened a new context (a prefix break or a
		// compaction), which starts with an empty ledger and the full offer;
		// or the store moved to another context after it (a refused turn in
		// a rewritten history staged one). Either way the context is new,
		// numbered past any the store has used.
		epoch := policy.Epoch
		if stored != nil && stored.Epoch >= epoch {
			epoch = stored.Epoch + 1
		}
		base = &PolicyEpisodeState{Epoch: epoch}
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
	// The reply is the message at reply: the decided message count for an
	// ordinary turn, the prefill's place for a completed prefill. The
	// context's prefix is the decided message count, as a served turn's is,
	// over the request's messages (a completed prefill's reply replaces the
	// prefill).
	entry.Message = reply
	next := base.withLedgerEntry(entry, pending.PrefixLen, digests.Digest(pending.PrefixLen))
	next.Exploration = policy.Exploration.Clone()
	return next
}
