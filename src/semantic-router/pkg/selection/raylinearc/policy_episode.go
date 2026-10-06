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
	"bytes"
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
	// Exclusions are the models this context no longer offers: a model that
	// refused a turn (ADR 0120). They live in the policy state, so they last
	// exactly as long as the context does: a compaction or a prefix break
	// starts a fresh state, and with it the full offer again.
	Exclusions []PolicyExclusion `json:"exclusions,omitempty"`
}

// PolicyExclusion removes every action of a trained model from the offer,
// with the failure class that excluded it.
type PolicyExclusion struct {
	Model string `json:"model"`
	Class string `json:"class"`
}

// MaxPolicyExclusions bounds the exclusions an episode can carry; a package
// serves far fewer models.
const MaxPolicyExclusions = 16

// Excludes reports whether model is excluded in this context.
func (state *PolicyEpisodeState) Excludes(model string) bool {
	if state == nil || model == "" {
		return false
	}
	for _, exclusion := range state.Exclusions {
		if exclusion.Model == model {
			return true
		}
	}
	return false
}

// WithExclusion is the state with model excluded for class, unchanged when
// the model is already excluded or the bound is reached.
func (state *PolicyEpisodeState) WithExclusion(model, class string) *PolicyEpisodeState {
	next := state.Clone()
	if next == nil {
		next = &PolicyEpisodeState{}
	}
	if next.PrefixLen == 0 && next.PrefixDigest == "" {
		// No turn has committed in this context yet. The state opens on the
		// empty prefix, which every next request extends, so the exclusion is
		// carried into it instead of read as a fresh context.
		next.PrefixDigest = MessagesDigest(nil, 0)
	}
	if model == "" || next.Excludes(model) || len(next.Exclusions) >= MaxPolicyExclusions {
		return next
	}
	next.Exclusions = append(next.Exclusions, PolicyExclusion{Model: model, Class: class})
	return next
}

// PolicyBoundaryDecision is a model decision taken at a schedule boundary.
// It is stored before dispatch, so a retry of the same request after the
// boundary request failed reuses the decided arm instead of deciding again:
// pathfinder's ModelRoutingState persists its boundary decision before
// dispatch, and a physical retry never re-opens the boundary. It holds only
// while the episode's turn clock is where it was decided and the request
// still extends the prefix it was decided for; any committed turn clears it.
type PolicyBoundaryDecision struct {
	Arm             int    `json:"arm"`
	TurnIndex       uint64 `json:"turn_index"`
	EpochStartTurn  uint64 `json:"epoch_start_turn"`
	CompactionCount int    `json:"compaction_count"`
	PrefixLen       int    `json:"prefix_len"`
	PrefixDigest    string `json:"prefix_digest"`
}

func (state *PolicyEpisodeState) Clone() *PolicyEpisodeState {
	if state == nil {
		return nil
	}
	cloned := *state
	cloned.Ledger = append([]PolicyLedgerEntry(nil), state.Ledger...)
	cloned.Exclusions = append([]PolicyExclusion(nil), state.Exclusions...)
	if len(cloned.Exclusions) == 0 {
		cloned.Exclusions = nil
	}
	return &cloned
}

// NewPolicyBoundaryDecision records arm as the decision for this request at
// the boundary turn resolved: turnIndex completed turns, in turn's
// compaction epoch.
func NewPolicyBoundaryDecision(
	arm int,
	turnIndex uint64,
	turn *PolicyEpisodeState,
	messages []json.RawMessage,
) *PolicyBoundaryDecision {
	return &PolicyBoundaryDecision{
		Arm: arm, TurnIndex: turnIndex, EpochStartTurn: turn.EpochStartTurn, CompactionCount: turn.CompactionCount,
		PrefixLen: len(messages), PrefixDigest: MessagesDigest(messages, len(messages)),
	}
}

// RetainedArm returns the stored arm if this decision is this request's: the
// same completed-turn count and compaction epoch, and a request that still
// extends the prefix it was decided for. A changed prefix or a moved clock
// leaves it unused.
func (boundary *PolicyBoundaryDecision) RetainedArm(
	messages []json.RawMessage,
	turnIndex uint64,
	turn *PolicyEpisodeState,
) (int, bool) {
	if boundary == nil || turn == nil {
		return -1, false
	}
	if boundary.TurnIndex != turnIndex || boundary.EpochStartTurn != turn.EpochStartTurn ||
		boundary.CompactionCount != turn.CompactionCount || boundary.PrefixLen > len(messages) ||
		MessagesDigest(messages, boundary.PrefixLen) != boundary.PrefixDigest {
		return -1, false
	}
	return boundary.Arm, true
}

// Validate refuses a stored boundary decision Next could not have produced.
func (boundary *PolicyBoundaryDecision) Validate(workerCount int) error {
	if boundary == nil {
		return nil
	}
	if boundary.Arm < 0 || boundary.Arm >= workerCount || boundary.CompactionCount < 0 ||
		boundary.PrefixLen < 1 || boundary.EpochStartTurn > boundary.TurnIndex ||
		!isLowerHex64(boundary.PrefixDigest) {
		return errors.New("ARC policy episode boundary decision is malformed")
	}
	return nil
}

// maxPolicyArmIDBytes bounds a trained arm id; pathfinder's are 64 hex.
const maxPolicyArmIDBytes = 128

// MaxPolicyExclusionModelBytes bounds an excluded model's name.
const MaxPolicyExclusionModelBytes = 128

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
	if len(state.Exclusions) > MaxPolicyExclusions {
		return errors.New("ARC policy episode exclusions exceed their bound")
	}
	for _, exclusion := range state.Exclusions {
		if exclusion.Model == "" || len(exclusion.Model) > MaxPolicyExclusionModelBytes || exclusion.Class == "" || len(exclusion.Class) > 64 {
			return errors.New("ARC policy episode exclusion is malformed")
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

// MessagesDigest hashes the first n messages in their canonical form
// (canonicalPrefixMessage), so a turn extends the recorded prefix whenever the
// conversation is the same one the client sent before, even where the client
// re-encodes a message it already sent. The bytes forwarded anywhere are
// never changed; only this comparison is.
func MessagesDigest(messages []json.RawMessage, n int) string {
	hash := sha256.New()
	for _, message := range messages[:n] {
		canonical := canonicalPrefixMessage(message)
		hash.Write([]byte(strconv.Itoa(len(canonical))))
		hash.Write([]byte{':'})
		hash.Write(canonical)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// canonicalPrefixMessage is a message as the prefix comparison sees it,
// matching the training side's extends rule (pathfinder#3214):
//
//   - cache_control is removed wherever it appears: a client moves its cache
//     breakpoints between turns without changing the conversation;
//   - a system message whose content is one {"type":"text","text":X} block,
//     with nothing else but cache_control, is the string X. Claude Code sends
//     a mid-conversation system message that way while it is the last
//     message and as a plain string once it is not, which read as a new
//     context on about 70% of real turns. Text is compared exactly; a block
//     with any other member, or more than one block, stays a list;
//   - object key order and whitespace are insignificant.
//
// A message that is not a JSON object is compared as received.
func canonicalPrefixMessage(message json.RawMessage) []byte {
	decoder := json.NewDecoder(bytes.NewReader(message))
	decoder.UseNumber()
	var value map[string]any
	if decoder.Decode(&value) != nil || value == nil {
		return message
	}
	withoutCacheControl(value)
	if role, _ := value["role"].(string); role == "system" {
		if blocks, ok := value["content"].([]any); ok && len(blocks) == 1 {
			if block, ok := blocks[0].(map[string]any); ok && len(block) == 2 && block["type"] == "text" {
				if text, ok := block["text"].(string); ok {
					value["content"] = text
				}
			}
		}
	}
	var canonical bytes.Buffer
	encoder := json.NewEncoder(&canonical)
	encoder.SetEscapeHTML(false)
	if encoder.Encode(value) != nil {
		return message
	}
	return bytes.TrimSuffix(canonical.Bytes(), []byte("\n"))
}

// withoutCacheControl removes every cache_control member under value.
func withoutCacheControl(value any) {
	switch typed := value.(type) {
	case map[string]any:
		delete(typed, "cache_control")
		for _, member := range typed {
			withoutCacheControl(member)
		}
	case []any:
		for _, element := range typed {
			withoutCacheControl(element)
		}
	}
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
	// Subagent is set for a request a harness declared a subagent's.
	Subagent *PolicySubagentSignal
}

// PolicySubagentSignal is how a subagent's request was keyed, and its link to
// the conversation that spawned it. The link is logged for analysis and
// accounting, and routes in one place only: when the gateway's turn-signal
// headers are trusted, ParentEpisodeIDHash names the session whose last
// served model is the second source of a derived hold, read after a
// two-stage package's stage_one_held_unknown refusal
// (extproc rayline_arc_derived_hold.go).
type PolicySubagentSignal struct {
	// KeySource is how the gateway chose the subagent's episode: "agent"
	// (the harness's agent id), "role", "task", or "unknown" when the
	// gateway did not say. Only "agent" makes the subagent its own episode.
	KeySource string
	// ParentEpisodeIDHash and ParentAgentIDHash hash the parent's episode id
	// and agent id as the gateway sent them; empty when it sent none.
	ParentEpisodeIDHash string
	ParentAgentIDHash   string
}

// OwnEpisode reports whether the subagent is its own episode.
func (signal *PolicySubagentSignal) OwnEpisode() bool {
	return signal != nil && signal.KeySource == SubagentKeySourceAgent
}

// SubagentKeySourceAgent is the key source that makes a subagent its own
// episode: the gateway keyed it on the harness's agent id.
const SubagentKeySourceAgent = "agent"

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
	if ordinal, summary, ok := newCompaction(previous, signals.Compaction, len(messages)); ok {
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
// a repeat, not a new context. Without one, a recognised summary is new if it
// differs from the last one recorded, because every request of a compacted
// context still opens with its summary. The same summary is also new when the
// request collapsed below the recorded prefix: two real compactions can write
// byte-identical summaries, and a carried summary rides a growing request
// while a new compaction shrinks it.
func newCompaction(
	previous *PolicyEpisodeState,
	signal *PolicyCompactionSignal,
	messageCount int,
) (int, string, bool) {
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
	if signal.SummaryDigest == "" ||
		previous != nil && signal.SummaryDigest == previous.CompactionSummary && messageCount >= previous.PrefixLen {
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
