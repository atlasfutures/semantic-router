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
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

func rawMessages(t *testing.T, messages ...string) ([]json.RawMessage, []string) {
	t.Helper()
	raw := make([]json.RawMessage, len(messages))
	roles := make([]string, len(messages))
	for index, message := range messages {
		raw[index] = json.RawMessage(message)
		var envelope struct{ Role string }
		if err := json.Unmarshal(raw[index], &envelope); err != nil {
			t.Fatal(err)
		}
		roles[index] = envelope.Role
	}
	return raw, roles
}

const (
	user1 = `{"role":"user","content":"fix the bug"}`
	asst1 = `{"role":"assistant","content":[{"type":"text","text":"reading"}]}`
	user2 = `{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"ok"}]}`
)

// A reply is attributed on the next turn at the index it lands on; a request
// that rewrites history starts a new epoch with nothing attributed, and is not
// a compaction: the schedule's boundary does not move.
func TestPolicyTurnAttributesRepliesAndResetsOnRewrite(t *testing.T) {
	first, roles := rawMessages(t, user1)
	turn, attribution, transition := PolicyTurn(nil, first, roles, 0, PolicyTurnSignals{})
	if turn.Epoch != 0 || len(attribution) != 0 || transition != PolicyTransitionStart {
		t.Fatalf("first turn: epoch %d attribution %v transition %s", turn.Epoch, attribution, transition)
	}
	committed := turn.Next(first, "action-a", "arm-a")

	second, roles := rawMessages(t, user1, asst1, user2)
	turn, attribution, transition = PolicyTurn(committed, second, roles, 1, PolicyTurnSignals{})
	if turn.Epoch != 0 || len(attribution) != 1 || attribution[0].Message != 1 ||
		attribution[0].ActionID != "action-a" || *attribution[0].ArmID != "arm-a" ||
		transition != PolicyTransitionExtends {
		t.Fatalf("extension: epoch %d attribution %+v transition %s", turn.Epoch, attribution, transition)
	}

	rewritten, roles := rawMessages(t, `{"role":"user","content":"summary of earlier work"}`, user2)
	turn, attribution, transition = PolicyTurn(turn.Next(second, "action-b", "arm-b"), rewritten, roles, 2, PolicyTurnSignals{})
	if turn.Epoch != 1 || turn.EpochStartTurn != 0 || turn.CompactionCount != 0 ||
		len(attribution) != 0 || len(turn.Ledger) != 0 || transition != PolicyTransitionPrefixBreak {
		t.Fatalf("rewrite: %+v attribution %v transition %s", turn, attribution, transition)
	}
}

// A prefix break after a compaction keeps that compaction's boundary.
func TestPolicyTurnPrefixBreakKeepsTheCompactionBoundary(t *testing.T) {
	previous := &PolicyEpisodeState{Epoch: 2, EpochStartTurn: 7, CompactionCount: 1, CompactionSummary: strings.Repeat("c", 64)}
	messages, _ := rawMessages(t, user1)
	previous = previous.Next(messages, "action-a", "arm-a")
	rewritten, roles := rawMessages(t, `{"role":"user","content":"/clear"}`)
	turn, _, transition := PolicyTurn(previous, rewritten, roles, 9, PolicyTurnSignals{})
	if transition != PolicyTransitionPrefixBreak || turn.Epoch != 3 || turn.EpochStartTurn != 7 ||
		turn.CompactionCount != 1 || turn.CompactionSummary != previous.CompactionSummary || len(turn.Ledger) != 0 {
		t.Fatalf("prefix break after compaction: %+v (%s)", turn, transition)
	}
}

func TestPolicyTurnExplicitCompactionMovesTheBoundaryOnce(t *testing.T) {
	first, _ := rawMessages(t, user1, asst1, user2)
	committed := (&PolicyEpisodeState{}).Next(first, "action-a", "arm-a")
	compacted, compactedRoles := rawMessages(t, `{"role":"user","content":"compacted"}`)
	signal := PolicyTurnSignals{Compaction: &PolicyCompactionSignal{Ordinal: 1}}
	turn, attribution, transition := PolicyTurn(committed, compacted, compactedRoles, 4, signal)
	if transition != PolicyTransitionCompaction || turn.Epoch != 1 || turn.EpochStartTurn != 4 ||
		turn.CompactionCount != 1 || len(attribution) != 0 || len(turn.Ledger) != 0 {
		t.Fatalf("compaction: %+v (%s)", turn, transition)
	}
	// The next request of the compacted context still carries the ordinal: it
	// is a repeat, so the transcript decides, and it extends.
	next := turn.Next(compacted, "action-b", "arm-b")
	grown, grownRoles := rawMessages(t, `{"role":"user","content":"compacted"}`, asst1, user2)
	turn, attribution, transition = PolicyTurn(next, grown, grownRoles, 5, signal)
	if transition != PolicyTransitionExtends || turn.Epoch != 1 || turn.EpochStartTurn != 4 ||
		turn.CompactionCount != 1 || len(attribution) != 1 {
		t.Fatalf("repeated ordinal: %+v (%s)", turn, transition)
	}
}

func TestPolicyTurnClaudeCodeSummaryIsANewCompactionOnlyWhenItChanges(t *testing.T) {
	summaryA := `{"role":"user","content":[{"type":"text","text":"<system-reminder>ctx</system-reminder>"},{"type":"text","text":"` +
		ClaudeCodeContinuationMarker + ` Summary A."}]}`
	summaryB := strings.Replace(summaryA, "Summary A.", "Summary B.", 1)
	first, _ := rawMessages(t, user1)
	state := (&PolicyEpisodeState{}).Next(first, "action-a", "arm-a")

	compactedA, rolesA := rawMessages(t, summaryA)
	signalA := PolicyTurnSignals{Compaction: &PolicyCompactionSignal{SummaryDigest: ClaudeCodeCompactionSummaryDigest(compactedA)}}
	turn, _, transition := PolicyTurn(state, compactedA, rolesA, 3, signalA)
	if transition != PolicyTransitionCompaction || turn.EpochStartTurn != 3 || turn.CompactionCount != 1 {
		t.Fatalf("first summary: %+v (%s)", turn, transition)
	}
	state = turn.Next(compactedA, "action-a", "arm-a")
	grownA, grownRoles := rawMessages(t, summaryA, asst1, user2)
	turn, _, transition = PolicyTurn(state, grownA, grownRoles, 4, signalA)
	if transition != PolicyTransitionExtends || turn.CompactionCount != 1 || turn.EpochStartTurn != 3 {
		t.Fatalf("same summary: %+v (%s)", turn, transition)
	}
	state = turn.Next(grownA, "action-a", "arm-a")
	compactedB, rolesB := rawMessages(t, summaryB)
	signalB := PolicyTurnSignals{Compaction: &PolicyCompactionSignal{SummaryDigest: ClaudeCodeCompactionSummaryDigest(compactedB)}}
	turn, _, transition = PolicyTurn(state, compactedB, rolesB, 6, signalB)
	if transition != PolicyTransitionCompaction || turn.EpochStartTurn != 6 || turn.CompactionCount != 2 {
		t.Fatalf("second summary: %+v (%s)", turn, transition)
	}
}

// A side call leaves the episode exactly as it was, even when its transcript
// does not extend the recorded prefix and it carries a compaction signal.
func TestPolicyTurnSideCallIsTransparent(t *testing.T) {
	messages, _ := rawMessages(t, user1, asst1, user2)
	previous := (&PolicyEpisodeState{Epoch: 2, EpochStartTurn: 3, CompactionCount: 1}).Next(messages, "action-a", "arm-a")
	side := PolicyTurnSignals{CallKind: PolicyCallSide, Compaction: &PolicyCompactionSignal{Ordinal: 2}}

	title, titleRoles := rawMessages(t, `{"role":"user","content":"write a title"}`)
	turn, attribution, transition := PolicyTurn(previous, title, titleRoles, 8, side)
	if transition != PolicyTransitionSideCall || !reflect.DeepEqual(turn, previous) || len(attribution) != 0 {
		t.Fatalf("non-extending side call: %+v attribution %v (%s)", turn, attribution, transition)
	}
	extended, extendedRoles := rawMessages(t, user1, asst1, user2, asst1, `{"role":"user","content":"summarize"}`)
	turn, attribution, _ = PolicyTurn(previous, extended, extendedRoles, 8, side)
	if !reflect.DeepEqual(turn, previous) || len(attribution) != 1 || attribution[0].Message != 3 {
		t.Fatalf("extending side call: %+v attribution %v", turn, attribution)
	}
}

// Pathfinder's task_turn_compaction_v1 boundaries, row for row, including
// the counted-compaction cases ModelRoutingState adds: a compaction before
// any completed turn.
func TestModelChangeAllowedFollowsTaskTurnCompaction(t *testing.T) {
	cases := []struct {
		completed, epochStart uint64
		compactions           int
		allowed               bool
	}{
		{0, 0, 0, true},
		{1, 0, 0, false},
		{4, 0, 0, false},
		{5, 0, 0, true},
		{6, 0, 0, false},
		{8, 8, 1, true},
		{9, 8, 1, false},
		{13, 8, 1, true},
		{14, 8, 1, false},
		// RoutingSchedule: a compaction before turn five does not erase it.
		{5, 3, 1, true},
		{8, 3, 1, true},
		// ModelRoutingState: a compaction before the first completed turn.
		{0, 0, 1, true},
		{5, 0, 1, true},
		{6, 0, 1, false},
	}
	for _, c := range cases {
		if got := ModelChangeAllowed(c.completed, c.epochStart, c.compactions); got != c.allowed {
			t.Errorf("completed=%d epochStart=%d compactions=%d: got %v", c.completed, c.epochStart, c.compactions, got)
		}
	}
}

func TestPolicyStateSurvivesTheEpisodeStore(t *testing.T) {
	state, err := NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	messages, _ := rawMessages(t, user1)
	state.Policy = (&PolicyEpisodeState{Epoch: 3, EpochStartTurn: 7}).Next(messages, policyTestActionA, "arm-a")
	now := time.Now().UTC()
	if err := state.Commit(1, 10, now); err != nil {
		t.Fatal(err)
	}
	payload, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	restored, _, err := unmarshalEpisodeState(payload, 2, now)
	if err != nil {
		t.Fatal(err)
	}
	// The memory backend stores a clone rather than bytes; both must keep it.
	for name, got := range map[string]*PolicyEpisodeState{
		"marshalled": restored.Policy, "cloned": cloneEpisodeState(state).Policy,
	} {
		want := state.Policy
		if got == nil || got.Epoch != want.Epoch || got.PrefixDigest != want.PrefixDigest ||
			len(got.Ledger) != 1 || got.Ledger[0] != want.Ledger[0] {
			t.Fatalf("%s: %+v, want %+v", name, got, want)
		}
	}
}

// policyTestActionA is an action id in the contract's shape: persisted policy
// state refuses any other.
const policyTestActionA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

// A persisted record from a shared store could carry indices Next never
// writes; reading one back must refuse it rather than panic slicing a request.
func TestPolicyStateRefusesMalformedPersistedRecords(t *testing.T) {
	now := time.Now().UTC()
	good := &PolicyEpisodeState{
		PrefixLen: 2, PrefixDigest: policyTestActionA,
		Ledger: []PolicyLedgerEntry{{Message: 1, ActionID: policyTestActionA, ArmID: "arm"}},
	}
	if err := good.Validate(); err != nil {
		t.Fatalf("a well-formed state was refused: %v", err)
	}
	for name, mutate := range map[string]func(*PolicyEpisodeState){
		"a negative prefix":         func(s *PolicyEpisodeState) { s.PrefixLen = -1 },
		"a negative epoch":          func(s *PolicyEpisodeState) { s.Epoch = -1 },
		"a malformed digest":        func(s *PolicyEpisodeState) { s.PrefixDigest = "nope" },
		"a negative message index":  func(s *PolicyEpisodeState) { s.Ledger[0].Message = -1 },
		"a message past the prefix": func(s *PolicyEpisodeState) { s.Ledger[0].Message = 3 },
		"a malformed action id":     func(s *PolicyEpisodeState) { s.Ledger[0].ActionID = "action-a" },
		"an oversized ledger": func(s *PolicyEpisodeState) {
			s.Ledger = make([]PolicyLedgerEntry, MaxPolicyLedgerEntries+1)
		},
	} {
		state := good.Clone()
		mutate(state)
		episode, err := NewEpisodeState(1)
		if err != nil {
			t.Fatal(err)
		}
		episode.Policy = state
		if err := validatePersistedEpisodeState(episode, now); err == nil {
			t.Fatalf("%s: accepted", name)
		}
	}
}
