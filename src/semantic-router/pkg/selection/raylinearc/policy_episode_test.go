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
// that rewrites history starts a new epoch with nothing attributed.
func TestPolicyTurnAttributesRepliesAndResetsOnRewrite(t *testing.T) {
	first, roles := rawMessages(t, user1)
	turn, attribution := PolicyTurn(nil, first, roles, 0)
	if turn.Epoch != 0 || len(attribution) != 0 {
		t.Fatalf("first turn: epoch %d attribution %v", turn.Epoch, attribution)
	}
	committed := turn.Next(first, "action-a", "arm-a")

	second, roles := rawMessages(t, user1, asst1, user2)
	turn, attribution = PolicyTurn(committed, second, roles, 1)
	if turn.Epoch != 0 || len(attribution) != 1 || attribution[0].Message != 1 ||
		attribution[0].ActionID != "action-a" || *attribution[0].ArmID != "arm-a" {
		t.Fatalf("extension: epoch %d attribution %+v", turn.Epoch, attribution)
	}

	rewritten, roles := rawMessages(t, `{"role":"user","content":"summary of earlier work"}`, user2)
	turn, attribution = PolicyTurn(turn.Next(second, "action-b", "arm-b"), rewritten, roles, 2)
	if turn.Epoch != 1 || turn.EpochStartTurn != 2 || len(attribution) != 0 || len(turn.Ledger) != 0 {
		t.Fatalf("rewrite: %+v attribution %v", turn, attribution)
	}
}

// Pathfinder's task_turn_compaction_v1 boundaries, row for row.
func TestModelChangeAllowedFollowsTaskTurnCompaction(t *testing.T) {
	cases := []struct {
		completed, epochStart uint64
		allowed               bool
	}{
		{0, 0, true},
		{1, 0, false},
		{4, 0, false},
		{5, 0, true},
		{6, 0, false},
		{8, 8, true},
		{9, 8, false},
		{13, 8, true},
		{14, 8, false},
	}
	for _, c := range cases {
		if got := ModelChangeAllowed(c.completed, c.epochStart); got != c.allowed {
			t.Errorf("completed=%d epochStart=%d: got %v", c.completed, c.epochStart, got)
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
	if err = state.Commit(1, 10, now); err != nil {
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
