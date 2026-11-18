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
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
)

var (
	pendingActionA = strings.Repeat("a", 64)
	pendingActionB = strings.Repeat("b", 64)
	pendingNow     = time.Date(2026, 11, 18, 12, 0, 0, 0, time.UTC)
)

const (
	pendingTask   = `{"role":"user","content":"task"}`
	pendingReply1 = `{"role":"assistant","content":"done one"}`
	pendingAsk2   = `{"role":"user","content":"next"}`
)

// executedEpisode is an episode after one served turn: the task, answered by
// arm 0 with action A.
func executedEpisode(t *testing.T) *EpisodeState {
	t.Helper()
	state, err := NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	messages, roles := rawMessages(t, pendingTask)
	turn, _, _ := PolicyTurn(nil, messages, roles, 0, PolicyTurnSignals{CallKind: PolicyCallMain})
	state.Policy = turn.Next(messages, pendingActionA, "arm-a")
	if err := state.Commit(0, 10, pendingNow.Add(-time.Minute)); err != nil {
		t.Fatal(err)
	}
	state.Warmth[1] = &cachecontext.WorkerWarmth{
		LastUsed: pendingNow.Add(-2 * time.Minute), LastInputTokens: 5,
		Call: &cachecontext.CallCache{ActionID: pendingActionB, CompletedAt: pendingNow.Add(-2 * time.Minute)},
	}
	return state
}

// withPendingLookup is the episode after a lookup over messages decided arm
// 1 (action B), as the lookup stores it.
func withPendingLookup(t *testing.T, state *EpisodeState, messages ...string) *EpisodeState {
	t.Helper()
	raw, roles := rawMessages(t, messages...)
	digests := NewPrefixDigests(raw)
	turn, _, _ := PolicyTurnFrom(state.Policy, digests, roles, state.TurnIndex, PolicyTurnSignals{CallKind: PolicyCallMain})
	next := turn.NextWithPricesFrom(digests, pendingActionB, "arm-b", nil)
	stored := cloneEpisodeState(state)
	stored.Pending = NewPendingDecision("rte_1", 1, "vendor/b", state.TurnIndex, 42, digests, next, pendingNow)
	return stored
}

// The next turn is told from a repeat of the same turn by the messages: a
// reply right after the decided messages is the next turn, anything else
// (the same messages, an edited or appended last message, a cut history)
// drops the decision.
func TestResolvePendingTellsTheNextTurnFromARepeat(t *testing.T) {
	stored := withPendingLookup(t, executedEpisode(t), pendingTask, pendingReply1, pendingAsk2)
	cases := map[string]struct {
		messages []string
		want     string
	}{
		"identical repeat":      {[]string{pendingTask, pendingReply1, pendingAsk2}, PendingDropped},
		"edited last message":   {[]string{pendingTask, pendingReply1, `{"role":"user","content":"next, edited"}`}, PendingDropped},
		"appended user message": {[]string{pendingTask, pendingReply1, pendingAsk2, `{"role":"user","content":"also"}`}, PendingDropped},
		"truncated history":     {[]string{pendingTask}, PendingDropped},
		"next turn":             {[]string{pendingTask, pendingReply1, pendingAsk2, `{"role":"assistant","content":"two"}`, `{"role":"user","content":"three"}`}, PendingInferred},
		"tool round": {[]string{
			pendingTask, pendingReply1, pendingAsk2,
			`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"ls","input":{}}]}`,
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"a.go"}]}`,
		}, PendingInferred},
		"reply only, no new message": {[]string{pendingTask, pendingReply1, pendingAsk2, `{"role":"assistant","content":"two"}`}, PendingInferred},
	}
	for name, tc := range cases {
		raw, roles := rawMessages(t, tc.messages...)
		resolved, resolution := ResolvePending(stored, NewPrefixDigests(raw), roles)
		if resolution != tc.want {
			t.Fatalf("%s: resolution %q, want %q", name, resolution, tc.want)
		}
		if resolved.Pending != nil || stored.Pending == nil {
			t.Fatalf("%s: the resolved state holds a pending decision, or the stored one was changed", name)
		}
	}
}

// A dropped decision leaves the trajectory exactly as executed: the policy
// sees none of it.
func TestResolvePendingDropLeavesTheExecutedEpisode(t *testing.T) {
	executed := executedEpisode(t)
	stored := withPendingLookup(t, executed, pendingTask, pendingReply1, pendingAsk2)
	raw, roles := rawMessages(t, pendingTask, pendingReply1, pendingAsk2)
	resolved, _ := ResolvePending(stored, NewPrefixDigests(raw), roles)
	if !reflect.DeepEqual(resolved, executed) {
		t.Fatalf("a repeat ask changed the trajectory:\n got %+v\nwant %+v", resolved, executed)
	}
}

// The next turn records the decision as a turn that ran: the ledger entry,
// marked inferred, the prefix, the previous arm, the turn count and the arm's
// warmth, keeping the arm's last call. Its attribution then names the reply.
func TestResolvePendingInfersTheDecisionRan(t *testing.T) {
	executed := executedEpisode(t)
	stored := withPendingLookup(t, executed, pendingTask, pendingReply1, pendingAsk2)
	raw, roles := rawMessages(t, pendingTask, pendingReply1, pendingAsk2, `{"role":"assistant","content":"two"}`, `{"role":"user","content":"three"}`)
	digests := NewPrefixDigests(raw)
	resolved, resolution := ResolvePending(stored, digests, roles)
	if resolution != PendingInferred {
		t.Fatalf("resolution %q", resolution)
	}
	if resolved.TurnIndex != executed.TurnIndex+1 || resolved.PreviousArm == nil || *resolved.PreviousArm != 1 ||
		resolved.ServedModel != "vendor/b" || resolved.PolicyBoundary != nil {
		t.Fatalf("trajectory turn %d previous %v served %q", resolved.TurnIndex, resolved.PreviousArm, resolved.ServedModel)
	}
	warmth := resolved.Warmth[1]
	if warmth.LastInputTokens != 42 || !warmth.LastUsed.Equal(pendingNow) || warmth.Call == nil || warmth.Call.ActionID != pendingActionB {
		t.Fatalf("warmth %+v", warmth)
	}
	policy := resolved.Policy
	if policy.Epoch != executed.Policy.Epoch || policy.PrefixLen != 3 || policy.PrefixDigest != digests.Digest(3) || len(policy.Ledger) != 2 {
		t.Fatalf("policy %+v", policy)
	}
	if entry := policy.Ledger[1]; entry.Message != 3 || entry.ActionID != pendingActionB || entry.Basis != PolicyLedgerBasisInferred {
		t.Fatalf("inferred entry %+v", entry)
	}
	if policy.Ledger[0].Basis != "" {
		t.Fatal("the served entry was marked")
	}
	turn, attribution, transition := PolicyTurnFrom(resolved.Policy, digests, roles, resolved.TurnIndex, PolicyTurnSignals{CallKind: PolicyCallMain})
	if transition != PolicyTransitionExtends || turn.Epoch != policy.Epoch || len(attribution) != 2 || attribution[1].Message != 3 {
		t.Fatalf("next turn %s attribution %+v", transition, attribution)
	}
}

// A decision that opened a new context (here a rewritten history) is
// inferred into that context, with an empty ledger and the full offer, not
// appended to the stored one.
func TestResolvePendingInfersIntoTheDecisionsContext(t *testing.T) {
	executed := executedEpisode(t)
	executed.Policy = executed.Policy.WithExclusion("vendor/x", "refused")
	stored := withPendingLookup(t, executed, `{"role":"user","content":"other task"}`)
	if stored.Pending.Policy.Epoch != executed.Policy.Epoch+1 {
		t.Fatalf("the lookup did not open a new context: %+v", stored.Pending.Policy)
	}
	raw, roles := rawMessages(t, `{"role":"user","content":"other task"}`, `{"role":"assistant","content":"ok"}`, pendingAsk2)
	resolved, resolution := ResolvePending(stored, NewPrefixDigests(raw), roles)
	if resolution != PendingInferred || resolved.Policy.Epoch != executed.Policy.Epoch+1 ||
		len(resolved.Policy.Ledger) != 1 || resolved.Policy.Ledger[0].Message != 1 || len(resolved.Policy.Exclusions) != 0 {
		t.Fatalf("resolution %q policy %+v", resolution, resolved.Policy)
	}
}

// A decision whose turn count is not the episode's is stale and dropped.
func TestResolvePendingDropsAStaleDecision(t *testing.T) {
	stored := withPendingLookup(t, executedEpisode(t), pendingTask, pendingReply1, pendingAsk2)
	stored.TurnIndex++
	raw, roles := rawMessages(t, pendingTask, pendingReply1, pendingAsk2, `{"role":"assistant","content":"two"}`)
	resolved, resolution := ResolvePending(stored, NewPrefixDigests(raw), roles)
	if resolution != PendingStale || resolved.Pending != nil || resolved.TurnIndex != stored.TurnIndex {
		t.Fatalf("resolution %q", resolution)
	}
	if _, resolution := ResolvePending(executedEpisode(t), NewPrefixDigests(raw), roles); resolution != PendingNone {
		t.Fatalf("no pending decision resolved as %q", resolution)
	}
}

// Any committed turn ends the pending decision.
func TestCommitClearsThePendingDecision(t *testing.T) {
	stored := withPendingLookup(t, executedEpisode(t), pendingTask, pendingReply1, pendingAsk2)
	if err := stored.Commit(0, 1, pendingNow); err != nil || stored.Pending != nil {
		t.Fatalf("commit left %+v (%v)", stored.Pending, err)
	}
}

// A long caller-minted route id is not kept, rather than making the record
// unwritable.
func TestNewPendingDecisionBoundsTheRouteID(t *testing.T) {
	raw, _ := rawMessages(t, pendingTask)
	long := NewPendingDecision(strings.Repeat("r", maxPendingRouteIDBytes+1), 0, "", 0, 0, NewPrefixDigests(raw), nil, pendingNow)
	if long.RouteID != "" || long.Validate(2) != nil {
		t.Fatalf("pending %+v", long)
	}
}

// The record carries a pending decision or an inferred entry under v11 only;
// a record with neither keeps its earlier bytes; a v10 reader's schema
// refuses a v11 record, and a mislabelled one is refused.
func TestEpisodeStateWireV11CarriesPendingDecisions(t *testing.T) {
	executed := executedEpisode(t)
	before, err := marshalEpisodeState(executed, 1, pendingNow)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(before, []byte(episodeStateSchemaV11)) || bytes.Contains(before, []byte("pending")) ||
		bytes.Contains(before, []byte("basis")) {
		t.Fatalf("a record without a pending decision changed: %s", before)
	}
	stored := withPendingLookup(t, executed, pendingTask, pendingReply1, pendingAsk2)
	v11, err := marshalEpisodeState(stored, 2, pendingNow)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(v11, []byte(episodeStateSchemaV11)) {
		t.Fatalf("pending record not v11: %.120s", v11)
	}
	decoded, _, err := unmarshalEpisodeState(v11, 2, pendingNow)
	if err != nil || !reflect.DeepEqual(decoded.Pending, stored.Pending) {
		t.Fatalf("pending did not round-trip: %+v (%v)", decoded.Pending, err)
	}
	raw, roles := rawMessages(t, pendingTask, pendingReply1, pendingAsk2, `{"role":"assistant","content":"two"}`)
	inferred, _ := ResolvePending(stored, NewPrefixDigests(raw), roles)
	marked, err := marshalEpisodeState(inferred, 3, pendingNow)
	if err != nil || !bytes.Contains(marked, []byte(episodeStateSchemaV11)) || !bytes.Contains(marked, []byte(`"basis":"inferred"`)) {
		t.Fatalf("an inferred entry was not written under v11: %v", err)
	}
	if decoded, _, err := unmarshalEpisodeState(marked, 2, pendingNow); err != nil || !reflect.DeepEqual(decoded.Policy, inferred.Policy) {
		t.Fatalf("inferred policy did not round-trip (%v)", err)
	}
	var earlier struct {
		SchemaVersion string `json:"schema_version"`
	}
	if err := json.Unmarshal(before, &earlier); err != nil {
		t.Fatal(err)
	}
	for name, payload := range map[string][]byte{
		"v11 labelled earlier": bytes.Replace(v11, []byte(episodeStateSchemaV11), []byte(earlier.SchemaVersion), 1),
		"earlier labelled v11": bytes.Replace(before, []byte(earlier.SchemaVersion), []byte(episodeStateSchemaV11), 1),
	} {
		if _, _, err := unmarshalEpisodeState(payload, 2, pendingNow); err == nil {
			t.Fatalf("%s was accepted", name)
		}
	}
}

// A stored pending decision NewPendingDecision could not have written is
// refused, as any malformed record is.
func TestEpisodeStateWireRefusesAMalformedPendingDecision(t *testing.T) {
	stored := withPendingLookup(t, executedEpisode(t), pendingTask, pendingReply1, pendingAsk2)
	for name, mutate := range map[string]func(*PendingDecision){
		"arm out of range":     func(p *PendingDecision) { p.Arm = 2 },
		"bad digest":           func(p *PendingDecision) { p.PrefixDigest = "x" },
		"no messages":          func(p *PendingDecision) { p.PrefixLen = 0 },
		"entry not the reply":  func(p *PendingDecision) { p.Policy.Entry.Message = 1 },
		"entry carries basis":  func(p *PendingDecision) { p.Policy.Entry.Basis = PolicyLedgerBasisInferred },
		"long route id":        func(p *PendingDecision) { p.RouteID = strings.Repeat("r", maxPendingRouteIDBytes+1) },
		"no decision time":     func(p *PendingDecision) { p.DecidedUnixMS = 0 },
		"epoch after its turn": func(p *PendingDecision) { p.Policy.EpochStartTurn = p.TurnIndex + 1 },
	} {
		state := cloneEpisodeState(stored)
		mutate(state.Pending)
		if _, err := marshalEpisodeState(state, 1, pendingNow); err == nil {
			t.Fatalf("%s was written", name)
		}
	}
	state := cloneEpisodeState(stored)
	state.Pending = nil
	state.Policy.Ledger[0].Basis = "reported"
	if _, err := marshalEpisodeState(state, 1, pendingNow); err == nil {
		t.Fatal("an unknown basis was written")
	}
	var payload map[string]any
	good, _ := marshalEpisodeState(stored, 1, pendingNow)
	if err := json.Unmarshal(good, &payload); err != nil || payload["pending"].(map[string]any)["route_id"] != "rte_1" {
		t.Fatalf("pending wire %v", payload["pending"])
	}
}

// A binding's model id has no length bound of its own on every cell, so a
// pending decision keeps a long one rather than making the record
// unwritable.
func TestPendingDecisionKeepsALongModel(t *testing.T) {
	stored := withPendingLookup(t, executedEpisode(t), pendingTask, pendingReply1, pendingAsk2)
	stored.Pending.Model = strings.Repeat("m", 300)
	if _, err := marshalEpisodeState(stored, 1, pendingNow); err != nil {
		t.Fatalf("a long model made the record unwritable: %v", err)
	}
}
