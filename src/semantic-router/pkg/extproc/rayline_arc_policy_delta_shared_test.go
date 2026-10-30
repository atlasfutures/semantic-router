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

package extproc

import (
	"bytes"
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// fakeStagedTurnStore is the record store a cell's instances share, fenced
// by turn as the Redis store is.
type fakeStagedTurnStore struct {
	mu      sync.Mutex
	records map[string][]byte
	turns   map[string]uint64
	failing bool
	// delay holds every call that long, past VSR's wait.
	delay time.Duration
}

func newFakeStagedTurnStore() *fakeStagedTurnStore {
	return &fakeStagedTurnStore{records: map[string][]byte{}, turns: map[string]uint64{}}
}

func (store *fakeStagedTurnStore) enter() error {
	store.mu.Lock()
	delay, failing := store.delay, store.failing
	store.mu.Unlock()
	time.Sleep(delay)
	if failing {
		return errors.New("unavailable")
	}
	return nil
}

func (store *fakeStagedTurnStore) LoadStagedTurn(_ context.Context, episode string) ([]byte, error) {
	if err := store.enter(); err != nil {
		return nil, err
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	return store.records[episode], nil
}

func (store *fakeStagedTurnStore) StoreStagedTurn(_ context.Context, episode string, turn uint64, record []byte) error {
	if err := store.enter(); err != nil {
		return err
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if _, ok := store.records[episode]; ok && store.turns[episode] > turn {
		return nil
	}
	store.records[episode], store.turns[episode] = bytes.Clone(record), turn
	return nil
}

func (store *fakeStagedTurnStore) DeleteStagedTurn(_ context.Context, episode string, turn uint64, match []byte) error {
	if err := store.enter(); err != nil {
		return err
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.turns[episode] > turn || (match != nil && !bytes.Equal(store.records[episode], match)) {
		return nil
	}
	delete(store.records, episode)
	delete(store.turns, episode)
	return nil
}

// otherInstance is the next turn served by another cell instance: the same
// episode store and service, a process with no staged turns of its own.
func otherInstance(fixture *deltaFixture) {
	fixture.scorer().staged = newPolicyStagedTurns()
}

func deltaHeldTurn(t *testing.T, fixture *deltaFixture) string {
	t.Helper()
	_, first := fixture.turn(t, policyTestRequest(t, deltaUser))
	complete(t, first, anthropicReply(t, deltaReplyContent))
	otherInstance(fixture)
	result, second := fixture.turn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult))
	complete(t, second, deltaReply(t))
	return result.RaylineARC.PolicyDecisionMode
}

// A held turn served by another instance finds the staged turn the previous
// one recorded beside the episode, and is sent as a delta.
func TestDeltaSharedRecordExtendsOnAnotherInstance(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	shared := newFakeStagedTurnStore()
	fixture.shared = shared
	if mode := deltaHeldTurn(t, fixture); mode != policyDecisionModeDelta {
		t.Fatalf("held turn on another instance went %s, want delta", mode)
	}
	record, ok := decodePolicyStagedTurn(shared.records[fixture.episode])
	if !ok || record.turnIndex != 2 || record.count != 4 {
		t.Fatalf("shared record after turn 2: %+v %v", record, ok)
	}
}

// Without a shared record, or when it cannot be read, the other instance
// sends the turn in full, as before.
func TestDeltaWithoutSharedRecordGoesInFull(t *testing.T) {
	for _, tc := range []struct {
		name  string
		store func() raylinearc.EpisodeStagedTurnStore
	}{
		{"no_store", func() raylinearc.EpisodeStagedTurnStore { return nil }},
		{"unreadable", func() raylinearc.EpisodeStagedTurnStore {
			store := newFakeStagedTurnStore()
			store.failing = true
			return store
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
			fixture.shared = tc.store()
			if mode := deltaHeldTurn(t, fixture); mode != policyDecisionModeSync {
				t.Fatalf("held turn went %s, want sync", mode)
			}
		})
	}
}

// A record for another turn, or one that does not decode, is not used.
func TestDeltaSharedRecordForAnotherTurnIsIgnored(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func([]byte) []byte
	}{
		{"stale_turn", func(record []byte) []byte {
			turn, _ := decodePolicyStagedTurn(record)
			turn.turnIndex++
			return encodePolicyStagedTurn(turn)
		}},
		{"undecodable", func([]byte) []byte { return []byte(`{"v":2}`) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
			shared := newFakeStagedTurnStore()
			fixture.shared = shared
			_, first := fixture.turn(t, policyTestRequest(t, deltaUser))
			complete(t, first, anthropicReply(t, deltaReplyContent))
			shared.records[fixture.episode] = tc.mutate(shared.records[fixture.episode])
			otherInstance(fixture)
			result, second := fixture.turn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult))
			complete(t, second, deltaReply(t))
			if mode := result.RaylineARC.PolicyDecisionMode; mode != policyDecisionModeSync {
				t.Fatalf("held turn went %s, want sync", mode)
			}
		})
	}
}

// A turn that stages nothing deletes the shared record, so no instance
// builds on the turn before it.
func TestDeltaTurnWithoutAppendClearsSharedRecord(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	shared := newFakeStagedTurnStore()
	fixture.shared = shared
	_, first := fixture.turn(t, policyTestRequest(t, deltaUser))
	complete(t, first, anthropicReply(t, deltaReplyContent))
	if _, ok := shared.records[fixture.episode]; !ok {
		t.Fatal("turn 1 left no shared record")
	}
	_, second := fixture.turn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult))
	complete(t, second, []byte(`{"type":"message","role":"assistant","content":[{"type":"redacted_thinking"}]}`))
	if _, ok := shared.records[fixture.episode]; ok {
		t.Fatal("a turn that appended nothing kept the shared record")
	}
}

func TestPolicyStagedRecordRoundTrip(t *testing.T) {
	turn := policyStagedTurn{
		turnIndex: 7, contextEpoch: "e", requestFormat: policyFormatAnthropic, count: 4,
		prefixDigest: "p", digest: "d", appendedFrom: 2, historyDigest: "h",
	}
	if decoded, ok := decodePolicyStagedTurn(encodePolicyStagedTurn(turn)); !ok || decoded != turn {
		t.Fatalf("round trip: %+v %v", decoded, ok)
	}
	unsettled := policyStagedTurn{turnIndex: 3, unsettled: true}
	if decoded, ok := decodePolicyStagedTurn(encodePolicyStagedTurn(unsettled)); !ok || decoded != unsettled {
		t.Fatalf("unsettled round trip: %+v %v", decoded, ok)
	}
	for _, bad := range []string{`{}`, `{"v":1,"turn":1,"count":0,"prefix":"p","digest":"d"}`,
		`{"v":1,"turn":1,"count":2,"prefix":"","digest":"d"}`, `{"v":1,"turn":1,"count":2,"prefix":"p","digest":"d","from":3}`, `nope`} {
		if _, ok := decodePolicyStagedTurn([]byte(bad)); ok {
			t.Errorf("%s decoded", bad)
		}
	}
}

// A store slower than VSR's wait costs the turn a full send, never its
// decision or its deadline.
func TestDeltaSlowSharedRecordIsAMiss(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	shared := newFakeStagedTurnStore()
	fixture.shared = shared
	_, first := fixture.turn(t, policyTestRequest(t, deltaUser))
	complete(t, first, anthropicReply(t, deltaReplyContent))
	otherInstance(fixture)
	shared.mu.Lock()
	shared.delay = 4 * policySharedStagedTurnTimeout
	shared.mu.Unlock()
	started := time.Now()
	result, second := fixture.turn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult))
	if waited := time.Since(started); waited > 3*policySharedStagedTurnTimeout {
		t.Fatalf("the turn waited %v on the store", waited)
	}
	if mode := result.RaylineARC.PolicyDecisionMode; mode != policyDecisionModeSync {
		t.Fatalf("held turn went %s, want sync", mode)
	}
	started = time.Now()
	complete(t, second, deltaReply(t))
	if waited := time.Since(started); waited > 4*policySharedStagedTurnTimeout {
		t.Fatalf("the commit waited %v on the store", waited)
	}
}

// A turn that lost its lease, aborting after the turn after it committed,
// leaves that turn's record.
func TestDeltaAbortLeavesALaterTurnsRecord(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	shared := newFakeStagedTurnStore()
	fixture.shared = shared
	_, first := fixture.turn(t, policyTestRequest(t, deltaUser))
	complete(t, first, anthropicReply(t, deltaReplyContent))
	_, stale := fixture.turn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult))
	// The turn after it commits a record of turn 2 meanwhile.
	later := encodePolicyStagedTurn(policyStagedTurn{
		turnIndex: 2, requestFormat: policyFormatAnthropic, count: 4, prefixDigest: "p", digest: "d", appendedFrom: 3,
	})
	if err := shared.StoreStagedTurn(context.Background(), fixture.episode, 2, later); err != nil {
		t.Fatal(err)
	}
	if err := stale.RaylineARCTransaction.abort(context.Background(), "test"); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(shared.records[fixture.episode], later) {
		t.Fatal("an aborted turn deleted a later turn's record")
	}
}
