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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// fakeStagedTurnStore is the record store a cell's instances share.
type fakeStagedTurnStore struct {
	mu      sync.Mutex
	records map[string][]byte
	failing bool
}

func newFakeStagedTurnStore() *fakeStagedTurnStore {
	return &fakeStagedTurnStore{records: map[string][]byte{}}
}

func (store *fakeStagedTurnStore) LoadStagedTurn(_ context.Context, episode string) ([]byte, error) {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.failing {
		return nil, errors.New("unavailable")
	}
	return store.records[episode], nil
}

func (store *fakeStagedTurnStore) StoreStagedTurn(_ context.Context, episode string, record []byte) error {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.failing {
		return errors.New("unavailable")
	}
	store.records[episode] = bytes.Clone(record)
	return nil
}

func (store *fakeStagedTurnStore) DeleteStagedTurn(_ context.Context, episode string, match []byte) error {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.failing {
		return errors.New("unavailable")
	}
	if match == nil || bytes.Equal(store.records[episode], match) {
		delete(store.records, episode)
	}
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
