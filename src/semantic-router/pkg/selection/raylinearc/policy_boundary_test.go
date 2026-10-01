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
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"
	"time"
)

func boundaryRequest(t *testing.T, extra ...string) ([]json.RawMessage, *PolicyEpisodeState) {
	t.Helper()
	messages, roles := rawMessages(t, append([]string{`{"role":"user","content":"task"}`}, extra...)...)
	turn, _, _ := PolicyTurn(nil, messages, roles, 0, PolicyTurnSignals{CallKind: PolicyCallUnknown})
	return messages, turn
}

// A stored boundary decision is this request's only while the clock is where
// it was decided and the request still extends the prefix it was decided for.
func TestPolicyBoundaryDecisionHoldsOnlyForItsRequest(t *testing.T) {
	messages, turn := boundaryRequest(t)
	boundary := NewPolicyBoundaryDecision(1, 0, turn, messages)
	if arm, ok := boundary.RetainedArm(messages, 0, turn); !ok || arm != 1 {
		t.Fatalf("the same request did not reuse its boundary decision: %d %v", arm, ok)
	}
	longer, _ := boundaryRequest(t, `{"role":"assistant","content":"partial"}`)
	if _, ok := boundary.RetainedArm(longer, 0, turn); !ok {
		t.Fatal("a request extending the decided prefix did not reuse the decision")
	}
	rewritten, _ := rawMessages(t, `{"role":"user","content":"another task"}`)
	if _, ok := boundary.RetainedArm(rewritten, 0, turn); ok {
		t.Fatal("a changed prefix reused the boundary decision")
	}
	if _, ok := boundary.RetainedArm(messages, 1, turn); ok {
		t.Fatal("a moved turn clock reused the boundary decision")
	}
	compacted := turn.Clone()
	compacted.CompactionCount, compacted.EpochStartTurn = 1, 0
	if _, ok := boundary.RetainedArm(messages, 0, compacted); ok {
		t.Fatal("a new compaction epoch reused the boundary decision")
	}
	var absent *PolicyBoundaryDecision
	if _, ok := absent.RetainedArm(messages, 0, turn); ok {
		t.Fatal("no stored decision was reused")
	}
}

// Any committed turn clears the stored decision, and a stored one survives
// persistence and is validated on the way back.
func TestPolicyBoundaryDecisionPersistsUntilACommit(t *testing.T) {
	messages, turn := boundaryRequest(t)
	state, _ := NewEpisodeState(2)
	state.PolicyBoundary = NewPolicyBoundaryDecision(1, 0, turn, messages)
	now := time.Now().UTC()
	payload, err := marshalEpisodeState(state, 3, now)
	requireARCNoError(t, err)
	decoded, _, err := unmarshalEpisodeState(payload, 2, now)
	requireARCNoError(t, err)
	if !reflect.DeepEqual(decoded.PolicyBoundary, state.PolicyBoundary) {
		t.Fatalf("boundary decision did not round-trip: %+v", decoded.PolicyBoundary)
	}
	requireARCNoError(t, decoded.Commit(1, 10, now))
	if decoded.PolicyBoundary != nil {
		t.Fatal("a committed turn kept the boundary decision")
	}
	state.PolicyBoundary.Arm = 2
	if _, err := marshalEpisodeState(state, 3, now); err == nil {
		t.Fatal("a boundary decision on an arm outside the pool was persisted")
	}
}

// Stage writes under the held lease and keeps it: the lease's own commit
// still lands, and an abort leaves the staged state for the next prepare.
func TestEpisodeStoresStageUnderTheLease(t *testing.T) {
	stores := map[string]func(*testing.T) EpisodeStore{
		"memory": func(t *testing.T) EpisodeStore {
			return newTestMemoryEpisodeStore(t, 4, func() time.Time { return time.Now().UTC() })
		},
	}
	if address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR"); address != "" {
		stores["redis"] = func(t *testing.T) EpisodeStore {
			return newTestRedisEpisodeStore(t, address, "test:rayline-arc-stage:"+HashEpisodeID(t.Name()+time.Now().String())+":", time.Minute)
		}
	}
	for name, open := range stores {
		t.Run(name, func(t *testing.T) {
			store := open(t)
			stager := store.(EpisodeStateStager)
			episode := HashEpisodeID("stage-" + name)
			ctx := context.Background()
			messages, turn := boundaryRequest(t)

			lease, state, err := store.Prepare(ctx, episode, 2)
			requireARCNoError(t, err)
			staged := cloneEpisodeState(state)
			staged.PolicyBoundary = NewPolicyBoundaryDecision(1, 0, turn, messages)
			requireARCNoError(t, stager.Stage(ctx, lease, staged))
			requireARCNoError(t, store.Abort(ctx, lease))

			lease, state, err = store.Prepare(ctx, episode, 2)
			requireARCNoError(t, err)
			if !reflect.DeepEqual(state.PolicyBoundary, staged.PolicyBoundary) {
				t.Fatalf("an aborted turn lost its staged boundary decision: %+v", state.PolicyBoundary)
			}
			requireARCNoError(t, stager.Stage(ctx, lease, staged))
			requireARCNoError(t, state.Commit(1, 10, time.Now().UTC()))
			requireARCNoError(t, store.Commit(ctx, lease, lease.Version(), state))
			if lost := stager.Stage(ctx, lease, staged); !errors.Is(lost, ErrEpisodeLeaseLost) {
				t.Fatalf("a stage after the lease was released = %v, want lease lost", lost)
			}

			lease, state, err = store.Prepare(ctx, episode, 2)
			requireARCNoError(t, err)
			if state.TurnIndex != 1 || state.PolicyBoundary != nil {
				t.Fatalf("after the commit: turn %d boundary %+v", state.TurnIndex, state.PolicyBoundary)
			}
			requireARCNoError(t, store.Abort(ctx, lease))
		})
	}
}
