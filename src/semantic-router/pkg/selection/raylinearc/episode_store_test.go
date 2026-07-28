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
	"errors"
	"strings"
	"testing"
	"time"
)

func TestMemoryEpisodeStoreSerializesAndCommits(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	store := newTestMemoryEpisodeStore(t, 4, func() time.Time {
		return now
	})
	episode := HashEpisodeID("memory-episode")
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	requireARCNoError(t, err)
	if state.TurnIndex != 0 || lease.Version() != 1 {
		t.Fatalf("initial state=%#v lease=%d", state, lease.Version())
	}

	waiting := make(chan struct{})
	acquired := make(chan *EpisodeState, 1)
	go func() {
		close(waiting)
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		nextLease, nextState, prepareErr := store.Prepare(ctx, episode, 2)
		if prepareErr != nil {
			acquired <- nil
			return
		}
		_ = store.Abort(context.Background(), nextLease)
		acquired <- nextState
	}()
	<-waiting
	select {
	case <-acquired:
		t.Fatal("same episode acquired before first lease finalized")
	case <-time.After(20 * time.Millisecond):
	}

	requireARCNoError(t, state.Commit(1, 123, now))
	requireARCNoError(t, store.Commit(
		context.Background(),
		lease,
		lease.Version(),
		state,
	))
	next := <-acquired
	if next == nil || next.TurnIndex != 1 ||
		next.PreviousArm == nil || *next.PreviousArm != 1 {
		t.Fatalf("committed state not observed: %#v", next)
	}
}

func TestMemoryEpisodeStoreDifferentEpisodesDoNotBlock(t *testing.T) {
	store := newTestMemoryEpisodeStore(t, 2, time.Now)
	first, _, err := store.Prepare(
		context.Background(),
		HashEpisodeID("first"),
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	second, _, err := store.Prepare(ctx, HashEpisodeID("second"), 2)
	if err != nil {
		t.Fatal(err)
	}
	if abortErr := store.Abort(
		context.Background(),
		first,
	); abortErr != nil {
		t.Fatal(abortErr)
	}
	if abortErr := store.Abort(
		context.Background(),
		second,
	); abortErr != nil {
		t.Fatal(abortErr)
	}
}

func TestMemoryEpisodeStoreTimeoutStaleLeaseAndCapacity(t *testing.T) {
	store := newTestMemoryEpisodeStore(t, 1, time.Now)
	episode := HashEpisodeID("held")
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, _, err := store.Prepare(ctx, episode, 2); !errors.Is(
		err,
		context.DeadlineExceeded,
	) {
		t.Fatalf("same-episode timeout error = %v", err)
	}
	if _, _, err := store.Prepare(
		context.Background(),
		HashEpisodeID("capacity"),
		2,
	); !errors.Is(err, ErrEpisodeCapacity) {
		t.Fatalf("capacity error = %v", err)
	}
	stale := lease
	stale.ownerToken = strings.Repeat("0", len(stale.ownerToken))
	if err := store.Commit(
		context.Background(),
		stale,
		stale.Version(),
		state,
	); !errors.Is(err, ErrEpisodeLeaseLost) {
		t.Fatalf("stale commit error = %v", err)
	}
	if err := store.Abort(context.Background(), lease); err != nil {
		t.Fatal(err)
	}
	if err := store.Abort(context.Background(), lease); err != nil {
		t.Fatalf("idempotent abort error = %v", err)
	}
}

func TestMemoryEpisodeStoreReapsIdleEntries(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	store := newTestMemoryEpisodeStore(t, 1, func() time.Time {
		return now
	})
	first, _, err := store.Prepare(
		context.Background(),
		HashEpisodeID("old"),
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	if abortErr := store.Abort(
		context.Background(),
		first,
	); abortErr != nil {
		t.Fatal(abortErr)
	}
	now = now.Add(2 * time.Minute)
	second, _, err := store.Prepare(
		context.Background(),
		HashEpisodeID("new"),
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	if abortErr := store.Abort(
		context.Background(),
		second,
	); abortErr != nil {
		t.Fatal(abortErr)
	}
}

func TestEpisodeStateWireRejectsFutureAndUnknownFields(t *testing.T) {
	now := time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	state.Warmth[0] = &WorkerWarmth{
		LastUsed:        now.Add(maxFutureClockSkew + time.Millisecond),
		LastInputTokens: 1,
	}
	if _, err := marshalEpisodeState(state, 1, now); err == nil {
		t.Fatal("future timestamp accepted")
	}
	payload := []byte(
		`{"schema_version":"rayline.arc.episode-state.v1",` +
			`"version":1,"previous_arm":null,"turn_index":0,` +
			`"warmth":[null],"secret":"no"}`,
	)
	if _, _, err := unmarshalEpisodeState(payload, 1, now); err == nil {
		t.Fatal("unknown field accepted")
	}
}

func newTestMemoryEpisodeStore(
	t *testing.T,
	maxEpisodes int,
	now func() time.Time,
) *MemoryEpisodeStore {
	t.Helper()
	store, err := NewMemoryEpisodeStore(MemoryEpisodeStoreConfig{
		MaxEpisodes: maxEpisodes,
		IdleTTL:     time.Minute,
		Now:         now,
	})
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func requireARCNoError(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
