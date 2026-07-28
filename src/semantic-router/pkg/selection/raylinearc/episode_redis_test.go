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
	"os"
	"testing"
	"time"
)

func TestRedisEpisodeStoreFencingPersistenceAndRenewal(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	prefix := "test:rayline-arc:" +
		HashEpisodeID(t.Name()+time.Now().String()) + ":"
	store := newTestRedisEpisodeStore(t, address, prefix, 120*time.Millisecond)
	episode := HashEpisodeID("redis-episode")

	firstLease, state, err := store.Prepare(
		context.Background(),
		episode,
		2,
	)
	requireARCNoError(t, err)
	requireARCNoError(t, store.Renew(
		context.Background(),
		firstLease,
	))
	requireARCNoError(t, state.Commit(
		1,
		222,
		time.Now().UTC(),
	))
	requireARCNoError(t, store.Commit(
		context.Background(),
		firstLease,
		firstLease.Version(),
		state,
	))

	restarted := newTestRedisEpisodeStore(
		t,
		address,
		prefix,
		120*time.Millisecond,
	)
	secondLease, persisted, err := restarted.Prepare(
		context.Background(),
		episode,
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	if secondLease.Version() <= firstLease.Version() ||
		persisted.TurnIndex != 1 ||
		persisted.PreviousArm == nil ||
		*persisted.PreviousArm != 1 {
		t.Fatalf(
			"lease=%d state=%#v",
			secondLease.Version(),
			persisted,
		)
	}
	if abortErr := restarted.Abort(
		context.Background(),
		secondLease,
	); abortErr != nil {
		t.Fatal(abortErr)
	}
	if abortErr := restarted.Abort(
		context.Background(),
		secondLease,
	); abortErr != nil {
		t.Fatalf("idempotent Redis abort error = %v", abortErr)
	}
}

func TestRedisEpisodeStoreContentionAndStaleCommit(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	prefix := "test:rayline-arc:" +
		HashEpisodeID(t.Name()+time.Now().String()) + ":"
	store := newTestRedisEpisodeStore(t, address, prefix, 80*time.Millisecond)
	episode := HashEpisodeID("contended")
	staleLease, staleState, err := store.Prepare(
		context.Background(),
		episode,
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	timeoutContext, cancel := context.WithTimeout(
		context.Background(),
		25*time.Millisecond,
	)
	defer cancel()
	if _, _, contentionErr := store.Prepare(
		timeoutContext,
		episode,
		2,
	); !errors.Is(contentionErr, context.DeadlineExceeded) {
		t.Fatalf("contention error = %v", contentionErr)
	}
	differentLease, _, err := store.Prepare(
		context.Background(),
		HashEpisodeID("different"),
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	if abortErr := store.Abort(
		context.Background(),
		differentLease,
	); abortErr != nil {
		t.Fatal(abortErr)
	}

	time.Sleep(100 * time.Millisecond)
	currentLease, _, err := store.Prepare(
		context.Background(),
		episode,
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	if commitErr := store.Commit(
		context.Background(),
		staleLease,
		staleLease.Version(),
		staleState,
	); !errors.Is(commitErr, ErrEpisodeLeaseLost) {
		t.Fatalf("stale commit error = %v", commitErr)
	}
	if abortErr := store.Abort(
		context.Background(),
		currentLease,
	); abortErr != nil {
		t.Fatal(abortErr)
	}
}

func newTestRedisEpisodeStore(
	t *testing.T,
	address string,
	prefix string,
	leaseTTL time.Duration,
) *RedisEpisodeStore {
	t.Helper()
	store, err := NewRedisEpisodeStore(RedisEpisodeStoreConfig{
		Address:   address,
		KeyPrefix: prefix,
		LeaseTTL:  leaseTTL,
		IdleTTL:   time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})
	readinessContext, cancel := context.WithTimeout(
		context.Background(),
		time.Second,
	)
	defer cancel()
	if readinessErr := store.Ready(
		readinessContext,
	); readinessErr != nil {
		t.Fatal(readinessErr)
	}
	return store
}
