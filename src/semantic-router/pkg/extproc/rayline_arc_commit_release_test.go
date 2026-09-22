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
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// testEpisodeHash is a real sha256 hex digest: the store validates the shape
// of an episode identity before it will prepare one.
const testEpisodeHash = "cf452e8b6e74a25ef0d03b6632555f725c5c8083ffc45460fc27674600bbcf37"

// cancelHonouringStore behaves like the Redis store rather than the in-memory
// one: it refuses work on a cancelled context. That is the case the fake
// transactions elsewhere in these tests cannot reach, because they stand in
// for the transaction itself and so never exercise finalizeOnce.
type cancelHonouringStore struct {
	inner raylinearc.EpisodeStore
	// abortedLive records whether the store's Abort ran on a context that was
	// still alive. A cancelled one cannot release the lease.
	abortedLive bool
	aborts      int
}

func (s *cancelHonouringStore) Prepare(
	ctx context.Context,
	episodeIDHash string,
	arms int,
) (raylinearc.Lease, *raylinearc.EpisodeState, error) {
	return s.inner.Prepare(ctx, episodeIDHash, arms)
}

func (s *cancelHonouringStore) Commit(
	ctx context.Context,
	lease raylinearc.Lease,
	version uint64,
	state *raylinearc.EpisodeState,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return s.inner.Commit(ctx, lease, version, state)
}

func (s *cancelHonouringStore) Abort(ctx context.Context, lease raylinearc.Lease) error {
	s.aborts++
	if ctx.Err() == nil {
		s.abortedLive = true
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return s.inner.Abort(ctx, lease)
}

// A commit cancelled by the lookup's deadline must still release the lease.
//
// finalizeOnce has already fired by the time the commit fails, so a later
// abort() is a no-op that returns the stored commit error -- the store is
// never asked again. If the commit's own cleanup also runs on the cancelled
// context, nothing releases the lease at all: it sits until its TTL while
// every later turn on that session is refused as contended.
func TestCancelledCommitStillReleasesTheLease(t *testing.T) {
	t.Parallel()
	memory, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{
		MaxEpisodes: 4,
		IdleTTL:     time.Minute,
	})
	if err != nil {
		t.Fatalf("NewMemoryEpisodeStore() error = %v", err)
	}
	store := &cancelHonouringStore{inner: memory}
	lease, state, err := store.Prepare(context.Background(), testEpisodeHash, 2)
	if err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	transaction := newRaylineARCEpisodeTransaction(
		store, lease, state, testEpisodeHash, time.Minute, nil,
	)
	transaction.selectionReady = true

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	if err := transaction.commit(cancelled, &RequestContext{}); err == nil {
		t.Fatal("a commit on a cancelled context reported success")
	}
	if store.aborts == 0 {
		t.Fatal("the failed commit never asked the store to release the lease")
	}
	if !store.abortedLive {
		t.Fatal("the lease was released on the cancelled context, so it was not released at all")
	}

	// And the lease really is gone: a fresh prepare on the same episode
	// succeeds rather than contending with a lease nobody holds.
	done := make(chan error, 1)
	go func() {
		_, _, prepareErr := store.Prepare(context.Background(), testEpisodeHash, 2)
		done <- prepareErr
	}()
	select {
	case prepareErr := <-done:
		if prepareErr != nil {
			t.Fatalf("the next turn on this session was refused: %v", prepareErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the next turn on this session blocked on a lease that was never released")
	}
}
