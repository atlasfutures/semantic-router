package raylinearc

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"
)

// relaxedStores runs a check against every store that serves relaxed
// episodes: memory always, Redis when RAYLINE_ARC_TEST_REDIS_ADDR is set.
func relaxedStores(t *testing.T, check func(t *testing.T, store interface {
	EpisodeStore
	EpisodeSnapshotStore
},
),
) {
	t.Run("memory", func(t *testing.T) {
		check(t, newTestMemoryEpisodeStore(t, 4, time.Now))
	})
	t.Run("redis", func(t *testing.T) {
		address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
		if address == "" {
			t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
		}
		prefix := "test:rayline-arc-relaxed:" + HashEpisodeID(t.Name()+time.Now().String()) + ":"
		check(t, newTestRedisEpisodeStore(t, address, prefix, time.Minute))
	})
}

func committedTurn(t *testing.T, state *EpisodeState, arm int) *EpisodeState {
	t.Helper()
	next := cloneEpisodeState(state)
	if err := next.Commit(arm, 10, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	return next
}

// A relaxed read takes no lease, and a relaxed commit lands only if the
// episode is still at the version read: of two turns that read the same
// version, the second commit loses with ErrEpisodeConflict.
func TestRelaxedCommitLandsOnlyOnTheVersionRead(t *testing.T) {
	relaxedStores(t, func(t *testing.T, store interface {
		EpisodeStore
		EpisodeSnapshotStore
	},
	) {
		ctx := context.Background()
		episode := HashEpisodeID("relaxed")
		first, read, err := store.Snapshot(ctx, episode, 2)
		if err != nil || read.Version() != 0 || first.TurnIndex != 0 {
			t.Fatalf("fresh snapshot = turn %v, version %d, err %v", first, read.Version(), err)
		}
		second, _, err := store.Snapshot(ctx, episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		if commitErr := store.CommitIfUnchanged(ctx, episode, read, committedTurn(t, first, 0)); commitErr != nil {
			t.Fatalf("first relaxed commit = %v", commitErr)
		}
		if commitErr := store.CommitIfUnchanged(ctx, episode, read, committedTurn(t, second, 1)); !errors.Is(commitErr, ErrEpisodeConflict) {
			t.Fatalf("second relaxed commit on the same read = %v, want ErrEpisodeConflict", commitErr)
		}
		after, afterRead, err := store.Snapshot(ctx, episode, 2)
		if err != nil || afterRead.Version() != read.Version()+1 || after.TurnIndex != 1 || after.PreviousArm == nil || *after.PreviousArm != 0 {
			t.Fatalf("after = %+v version %d err %v, want the first turn at version %d", after, afterRead.Version(), err, read.Version()+1)
		}
	})
}

// A strict lease and a relaxed commit never interleave: a relaxed commit
// refuses while a lease holds the episode, and a strict prepare after a
// relaxed commit reads that commit and advances past its version.
func TestRelaxedAndStrictRespectEachOther(t *testing.T) {
	relaxedStores(t, func(t *testing.T, store interface {
		EpisodeStore
		EpisodeSnapshotStore
	},
	) {
		ctx := context.Background()
		episode := HashEpisodeID("mixed")
		state, read, err := store.Snapshot(ctx, episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		if commitErr := store.CommitIfUnchanged(ctx, episode, read, committedTurn(t, state, 1)); commitErr != nil {
			t.Fatalf("relaxed commit = %v", commitErr)
		}
		lease, prepared, err := store.Prepare(ctx, episode, 2)
		if err != nil {
			t.Fatalf("strict prepare after a relaxed commit = %v", err)
		}
		if prepared.TurnIndex != 1 {
			t.Fatalf("strict prepare read turn %d, want the relaxed commit's 1", prepared.TurnIndex)
		}
		snapshot, snapshotRead, err := store.Snapshot(ctx, episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		if err := store.CommitIfUnchanged(ctx, episode, snapshotRead, committedTurn(t, snapshot, 0)); !errors.Is(err, ErrEpisodeConflict) {
			t.Fatalf("relaxed commit under a strict lease = %v, want ErrEpisodeConflict", err)
		}
		if err := store.Abort(ctx, lease); err != nil {
			t.Fatal(err)
		}
	})
}

// advanceRelaxed commits n relaxed turns to episode.
func advanceRelaxed(t *testing.T, store EpisodeSnapshotStore, episode string, n int) {
	t.Helper()
	for i := range n {
		state, read, err := store.Snapshot(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		if err := store.CommitIfUnchanged(context.Background(), episode, read, committedTurn(t, state, i%2)); err != nil {
			t.Fatalf("advance %d = %v", i, err)
		}
	}
}

// A memory entry evicted and recreated back to the version a stale relaxed
// turn read must not accept that turn's commit.
func TestRelaxedCommitLosesToARecreatedMemoryEntry(t *testing.T) {
	store := newTestMemoryEpisodeStore(t, 1, time.Now)
	episode := HashEpisodeID("recreated")
	advanceRelaxed(t, store, episode, 1)
	stale, staleRead, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Capacity one: another episode evicts this one.
	advanceRelaxed(t, store, HashEpisodeID("evictor"), 1)
	advanceRelaxed(t, store, episode, 1)
	_, recreated, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil || recreated.Version() != staleRead.Version() {
		t.Fatalf("recreated at version %d, want the stale read's %d to exercise ABA", recreated.Version(), staleRead.Version())
	}
	if err := store.CommitIfUnchanged(context.Background(), episode, staleRead, committedTurn(t, stale, 1)); !errors.Is(err, ErrEpisodeConflict) {
		t.Fatalf("stale commit to a recreated entry = %v, want ErrEpisodeConflict", err)
	}
}

// A Redis episode that expired and was recreated back to the version a stale
// relaxed turn read must not accept that turn's commit.
func TestRelaxedCommitLosesToARecreatedRedisEpisode(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	prefix := "test:rayline-arc-aba:" + HashEpisodeID(t.Name()+time.Now().String()) + ":"
	store := newTestRedisEpisodeStore(t, address, prefix, time.Minute)
	episode := HashEpisodeID("recreated")
	advanceRelaxed(t, store, episode, 1)
	stale, staleRead, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Expiry: the fence and the state go away together.
	keys := store.keys(episode)
	if commitErr := store.client.Del(context.Background(), keys[1], keys[2]).Err(); commitErr != nil {
		t.Fatal(commitErr)
	}
	time.Sleep(5 * time.Millisecond)
	advanceRelaxed(t, store, episode, 1)
	_, recreated, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil || recreated.Version() != staleRead.Version() {
		t.Fatalf("recreated at version %d, want the stale read's %d to exercise ABA", recreated.Version(), staleRead.Version())
	}
	if err := store.CommitIfUnchanged(context.Background(), episode, staleRead, committedTurn(t, stale, 1)); !errors.Is(err, ErrEpisodeConflict) {
		t.Fatalf("stale commit to a recreated episode = %v, want ErrEpisodeConflict", err)
	}
}

// A read that found the episode absent cannot tell an episode that was never
// created from one created and expired since; both look absent. Only a read
// older than the idle TTL could have seen a newer incarnation come and go, so
// such a read is refused, and a fresh one still commits.
func TestRelaxedAbsentReadOlderThanIdleTTLLoses(t *testing.T) {
	clock := time.Now()
	now := func() time.Time { return clock }
	store, err := NewMemoryEpisodeStore(MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute, Now: now})
	if err != nil {
		t.Fatal(err)
	}
	episode := HashEpisodeID("absent")
	stale, staleRead, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	// A newer incarnation is created, then idles out.
	advanceRelaxed(t, store, episode, 1)
	clock = clock.Add(time.Minute + time.Second)
	if _, read, _ := store.Snapshot(context.Background(), episode, 2); read.Version() != 0 {
		t.Fatalf("the newer incarnation did not expire: version %d", read.Version())
	}
	if commitErr := store.CommitIfUnchanged(context.Background(), episode, staleRead, committedTurn(t, stale, 1)); !errors.Is(commitErr, ErrEpisodeConflict) {
		t.Fatalf("stale absent-read commit = %v, want ErrEpisodeConflict", commitErr)
	}
	fresh, freshRead, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.CommitIfUnchanged(context.Background(), episode, freshRead, committedTurn(t, fresh, 0)); err != nil {
		t.Fatalf("fresh absent-read commit = %v", err)
	}
}

// The same bound on Redis, whose expiry runs on its own TTL.
func TestRelaxedRedisReadOlderThanIdleTTLLoses(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	clock := time.Now()
	store, err := NewRedisEpisodeStore(RedisEpisodeStoreConfig{
		Address:   address,
		KeyPrefix: "test:rayline-arc-stale:" + HashEpisodeID(t.Name()+clock.String()) + ":",
		LeaseTTL:  time.Second,
		IdleTTL:   time.Minute,
		Now:       func() time.Time { return clock },
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	episode := HashEpisodeID("absent")
	stale, staleRead, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	clock = clock.Add(time.Minute)
	if err := store.CommitIfUnchanged(context.Background(), episode, staleRead, committedTurn(t, stale, 1)); !errors.Is(err, ErrEpisodeConflict) {
		t.Fatalf("commit of a read one idle TTL old = %v, want ErrEpisodeConflict", err)
	}
}

// Capacity eviction is immediate, not TTL-bounded: an episode created after
// an absent read and evicted right away must not let the stale read commit.
func TestRelaxedAbsentReadLosesToACreatedAndEvictedEpisode(t *testing.T) {
	store := newTestMemoryEpisodeStore(t, 1, time.Now)
	episode := HashEpisodeID("absent-evicted")
	stale, staleRead, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	advanceRelaxed(t, store, episode, 1)
	advanceRelaxed(t, store, HashEpisodeID("evictor"), 1)
	if _, read, _ := store.Snapshot(context.Background(), episode, 2); read.Version() != 0 {
		t.Fatalf("the episode was not evicted: version %d", read.Version())
	}
	if err := store.CommitIfUnchanged(context.Background(), episode, staleRead, committedTurn(t, stale, 1)); !errors.Is(err, ErrEpisodeConflict) {
		t.Fatalf("stale absent read after create-and-evict = %v, want ErrEpisodeConflict", err)
	}
}

// A relaxed read taken while a strict lease holds a memory entry must not
// commit over the strict turn's commit.
func TestRelaxedReadDuringAStrictLeaseCannotOverwriteItsCommit(t *testing.T) {
	store := newTestMemoryEpisodeStore(t, 4, time.Now)
	episode := HashEpisodeID("leased")
	lease, prepared, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	relaxed, relaxedRead, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	if commitErr := store.Commit(context.Background(), lease, lease.Version(), committedTurn(t, prepared, 0)); commitErr != nil {
		t.Fatalf("strict commit = %v", commitErr)
	}
	if commitErr := store.CommitIfUnchanged(context.Background(), episode, relaxedRead, committedTurn(t, relaxed, 1)); !errors.Is(commitErr, ErrEpisodeConflict) {
		t.Fatalf("relaxed commit after the strict commit = %v, want ErrEpisodeConflict", commitErr)
	}
	after, _, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil || after.PreviousArm == nil || *after.PreviousArm != 0 {
		t.Fatalf("the strict commit was overwritten: %+v err %v", after, err)
	}
}

// A relaxed read is episode activity: an episode read shortly before its idle
// deadline must still accept that read's commit afterwards.
func TestRelaxedReadRestartsTheIdleWindow(t *testing.T) {
	clock := time.Now()
	store, err := NewMemoryEpisodeStore(MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute, Now: func() time.Time { return clock }})
	if err != nil {
		t.Fatal(err)
	}
	episode := HashEpisodeID("idle")
	advanceRelaxed(t, store, episode, 1)
	clock = clock.Add(50 * time.Second)
	state, read, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil || read.Version() != 1 {
		t.Fatalf("snapshot = version %d, err %v", read.Version(), err)
	}
	clock = clock.Add(20 * time.Second)
	// Activity elsewhere reaps every entry past its idle deadline.
	if _, _, err := store.Snapshot(context.Background(), HashEpisodeID("elsewhere"), 2); err != nil {
		t.Fatal(err)
	}
	if err := store.CommitIfUnchanged(context.Background(), episode, read, committedTurn(t, state, 0)); err != nil {
		t.Fatalf("commit 20 s after a read near the idle deadline = %v", err)
	}
}

// On Redis a relaxed read restarts the idle TTL of both episode keys.
func TestRelaxedRedisReadRestartsTheIdleTTL(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	prefix := "test:rayline-arc-idle:" + HashEpisodeID(t.Name()+time.Now().String()) + ":"
	store := newTestRedisEpisodeStore(t, address, prefix, time.Second)
	episode := HashEpisodeID("idle")
	advanceRelaxed(t, store, episode, 1)
	keys := store.keys(episode)
	for _, key := range keys[1:] {
		if err := store.client.PExpire(context.Background(), key, time.Second).Err(); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := store.Snapshot(context.Background(), episode, 2); err != nil {
		t.Fatal(err)
	}
	for _, key := range keys[1:] {
		ttl, err := store.client.PTTL(context.Background(), key).Result()
		if err != nil || ttl < 30*time.Second {
			t.Fatalf("%s TTL after a relaxed read = %v (err %v), want the idle TTL restarted", key, ttl, err)
		}
	}
}
