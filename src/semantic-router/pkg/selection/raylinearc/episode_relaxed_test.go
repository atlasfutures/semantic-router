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
)) {
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
		first, version, err := store.Snapshot(ctx, episode, 2)
		if err != nil || version != 0 || first.TurnIndex != 0 {
			t.Fatalf("fresh snapshot = turn %v, version %d, err %v", first, version, err)
		}
		second, _, err := store.Snapshot(ctx, episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		if err := store.CommitIfUnchanged(ctx, episode, version, committedTurn(t, first, 0)); err != nil {
			t.Fatalf("first relaxed commit = %v", err)
		}
		if err := store.CommitIfUnchanged(ctx, episode, version, committedTurn(t, second, 1)); !errors.Is(err, ErrEpisodeConflict) {
			t.Fatalf("second relaxed commit on the same version = %v, want ErrEpisodeConflict", err)
		}
		after, afterVersion, err := store.Snapshot(ctx, episode, 2)
		if err != nil || afterVersion != version+1 || after.TurnIndex != 1 || after.PreviousArm == nil || *after.PreviousArm != 0 {
			t.Fatalf("after = %+v version %d err %v, want the first turn at version %d", after, afterVersion, err, version+1)
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
		state, version, err := store.Snapshot(ctx, episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		if err := store.CommitIfUnchanged(ctx, episode, version, committedTurn(t, state, 1)); err != nil {
			t.Fatalf("relaxed commit = %v", err)
		}
		lease, prepared, err := store.Prepare(ctx, episode, 2)
		if err != nil {
			t.Fatalf("strict prepare after a relaxed commit = %v", err)
		}
		if prepared.TurnIndex != 1 {
			t.Fatalf("strict prepare read turn %d, want the relaxed commit's 1", prepared.TurnIndex)
		}
		snapshot, snapshotVersion, err := store.Snapshot(ctx, episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		if err := store.CommitIfUnchanged(ctx, episode, snapshotVersion, committedTurn(t, snapshot, 0)); !errors.Is(err, ErrEpisodeConflict) {
			t.Fatalf("relaxed commit under a strict lease = %v, want ErrEpisodeConflict", err)
		}
		if err := store.Abort(ctx, lease); err != nil {
			t.Fatal(err)
		}
	})
}
