package raylinearc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// servedCommitStore is an episode store that records served workers in its
// commits.
type servedCommitStore interface {
	EpisodeStore
	EpisodeSnapshotStore
	ServedModelStore
}

// served is the prepared state, set to record worker.
func served(state *EpisodeState, worker string) *EpisodeState {
	state.ServedModel = worker
	return state
}

// The record is written by the episode commit and nothing else: a strict
// commit records its turn's worker, a later one replaces it, and a commit the
// store refuses -- a lost lease, a relaxed read the episode moved past --
// records nothing. Out-of-order arrival is decided by the episode's revision,
// never by a clock.
func assertCommitRecordsServedModel(t *testing.T, store servedCommitStore, episode string) {
	t.Helper()
	ctx := context.Background()
	if worker, err := store.LastServedModel(ctx, episode); err != nil || worker != "" {
		t.Fatalf("empty store read %q, %v", worker, err)
	}
	commit := func(worker string) (Lease, *EpisodeState) {
		lease, state, err := store.Prepare(ctx, episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		if err := store.Commit(ctx, lease, lease.Version(), served(state, worker)); err != nil {
			t.Fatal(err)
		}
		return lease, state
	}
	first, firstState := commit("glm")
	if worker, _ := store.LastServedModel(ctx, episode); worker != "glm" {
		t.Fatalf("record %q after the first commit", worker)
	}
	commit("opus")
	// The first turn's commit replayed after the second: its revision is
	// stale, the store refuses it, and the record stays the second turn's.
	if err := store.Commit(ctx, first, first.Version(), served(firstState, "glm")); err == nil {
		t.Fatal("a stale commit landed")
	}
	if worker, _ := store.LastServedModel(ctx, episode); worker != "opus" {
		t.Fatalf("record %q after a stale replay, want opus", worker)
	}
	// Relaxed: two turns read the same revision; the one that commits
	// second conflicts and records nothing.
	stateA, readA, err := store.Snapshot(ctx, episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	stateB, readB, err := store.Snapshot(ctx, episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err = store.CommitIfUnchanged(ctx, episode, readB, served(stateB, "kimi")); err != nil {
		t.Fatal(err)
	}
	if err = store.CommitIfUnchanged(ctx, episode, readA, served(stateA, "glm")); !errors.Is(err, ErrEpisodeConflict) {
		t.Fatalf("the earlier read committed: %v", err)
	}
	if worker, _ := store.LastServedModel(ctx, episode); worker != "kimi" {
		t.Fatalf("record %q after a conflicting relaxed commit, want kimi", worker)
	}
	// A commit without a served worker (a side call, artifact mode) leaves
	// the record as it is.
	lease, state, err := store.Prepare(ctx, episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Commit(ctx, lease, lease.Version(), served(state, "")); err != nil {
		t.Fatal(err)
	}
	if worker, _ := store.LastServedModel(ctx, episode); worker != "kimi" {
		t.Fatalf("record %q after a commit naming no worker", worker)
	}
}

func TestMemoryCommitRecordsServedModel(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	store, err := NewMemoryEpisodeStore(MemoryEpisodeStoreConfig{
		MaxEpisodes: 2, IdleTTL: time.Minute, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	episode := HashEpisodeID("served-episode")
	assertCommitRecordsServedModel(t, store, episode)
	// The record outlives the episode, for ServedModelTTL.
	now = now.Add(time.Hour)
	if worker, _ := store.LastServedModel(context.Background(), episode); worker != "kimi" {
		t.Fatalf("record %q after the episode's idle TTL", worker)
	}
	now = now.Add(ServedModelTTL)
	if worker, _ := store.LastServedModel(context.Background(), episode); worker != "" {
		t.Fatalf("an expired record read %q", worker)
	}
	// At most the episode capacity of records, dropping the oldest.
	for _, name := range []string{"a", "b", "c"} {
		if err := store.SeedServedModel(HashEpisodeID(name), "glm"); err != nil {
			t.Fatal(err)
		}
		now = now.Add(time.Second)
	}
	if len(store.served) != 2 {
		t.Fatalf("kept %d records, want the capacity of 2", len(store.served))
	}
	if worker, _ := store.LastServedModel(context.Background(), HashEpisodeID("a")); worker != "" {
		t.Fatal("the oldest record was not the one dropped")
	}
}

func TestRedisCommitRecordsServedModel(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	store := newTestRedisEpisodeStore(t, address, "vsr:served-test:"+strings.ReplaceAll(t.Name(), "/", "_")+":"+
		time.Now().Format("150405.000000")+":", time.Second)
	episode := HashEpisodeID("served-episode")
	assertCommitRecordsServedModel(t, store, episode)
	ttl, err := store.client.TTL(context.Background(), store.servedKey(episode)).Result()
	if err != nil || ttl <= time.Hour {
		t.Fatalf("record TTL %v, %v; want ServedModelTTL", ttl, err)
	}
}

// A two-stage package's cold refusal gets its own class; any other
// selection_refused keeps the contract's code.
func TestPolicyServiceClassifiesStageOneHeldUnknown(t *testing.T) {
	for _, tc := range []struct {
		detail map[string]any
		want   string
	}{
		{map[string]any{"reason": "stage_one_held_unknown"}, PolicyStageOneHeldUnknownClass},
		{map[string]any{"reason": "no_supported_selection"}, "selection_refused"},
		{map[string]any{}, "selection_refused"},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(http.StatusUnprocessableEntity)
			_ = json.NewEncoder(writer).Encode(map[string]any{"error": "selection_refused", "detail": tc.detail})
		}))
		client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: time.Second})
		_, err := client.Decide(context.Background(), PolicyDecisionRequest{})
		server.Close()
		var failure *PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != tc.want {
			t.Fatalf("detail %v: err = %v, want class %s", tc.detail, err, tc.want)
		}
	}
}
