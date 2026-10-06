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

func assertServedWorkerStore(t *testing.T, store ServedWorkerStore, advance func(time.Duration)) {
	t.Helper()
	ctx := context.Background()
	episode := HashEpisodeID("served-episode")
	if worker, err := store.LastServedWorker(ctx, episode); err != nil || worker != "" {
		t.Fatalf("empty store read %q, %v", worker, err)
	}
	earlier, later := time.Unix(1_700_000_000, 0), time.Unix(1_700_000_000, 1000)
	if err := store.RecordServedWorker(ctx, episode, "glm", earlier); err != nil {
		t.Fatal(err)
	}
	if err := store.RecordServedWorker(ctx, episode, "opus", later); err != nil {
		t.Fatal(err)
	}
	// The earlier turn's write lands last; the later turn's record stands.
	if err := store.RecordServedWorker(ctx, episode, "glm", earlier); err != nil {
		t.Fatal(err)
	}
	if worker, err := store.LastServedWorker(ctx, episode); err != nil || worker != "opus" {
		t.Fatalf("read %q, %v; want the later turn's worker", worker, err)
	}
	if err := store.RecordServedWorker(ctx, "not-a-hash", "opus", later); err == nil {
		t.Fatal("recorded under an invalid episode hash")
	}
	if err := store.RecordServedWorker(ctx, episode, "opus", time.Time{}); err == nil {
		t.Fatal("recorded without a stamp")
	}
	if advance != nil {
		advance(ServedWorkerTTL + time.Second)
		if worker, _ := store.LastServedWorker(ctx, episode); worker != "" {
			t.Fatalf("an expired record read %q", worker)
		}
	}
}

// The memory store keeps the record past the episode's own idle TTL, for
// ServedWorkerTTL, and at most its episode capacity of records.
func TestMemoryServedWorkerStore(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	store, err := NewMemoryEpisodeStore(MemoryEpisodeStoreConfig{
		MaxEpisodes: 2, IdleTTL: time.Minute, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	assertServedWorkerStore(t, store, func(by time.Duration) { now = now.Add(by) })
	ctx := context.Background()
	for _, episode := range []string{"a", "b", "c"} {
		if err := store.RecordServedWorker(ctx, HashEpisodeID(episode), "glm", now); err != nil {
			t.Fatal(err)
		}
		now = now.Add(time.Second)
	}
	if len(store.served) != 2 {
		t.Fatalf("kept %d records, want the capacity of 2", len(store.served))
	}
	if worker, _ := store.LastServedWorker(ctx, HashEpisodeID("a")); worker != "" {
		t.Fatal("the oldest record was not the one dropped")
	}
}

func TestRedisServedWorkerStore(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	store := newTestRedisEpisodeStore(t, address, "vsr:served-test:"+strings.ReplaceAll(t.Name(), "/", "_")+":", time.Second)
	assertServedWorkerStore(t, store, nil)
	ttl, err := store.client.TTL(context.Background(), store.servedKey(HashEpisodeID("served-episode"))).Result()
	if err != nil || ttl <= time.Hour {
		t.Fatalf("record TTL %v, %v; want ServedWorkerTTL", ttl, err)
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
