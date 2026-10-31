package raylinearc

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"
)

func TestModelFamilyNamesTheVendorFamiliesOnly(t *testing.T) {
	for model, want := range map[string]string{
		"claude-opus-5.5":             "claude",
		"anthropic/claude-sonnet-5.5": "claude",
		"Claude-Haiku-5.5":            "claude",
		"gpt-6-luna":                  "gpt",
		"openai/gpt-6.1-sol":          "gpt",
		"glm-5.3-flash":               "",
		"deepseek-v4.1-flash":         "",
		"claudette":                   "",
		"":                            "",
	} {
		if got := ModelFamily(model); got != want {
			t.Errorf("ModelFamily(%q) = %q, want %q", model, got, want)
		}
	}
	if !ExcludesFamily([]string{"claude"}, "anthropic/claude-opus-5.5") || ExcludesFamily([]string{"claude"}, "gpt-6-luna") ||
		ExcludesFamily([]string{"claude"}, "glm-5.3") || ExcludesFamily(nil, "claude-opus-5.5") {
		t.Fatal("ExcludesFamily matched the wrong models")
	}
}

func exerciseSessionFamilyStore(t *testing.T, store SessionFamilyExclusionStore, advance func(time.Duration)) {
	t.Helper()
	ctx := context.Background()
	if families, err := store.LoadSessionFamilyExclusions(ctx, "user:conv"); err != nil || len(families) != 0 {
		t.Fatalf("absent session: %v %v", families, err)
	}
	if err := store.AddSessionFamilyExclusion(ctx, "user:conv", "claude", time.Hour); err != nil {
		t.Fatal(err)
	}
	if err := store.AddSessionFamilyExclusion(ctx, "user:conv", "claude", time.Hour); err != nil {
		t.Fatal(err)
	}
	if err := store.AddSessionFamilyExclusion(ctx, "user:conv", "gpt", 2*time.Hour); err != nil {
		t.Fatal(err)
	}
	families, err := store.LoadSessionFamilyExclusions(ctx, "user:conv")
	slices.Sort(families)
	if err != nil || !slices.Equal(families, []string{"claude", "gpt"}) {
		t.Fatalf("session families = %v %v", families, err)
	}
	if other, _ := store.LoadSessionFamilyExclusions(ctx, "user:other"); len(other) != 0 {
		t.Fatalf("another session reads %v", other)
	}
	for _, bad := range []struct{ key, family string }{{"", "claude"}, {"user:conv", ""}, {"user:conv", "two words"}} {
		if store.AddSessionFamilyExclusion(ctx, bad.key, bad.family, time.Hour) == nil {
			t.Fatalf("accepted %+v", bad)
		}
	}
	if advance != nil {
		advance(3 * time.Hour)
		if expired, _ := store.LoadSessionFamilyExclusions(ctx, "user:conv"); len(expired) != 0 {
			t.Fatalf("expired record still reads %v", expired)
		}
	}
}

func TestMemorySessionFamilyExclusions(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	store, err := NewMemoryEpisodeStore(MemoryEpisodeStoreConfig{
		MaxEpisodes: 1, IdleTTL: time.Minute, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	exerciseSessionFamilyStore(t, store, func(step time.Duration) { now = now.Add(step) })
	// The bound: a ninth family is dropped, never an error.
	for index := 0; index < MaxSessionFamilyExclusions+1; index++ {
		if err := store.AddSessionFamilyExclusion(context.Background(), "user:many", string(rune('a'+index)), time.Hour); err != nil {
			t.Fatal(err)
		}
	}
	if families, _ := store.LoadSessionFamilyExclusions(context.Background(), "user:many"); len(families) != MaxSessionFamilyExclusions {
		t.Fatalf("bounded record holds %d", len(families))
	}
}

func TestRedisSessionFamilyExclusions(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	prefix := "test:rayline-arc:" + HashEpisodeID(t.Name()+time.Now().String()) + ":"
	store := newTestRedisEpisodeStore(t, address, prefix, time.Second)
	exerciseSessionFamilyStore(t, store, nil)
	if ttl := store.client.PTTL(context.Background(), store.sessionFamilyKey("user:conv")).Val(); ttl <= time.Hour || ttl > 2*time.Hour {
		t.Fatalf("record TTL %v, want the latest exclusion's 2h", ttl)
	}
}
