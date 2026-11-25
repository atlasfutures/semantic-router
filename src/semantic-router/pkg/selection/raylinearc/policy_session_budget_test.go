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
	"os"
	"strings"
	"testing"
	"time"
)

// The caller id is opaque: it is counted under a hash, never as itself, and an
// absent or unprintable one has no budget.
func TestPolicySessionCallerKeyHashesTheCaller(t *testing.T) {
	key := PolicySessionCallerKey(" caller-a ")
	if !validEpisodeIDHash(key) || strings.Contains(key, "caller-a") || key != PolicySessionCallerKey("caller-a") {
		t.Fatalf("key %q", key)
	}
	if key == PolicySessionCallerKey("caller-b") {
		t.Fatal("two callers share a key")
	}
	for _, caller := range []string{"", "  ", "a b", strings.Repeat("x", maxPolicySessionCallerBytes+1)} {
		if got := PolicySessionCallerKey(caller); got != "" {
			t.Fatalf("caller %q keyed %q, want none", caller, got)
		}
	}
}

// checkBudgetWindow runs the budget's contract on a fake clock: the limit
// admits, the next is past it and is not counted, other callers are separate,
// and starts that leave the window free the budget again.
func checkBudgetWindow(t *testing.T, budget PolicySessionBudget, clock *time.Time) {
	t.Helper()
	ctx := context.Background()
	caller, other := PolicySessionCallerKey("caller-a"), PolicySessionCallerKey("caller-b")
	admit := func(key string) bool {
		t.Helper()
		admitted, err := budget.AdmitPolicySessionStart(ctx, key, 3, 10*time.Minute)
		if err != nil {
			t.Fatal(err)
		}
		return admitted
	}
	for start := 0; start < 3; start++ {
		if !admit(caller) {
			t.Fatalf("start %d refused under the budget", start)
		}
		*clock = clock.Add(time.Minute)
	}
	for attempt := 0; attempt < 3; attempt++ {
		if admit(caller) {
			t.Fatalf("attempt %d admitted past the budget", attempt)
		}
	}
	if !admit(other) {
		t.Fatal("another caller was charged for this one's starts")
	}
	// The first start (t=0) leaves the window at t=10m; the refused attempts
	// were never counted, so exactly one start frees up.
	*clock = clock.Add(7*time.Minute + time.Second)
	if !admit(caller) {
		t.Fatal("the window did not roll over")
	}
	if admit(caller) {
		t.Fatal("more than the expired start was freed")
	}
	*clock = clock.Add(10 * time.Minute)
	for start := 0; start < 3; start++ {
		if !admit(caller) {
			t.Fatalf("start %d refused after the whole window passed", start)
		}
	}
	if _, err := budget.AdmitPolicySessionStart(ctx, "not-a-key", 3, time.Minute); err == nil {
		t.Fatal("an invalid caller key was counted")
	}
}

func TestMemoryPolicySessionBudgetWindow(t *testing.T) {
	clock := time.Unix(1_800_000_000, 0)
	budget := NewMemoryPolicySessionBudget()
	budget.now = func() time.Time { return clock }
	checkBudgetWindow(t, budget, &clock)
	// Callers whose starts all left the window are dropped.
	clock = clock.Add(time.Hour)
	if _, err := budget.AdmitPolicySessionStart(context.Background(), PolicySessionCallerKey("caller-c"), 3, 10*time.Minute); err != nil {
		t.Fatal(err)
	}
	if len(budget.starts) != 1 {
		t.Fatalf("budget keeps %d callers, want 1", len(budget.starts))
	}
}

// The Redis budget is shared by every router instance, beside the episodes,
// and expires with its window.
func TestRedisEpisodeStorePolicySessionBudget(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	prefix := "test:rayline-arc:" + HashEpisodeID(t.Name()+time.Now().String()) + ":"
	store := newTestRedisEpisodeStore(t, address, prefix, time.Second)
	clock := time.Now()
	store.now = func() time.Time { return clock }
	checkBudgetWindow(t, store, &clock)
	key := store.sessionStartsKey(PolicySessionCallerKey("caller-a"))
	ttl, err := store.client.PTTL(context.Background(), key).Result()
	requireARCNoError(t, err)
	if ttl <= 0 || ttl > 10*time.Minute {
		t.Fatalf("budget TTL = %s, want within the window", ttl)
	}
	if keys, _ := store.client.Keys(context.Background(), prefix+"*caller*").Result(); len(keys) != 0 {
		t.Fatalf("the raw caller id reached a key: %v", keys)
	}
}

// A store that cannot be reached answers an error, which the cell reads as
// fail open.
func TestRedisEpisodeStorePolicySessionBudgetUnreachable(t *testing.T) {
	store, err := NewRedisEpisodeStore(RedisEpisodeStoreConfig{
		Address: "127.0.0.1:1", KeyPrefix: "test:", LeaseTTL: time.Second, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := store.AdmitPolicySessionStart(ctx, PolicySessionCallerKey("caller-a"), 3, time.Minute); err == nil {
		t.Fatal("an unreachable store answered a count")
	}
}
