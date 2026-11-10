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
	"sync/atomic"
	"testing"
	"time"
)

// failingRouteCallerShared stands for a Redis that does not answer.
type failingRouteCallerShared struct{ calls atomic.Int64 }

func (shared *failingRouteCallerShared) acquire(context.Context, routeCallerKeys, int, int, time.Duration,
	time.Time, string, time.Duration,
) (int, error) {
	shared.calls.Add(1)
	return 0, errors.New("redis unavailable")
}

func (shared *failingRouteCallerShared) release(context.Context, routeCallerKeys, string) error {
	return nil
}

func newTestRouteCallerBudget(rate, inflight int) *RouteCallerBudget {
	// No pool: each test gets its own per-process table.
	return NewRouteCallerBudget(RouteCallerBudgetConfig{LookupsPerMinute: rate, MaxInflight: inflight, LookupDeadline: time.Second})
}

func refusalOf(t *testing.T, err error) *RouteCallerRefusal {
	t.Helper()
	var refusal *RouteCallerRefusal
	if !errors.As(err, &refusal) || !errors.Is(err, ErrRouteCallerOverBudget) {
		t.Fatalf("err = %v, want a caller budget refusal", err)
	}
	return refusal
}

// Past lookups_per_minute a caller is refused until its window turns over,
// and told exactly when that is. Other callers are unaffected.
func TestRouteCallerBudgetTripsAtTheRate(t *testing.T) {
	t.Parallel()
	budget := newTestRouteCallerBudget(3, 64)
	// 15 s into a window: 45 s left.
	now := time.UnixMilli(1_000 * 60_000).Add(15 * time.Second)
	budget.now = func() time.Time { return now }
	for range 3 {
		ticket, err := budget.Acquire(context.Background(), "caller-a")
		if err != nil {
			t.Fatalf("lookup within the rate refused: %v", err)
		}
		ticket.Release()
	}
	_, err := budget.Acquire(context.Background(), "caller-a")
	refusal := refusalOf(t, err)
	if refusal.Outcome != RouteCallerRateLimited || refusal.Backend != RouteLookupBackendLocal {
		t.Fatalf("refusal = %+v, want rate_limited on the local backend", refusal)
	}
	if refusal.RetryAfter != 45*time.Second {
		t.Fatalf("RetryAfter = %v, want the 45 s left in the window", refusal.RetryAfter)
	}
	if _, err := budget.Acquire(context.Background(), "caller-b"); err != nil {
		t.Fatalf("another caller was refused: %v", err)
	}
	// The next window admits the caller again.
	now = now.Add(45 * time.Second)
	if _, err := budget.Acquire(context.Background(), "caller-a"); err != nil {
		t.Fatalf("a new window still refused the caller: %v", err)
	}
}

// Past max_inflight a caller is refused until one of its lookups answers.
func TestRouteCallerBudgetCapsInflight(t *testing.T) {
	t.Parallel()
	budget := newTestRouteCallerBudget(100, 2)
	first, err := budget.Acquire(context.Background(), "caller")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := budget.Acquire(context.Background(), "caller"); err != nil {
		t.Fatal(err)
	}
	_, err = budget.Acquire(context.Background(), "caller")
	if refusal := refusalOf(t, err); refusal.Outcome != RouteCallerInflightLimited || refusal.RetryAfter <= 0 {
		t.Fatalf("refusal = %+v, want inflight_limited with a Retry-After", refusal)
	}
	first.Release()
	first.Release()
	if _, err := budget.Acquire(context.Background(), "caller"); err != nil {
		t.Fatalf("a released slot was not reusable: %v", err)
	}
	if _, err := budget.Acquire(context.Background(), "caller"); err == nil {
		t.Fatal("a double release freed two slots")
	}
}

// A lookup that names no caller is not counted against anyone.
func TestRouteCallerBudgetLeavesAnonymousLookupsUncounted(t *testing.T) {
	t.Parallel()
	budget := newTestRouteCallerBudget(1, 1)
	for range 5 {
		ticket, err := budget.Acquire(context.Background(), "")
		if err != nil || ticket.Outcome != RouteCallerAnonymous {
			t.Fatalf("anonymous lookup = %+v, %v; want admitted uncounted", ticket, err)
		}
	}
	if NewRouteCallerBudget(RouteCallerBudgetConfig{}) != nil {
		t.Fatal("a zero config built a budget")
	}
	var disabled *RouteCallerBudget
	if _, err := disabled.Acquire(context.Background(), "caller"); err != nil {
		t.Fatalf("a nil budget refused: %v", err)
	}
}

// A Redis error fails open to the per-process count, trips a backoff so the
// next lookups do not each pay a timeout, and is reported once.
func TestRouteCallerBudgetFailsOpenToTheLocalCount(t *testing.T) {
	t.Parallel()
	budget := newTestRouteCallerBudget(2, 64)
	shared := &failingRouteCallerShared{}
	budget.shared = shared
	var reported atomic.Int64
	budget.SetFailOpenObserver(func(error) { reported.Add(1) })
	for range 2 {
		ticket, err := budget.Acquire(context.Background(), "caller")
		if err != nil {
			t.Fatalf("a Redis failure refused the lookup: %v", err)
		}
		if ticket.Outcome != RouteCallerFailOpen || ticket.Backend != RouteLookupBackendLocal {
			t.Fatalf("ticket = %+v, want fail_open on the local backend", ticket)
		}
	}
	if calls := shared.calls.Load(); calls != 1 {
		t.Fatalf("Redis called %d times, want once before the backoff", calls)
	}
	if reported.Load() != 1 {
		t.Fatalf("fail open reported %d times, want once", reported.Load())
	}
	// Failing open still bounds the caller per process.
	_, err := budget.Acquire(context.Background(), "caller")
	if refusal := refusalOf(t, err); refusal.Backend != RouteLookupBackendLocal {
		t.Fatalf("refusal = %+v, want the local count to hold", refusal)
	}
}

// A Redis that is down (nothing listening) is the same: the lookup is
// admitted on the per-process count, promptly.
func TestRouteCallerBudgetFailsOpenWhenRedisIsDown(t *testing.T) {
	t.Parallel()
	store, err := NewRedisEpisodeStore(RedisEpisodeStoreConfig{
		Address: "127.0.0.1:1", KeyPrefix: "test:", LeaseTTL: time.Second, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	budget := NewRouteCallerBudget(RouteCallerBudgetConfig{
		LookupsPerMinute: 10, MaxInflight: 2, LookupDeadline: time.Second, Store: store,
	})
	t.Cleanup(func() { _ = budget.Close() })
	if !budget.Distributed() {
		t.Fatal("a Redis episode store built a per-process budget")
	}
	started := time.Now()
	ticket, err := budget.Acquire(context.Background(), "caller")
	if err != nil {
		t.Fatalf("Redis down refused the lookup: %v", err)
	}
	if ticket.Outcome != RouteCallerFailOpen {
		t.Fatalf("outcome = %s, want fail_open", ticket.Outcome)
	}
	if waited := time.Since(started); waited > time.Second {
		t.Fatalf("failing open took %v", waited)
	}
	ticket.Release()
}

// Also joins a second hold to the lane ticket: one Release frees both,
// once, and a zero ticket joined to a hold becomes held.
func TestRouteLookupTicketAlsoReleasesBoth(t *testing.T) {
	t.Parallel()
	var released atomic.Int64
	joined := RouteLookupTicket{}.Also(func() { released.Add(1) })
	if !joined.Held() {
		t.Fatal("a joined zero ticket is not held")
	}
	joined.Release()
	joined.Release()
	if released.Load() != 1 {
		t.Fatalf("released %d times, want once", released.Load())
	}
	gate := NewRouteLookupGate(RouteLookupGateConfig{MaxInflight: 1, QueueWait: 10 * time.Millisecond, LookupDeadline: time.Second})
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ticket = ticket.Also(func() { released.Add(1) })
	ticket.Release()
	if released.Load() != 2 {
		t.Fatal("the joined hold was not released")
	}
	next, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("the lane slot was not released: %v", err)
	}
	next.Release()
}

// Against a real Redis: two router instances share one caller's rate and
// in-flight counts, a release frees the shared slot, and a crashed holder's
// slot frees itself at its lease.
func TestRouteCallerBudgetOnRedis(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	pool := RouteLookupPoolID(t.Name(), time.Now().String())
	build := func(prefix string, rate, inflight int) *RouteCallerBudget {
		budget := NewRouteCallerBudget(RouteCallerBudgetConfig{
			LookupsPerMinute: rate, MaxInflight: inflight, LookupDeadline: 100 * time.Millisecond,
			Pool: pool, Store: newTestRedisEpisodeStore(t, address, prefix, time.Second),
		})
		if !budget.Distributed() {
			t.Fatal("a Redis episode store built a per-process budget")
		}
		// Each budget stands for another router instance: its own local
		// table, the shared Redis counts.
		budget.local = routeCallerLocalFor("", rate, inflight)
		t.Cleanup(func() { _ = budget.Close() })
		return budget
	}

	// In flight: two across two instances, then the third is refused.
	instanceA := build("test:caller-a:", 100, 2)
	instanceB := build("test:caller-b:", 100, 2)
	first, err := instanceA.Acquire(context.Background(), "caller")
	if err != nil || first.Backend != RouteLookupBackendRedis {
		t.Fatalf("first = %+v, %v; want a Redis slot", first, err)
	}
	if _, err := instanceB.Acquire(context.Background(), "caller"); err != nil {
		t.Fatal(err)
	}
	_, err = instanceB.Acquire(context.Background(), "caller")
	if refusal := refusalOf(t, err); refusal.Outcome != RouteCallerInflightLimited || refusal.Backend != RouteLookupBackendRedis {
		t.Fatalf("refusal = %+v, want inflight_limited cell-wide", refusal)
	}
	first.Release()
	instanceA.releases.Wait()
	if _, err := instanceB.Acquire(context.Background(), "caller"); err != nil {
		t.Fatalf("a released Redis slot was not reusable: %v", err)
	}

	// Rate: three a minute across two instances.
	rateA := build("test:rate-a:", 3, 64)
	rateB := build("test:rate-b:", 3, 64)
	for _, instance := range []*RouteCallerBudget{rateA, rateB, rateA} {
		ticket, err := instance.Acquire(context.Background(), "rate-caller")
		if err != nil {
			t.Fatalf("lookup within the rate refused: %v", err)
		}
		ticket.Release()
	}
	_, err = rateB.Acquire(context.Background(), "rate-caller")
	if refusal := refusalOf(t, err); refusal.Outcome != RouteCallerRateLimited || refusal.RetryAfter <= 0 ||
		refusal.RetryAfter > RouteCallerWindow {
		t.Fatalf("refusal = %+v, want rate_limited with the window's remainder", refusal)
	}

	// A crashed holder never releases; its slot frees at its lease.
	crashed := build("test:crash-a:", 100, 1)
	crashed.leaseTTL = 150 * time.Millisecond
	if _, err := crashed.Acquire(context.Background(), "crash-caller"); err != nil {
		t.Fatal(err)
	}
	survivor := build("test:crash-b:", 100, 1)
	if _, err := survivor.Acquire(context.Background(), "crash-caller"); err == nil {
		t.Fatal("a held slot admitted a second lookup")
	}
	time.Sleep(200 * time.Millisecond)
	if _, err := survivor.Acquire(context.Background(), "crash-caller"); err != nil {
		t.Fatalf("a crashed holder's slot never freed: %v", err)
	}
}
