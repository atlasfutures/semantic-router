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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
)

// fakeRouteLookupLeases is the shared store two gates (two router instances)
// contend on. It honours tokens the way the Redis scripts do.
type fakeRouteLookupLeases struct {
	mu      sync.Mutex
	slots   []string
	failing atomic.Bool
	// writeThenFail applies the acquire and then reports an error, as a
	// reply lost after the write would.
	writeThenFail atomic.Bool
	calls         atomic.Int64
}

func newFakeRouteLookupLeases(slots int) *fakeRouteLookupLeases {
	return &fakeRouteLookupLeases{slots: make([]string, slots)}
}

func (fake *fakeRouteLookupLeases) acquire(_ context.Context, token string, _ time.Duration) (bool, error) {
	fake.calls.Add(1)
	if fake.failing.Load() {
		return false, errors.New("redis unavailable")
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	for index, holder := range fake.slots {
		if holder == "" {
			fake.slots[index] = token
			if fake.writeThenFail.Load() {
				return false, errors.New("reply lost")
			}
			return true, nil
		}
	}
	return false, nil
}

func (fake *fakeRouteLookupLeases) release(_ context.Context, token string) error {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	for index, holder := range fake.slots {
		if holder == token {
			fake.slots[index] = ""
		}
	}
	return nil
}

func (fake *fakeRouteLookupLeases) held() int {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	count := 0
	for _, holder := range fake.slots {
		if holder != "" {
			count++
		}
	}
	return count
}

func testRouteLookupGate(limit int, wait time.Duration, leases routeLookupLeases) *RouteLookupGate {
	gate := NewRouteLookupGate(RouteLookupGateConfig{
		MaxInflight:    limit,
		QueueWait:      wait,
		LookupDeadline: time.Second,
	})
	gate.leases = leases
	gate.pollMin = time.Millisecond
	gate.pollMax = 5 * time.Millisecond
	return gate
}

func TestRouteLookupGateDisabledAdmitsEverything(t *testing.T) {
	t.Parallel()
	gate := NewRouteLookupGate(RouteLookupGateConfig{MaxInflight: 0})
	if gate != nil {
		t.Fatal("max_inflight 0 built a gate, want admission disabled")
	}
	for range 3 {
		ticket, err := gate.Acquire(context.Background())
		if err != nil {
			t.Fatalf("disabled gate refused: %v", err)
		}
		ticket.Release()
	}
}

// The per-process stage serializes lookups in one instance: the second waits
// for the first and is admitted the moment it answers.
func TestRouteLookupGateQueuesBehindTheHolder(t *testing.T) {
	t.Parallel()
	gate := testRouteLookupGate(1, time.Second, nil)
	first, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if first.Outcome != RouteLookupAdmitted || first.Backend != RouteLookupBackendLocal {
		t.Fatalf("first = %+v, want admitted on the local gate", first)
	}
	go func() {
		time.Sleep(50 * time.Millisecond)
		first.Release()
	}()
	second, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("second lookup refused while the holder answered inside the wait: %v", err)
	}
	defer second.Release()
	if second.Outcome != RouteLookupQueued || second.Wait < 40*time.Millisecond {
		t.Fatalf("second = %+v, want queued behind the holder", second)
	}
}

func TestRouteLookupGateShedsAfterTheQueueWait(t *testing.T) {
	t.Parallel()
	gate := testRouteLookupGate(1, 30*time.Millisecond, nil)
	holder, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer holder.Release()
	_, err = gate.Acquire(context.Background())
	var refusal *RouteLookupRefusal
	if !errors.As(err, &refusal) || !errors.Is(err, ErrRouteLookupShed) {
		t.Fatalf("err = %v, want a shed refusal", err)
	}
	if refusal.Outcome != RouteLookupShed || refusal.Wait < 25*time.Millisecond {
		t.Fatalf("refusal = %+v, want shed after the queue wait", refusal)
	}
}

// A lookup whose own context ends while it waits was cancelled, not shed:
// the handler answers a spent deadline as 504, not as a capacity 429.
func TestRouteLookupGateReportsACancelledWaitAsTheContextError(t *testing.T) {
	t.Parallel()
	for name, leases := range map[string]routeLookupLeases{
		"local": nil,
		"redis": newFakeRouteLookupLeases(1),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			// With a shared store the waiter is another instance, so its
			// wait is spent polling the lease rather than the local queue.
			holderGate := testRouteLookupGate(1, time.Second, leases)
			waiterGate := holderGate
			if leases != nil {
				waiterGate = testRouteLookupGate(1, time.Second, leases)
			}
			holder, err := holderGate.Acquire(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			defer holder.Release()
			lookup, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
			defer cancel()
			_, err = waiterGate.Acquire(lookup)
			if !errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrRouteLookupShed) {
				t.Fatalf("err = %v, want the lookup's deadline", err)
			}
		})
	}
}

// Two gates on one store are two router instances in one cell: together
// they admit max_inflight, not max_inflight each.
func TestRouteLookupGateIsCellWideOverTheSharedStore(t *testing.T) {
	t.Parallel()
	shared := newFakeRouteLookupLeases(1)
	instanceA := testRouteLookupGate(1, 30*time.Millisecond, shared)
	instanceB := testRouteLookupGate(1, 30*time.Millisecond, shared)
	holder, err := instanceA.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if holder.Backend != RouteLookupBackendRedis {
		t.Fatalf("holder = %+v, want a cell-wide lease", holder)
	}
	if _, err := instanceB.Acquire(context.Background()); !errors.Is(err, ErrRouteLookupShed) {
		t.Fatalf("second instance err = %v, want shed while the first holds the cell's slot", err)
	}
	holder.Release()
	holder.Release() // idempotent
	instanceA.releases.Wait()
	if shared.held() != 0 {
		t.Fatalf("held = %d after release, want 0", shared.held())
	}
	next, err := instanceB.Acquire(context.Background())
	if err != nil {
		t.Fatalf("second instance refused after the slot freed: %v", err)
	}
	next.Release()
}

func TestRouteLookupGateWaitsForTheCellSlot(t *testing.T) {
	t.Parallel()
	shared := newFakeRouteLookupLeases(1)
	instanceA := testRouteLookupGate(1, time.Second, shared)
	instanceB := testRouteLookupGate(1, time.Second, shared)
	holder, err := instanceA.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		time.Sleep(50 * time.Millisecond)
		holder.Release()
	}()
	waited, err := instanceB.Acquire(context.Background())
	if err != nil {
		t.Fatalf("refused while the other instance answered inside the wait: %v", err)
	}
	defer waited.Release()
	if waited.Outcome != RouteLookupQueued || waited.Backend != RouteLookupBackendRedis {
		t.Fatalf("waited = %+v, want queued on the cell-wide lease", waited)
	}
}

// A Redis failure admits on the per-process cap rather than refusing the
// lookup, then stops consulting Redis for the backoff and reports the trip
// once.
func TestRouteLookupGateFailsOpenToTheLocalCap(t *testing.T) {
	t.Parallel()
	shared := newFakeRouteLookupLeases(1)
	shared.failing.Store(true)
	gate := testRouteLookupGate(1, 30*time.Millisecond, shared)
	var trips atomic.Int64
	gate.SetFailOpenObserver(func(error) { trips.Add(1) })

	first, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("a Redis error refused the lookup: %v", err)
	}
	if first.Outcome != RouteLookupFailOpen || first.Backend != RouteLookupBackendLocal {
		t.Fatalf("first = %+v, want fail_open on the local gate", first)
	}
	// The local cap still binds while open.
	if _, err := gate.Acquire(context.Background()); !errors.Is(err, ErrRouteLookupShed) {
		t.Fatalf("err = %v, want the per-process cap to hold while Redis is down", err)
	}
	first.Release()
	calls := shared.calls.Load()
	second, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	second.Release()
	if shared.calls.Load() != calls {
		t.Fatal("the gate consulted Redis again inside the backoff")
	}
	if second.Outcome != RouteLookupFailOpen {
		t.Fatalf("second = %+v, want fail_open during the backoff", second)
	}
	if trips.Load() != 1 {
		t.Fatalf("fail-open reported %d times, want once per trip", trips.Load())
	}
}

// A reply lost after the write still took a slot; the gate releases the
// token so the slot does not sit out the TTL.
func TestRouteLookupGateReleasesASlotWhoseReplyWasLost(t *testing.T) {
	t.Parallel()
	shared := newFakeRouteLookupLeases(1)
	shared.writeThenFail.Store(true)
	gate := testRouteLookupGate(1, 30*time.Millisecond, shared)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Release()
	gate.releases.Wait()
	if shared.held() != 0 {
		t.Fatal("a lease whose reply was lost was left held until its TTL")
	}
}

// blockingRouteLookupLeases is a Redis that answers nothing until the call's
// context ends.
type blockingRouteLookupLeases struct{}

func (blockingRouteLookupLeases) acquire(ctx context.Context, _ string, _ time.Duration) (bool, error) {
	<-ctx.Done()
	return false, ctx.Err()
}

func (blockingRouteLookupLeases) release(context.Context, string) error { return nil }

// A Redis that accepts the call and never answers is bounded: the lookup is
// shed close to its queue wait (plus at most one minimum call), and the
// stall counts as a Redis failure, so the next lookups fail open instead of
// each waiting on it.
func TestRouteLookupGateBoundsAStalledCallAndTripsTheBackoff(t *testing.T) {
	t.Parallel()
	gate := testRouteLookupGate(1, 30*time.Millisecond, blockingRouteLookupLeases{})
	started := time.Now()
	_, err := gate.Acquire(context.Background())
	if !errors.Is(err, ErrRouteLookupShed) {
		t.Fatalf("err = %v, want shed at the queue wait", err)
	}
	if waited := time.Since(started); waited > 200*time.Millisecond {
		t.Fatalf("shed after %v, want about the queue wait plus one minimum call", waited)
	}
	if !gate.redisBackingOff() {
		t.Fatal("a Redis that never answered did not trip the backoff")
	}
	next, err := gate.Acquire(context.Background())
	if err != nil || next.Outcome != RouteLookupFailOpen {
		t.Fatalf("next = %+v, %v; want fail_open during the backoff", next, err)
	}
	next.Release()
}

// A stall with a long queue wait fails open after one call timeout rather
// than holding the lookup for the whole wait.
func TestRouteLookupGateFailsOpenAfterOneStalledCall(t *testing.T) {
	t.Parallel()
	gate := testRouteLookupGate(1, 5*time.Second, blockingRouteLookupLeases{})
	started := time.Now()
	ticket, err := gate.Acquire(context.Background())
	if err != nil || ticket.Outcome != RouteLookupFailOpen {
		t.Fatalf("ticket = %+v, %v; want fail_open", ticket, err)
	}
	ticket.Release()
	if waited := time.Since(started); waited > time.Second {
		t.Fatalf("failed open after %v, want about one call timeout", waited)
	}
}

// Release frees the local slot at once and the shared one in the background.
func TestRouteLookupGateReleasesTheSharedSlotOffThePath(t *testing.T) {
	t.Parallel()
	shared := newFakeRouteLookupLeases(1)
	gate := testRouteLookupGate(1, 30*time.Millisecond, shared)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ticket.Release()
	gate.releases.Wait()
	if shared.held() != 0 {
		t.Fatal("the shared slot was not released")
	}
	if err := gate.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRouteLookupPoolIDSeparatesBackends(t *testing.T) {
	t.Parallel()
	if RouteLookupPoolID("policy_service", "https://a") != RouteLookupPoolID("policy_service", "https://a") {
		t.Fatal("one backend computed two pools")
	}
	if RouteLookupPoolID("policy_service", "https://a") == RouteLookupPoolID("policy_service", "https://b") {
		t.Fatal("two backends share a pool")
	}
	if RouteLookupPoolID("policy_service", "https://a/") != RouteLookupPoolID("policy_service", "https://a") {
		t.Fatal("a trailing slash split one backend into two pools")
	}
	if RouteLookupPoolID("ab", "c") == RouteLookupPoolID("a", "bc") {
		t.Fatal("pool ids are ambiguous across field boundaries")
	}
}

// Against a real Redis: the Lua scripts take one slot per token, release
// only their own, and a crashed holder's slot frees itself at the TTL.
func TestRouteLookupGateOnRedis(t *testing.T) {
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	pool := RouteLookupPoolID(t.Name(), time.Now().String())
	storeA := newTestRedisEpisodeStore(t, address, "test:routes-a:", time.Second)
	storeB := newTestRedisEpisodeStore(t, address, "test:routes-b:", time.Second)
	build := func(store EpisodeStore, limit int, wait time.Duration) *RouteLookupGate {
		gate := NewRouteLookupGate(RouteLookupGateConfig{
			MaxInflight:    limit,
			QueueWait:      wait,
			LookupDeadline: 100 * time.Millisecond,
			Pool:           pool,
			Store:          store,
		})
		if !gate.Distributed() {
			t.Fatal("a Redis episode store built a per-process gate")
		}
		// Each gate stands for another router instance, i.e. another
		// process: its own local stage, the shared Redis slots.
		gate.local = admission.NewSemaphore(limit, routeLookupLocalQueue, 0, admission.OverflowShed)
		return gate
	}

	// Two slots across two instances, then a third is shed.
	instanceA := build(storeA, 2, 50*time.Millisecond)
	instanceB := build(storeB, 2, 50*time.Millisecond)
	first, err := instanceA.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	second, err := instanceB.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if first.Backend != RouteLookupBackendRedis || second.Backend != RouteLookupBackendRedis {
		t.Fatalf("tickets = %+v, %+v; want Redis leases", first, second)
	}
	if _, err := instanceB.Acquire(context.Background()); !errors.Is(err, ErrRouteLookupShed) {
		t.Fatalf("third lookup err = %v, want shed at the cell-wide cap", err)
	}
	// A replayed acquire (a reply lost after the write) returns the token's
	// own slot rather than reporting it busy or taking a second one.
	leases := instanceA.leases.(*redisRouteLookupLeases)
	replayed, err := leases.acquire(context.Background(), "replayed-token", time.Second)
	if err != nil || replayed {
		t.Fatalf("acquire with every slot held = %v, %v; want busy", replayed, err)
	}
	first.Release()
	instanceA.releases.Wait()
	for range 2 {
		got, err := leases.acquire(context.Background(), "replayed-token", time.Second)
		if err != nil || !got {
			t.Fatalf("replayed acquire = %v, %v; want the token's own slot", got, err)
		}
	}
	if _, err := instanceB.Acquire(context.Background()); !errors.Is(err, ErrRouteLookupShed) {
		t.Fatalf("err = %v, want shed: the replay must not have taken a second slot", err)
	}
	if err := leases.release(context.Background(), "replayed-token"); err != nil {
		t.Fatal(err)
	}
	third, err := instanceB.Acquire(context.Background())
	if err != nil {
		t.Fatalf("lookup refused after a slot freed: %v", err)
	}
	second.Release()
	third.Release()
	instanceB.releases.Wait()
	for _, gate := range []*RouteLookupGate{instanceA, instanceB} {
		if err := gate.Close(); err != nil {
			t.Fatal(err)
		}
	}

	// A crashed holder never releases; its slot frees at deadline + margin.
	crashed := build(storeA, 1, 50*time.Millisecond)
	crashed.leaseTTL = 150 * time.Millisecond
	if _, err := crashed.Acquire(context.Background()); err != nil {
		t.Fatal(err)
	}
	survivor := build(storeB, 1, time.Second)
	started := time.Now()
	ticket, err := survivor.Acquire(context.Background())
	if err != nil {
		t.Fatalf("a crashed holder's slot never freed: %v", err)
	}
	ticket.Release()
	if waited := time.Since(started); waited < 100*time.Millisecond {
		t.Fatalf("admitted after %v, before the crashed holder's TTL", waited)
	}
}

// Two recipes (or two router generations) calling one backend share the
// process's local cap, so with no shared store they still admit
// max_inflight together, not each.
func TestRouteLookupGateSharesTheLocalCapAcrossGatesOfOnePool(t *testing.T) {
	t.Parallel()
	pool := RouteLookupPoolID(t.Name())
	build := func() *RouteLookupGate {
		return NewRouteLookupGate(RouteLookupGateConfig{
			MaxInflight: 1, QueueWait: 30 * time.Millisecond, LookupDeadline: time.Second, Pool: pool,
		})
	}
	recipeA, recipeB := build(), build()
	holder, err := recipeA.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := recipeB.Acquire(context.Background()); !errors.Is(err, ErrRouteLookupShed) {
		t.Fatalf("err = %v, want the second recipe shed by the shared local cap", err)
	}
	holder.Release()
	next, err := recipeB.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	next.Release()
}

// A cell-wide slot reports the lease it holds, so an abandoned lookup's hold
// can end before it; a per-process slot has none.
func TestRouteLookupTicketReportsItsLease(t *testing.T) {
	t.Parallel()
	shared := testRouteLookupGate(1, 30*time.Millisecond, newFakeRouteLookupLeases(1))
	leased, err := shared.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer leased.Release()
	left, ok := leased.LeaseRemaining(time.Now())
	if !ok || left > shared.leaseTTL || left < shared.leaseTTL-time.Second {
		t.Fatalf("lease remaining = %v, %v; want about %v", left, ok, shared.leaseTTL)
	}
	local := testRouteLookupGate(1, 30*time.Millisecond, nil)
	ticket, err := local.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Release()
	if _, ok := ticket.LeaseRemaining(time.Now()); ok {
		t.Fatal("a per-process slot reported a lease")
	}
}
