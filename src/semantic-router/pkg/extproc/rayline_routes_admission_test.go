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
	"errors"
	"strings"
	"testing"
	"time"
)

func lookupGatedRouter(inflight, queueMS int) *OpenAIRouter {
	router := routesRouter(true)
	settings := &router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI
	settings.MaxInflightLookups = inflight
	settings.LookupQueueMS = queueMS
	return router
}

// Unset, the gate admits everything at once: a cell that has not opted in
// behaves exactly as before the setting existed.
func TestRoutesLookupGateUnsetAdmitsEverything(t *testing.T) {
	t.Parallel()
	gate := lookupGatedRouter(0, 0).raylineRoutesLookups()
	for range 10 {
		release, waited, err := gate.acquire(context.Background())
		if err != nil || waited > 50*time.Millisecond {
			t.Fatalf("unset gate refused or delayed: waited=%v err=%v", waited, err)
		}
		defer release()
	}
}

// With one slot, a second lookup waits for the first and then runs: two
// large lookups serialize instead of both holding the encoder.
func TestRoutesLookupGateSerializesLookups(t *testing.T) {
	t.Parallel()
	gate := lookupGatedRouter(1, 2000).raylineRoutesLookups()
	first, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("first lookup refused: %v", err)
	}
	admitted := make(chan time.Duration, 1)
	go func() {
		release, waited, err := gate.acquire(context.Background())
		if err != nil {
			t.Errorf("second lookup refused: %v", err)
			admitted <- -1
			return
		}
		release()
		admitted <- waited
	}()
	select {
	case <-admitted:
		t.Fatal("second lookup ran while the first held the only slot")
	case <-time.After(100 * time.Millisecond):
	}
	first()
	select {
	case waited := <-admitted:
		if waited < 100*time.Millisecond {
			t.Fatalf("second lookup waited %v, want it to have queued behind the first", waited)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("second lookup never ran after the first released")
	}
}

// A lookup that cannot get a slot within the queue time is answered 429 with
// a Retry-After long enough to outlast a typical large encode.
func TestRoutesLookupGateShedsWithRetryAfter(t *testing.T) {
	t.Parallel()
	router := lookupGatedRouter(1, 100)
	gate := router.raylineRoutesLookups()
	held, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("first lookup refused: %v", err)
	}
	defer held()
	_, waited, err := gate.acquire(context.Background())
	if !errors.Is(err, errRaylineRoutesLookupShed) {
		t.Fatalf("err = %v, want a shed", err)
	}
	if waited < 100*time.Millisecond {
		t.Fatalf("shed after %v, want after the 100 ms queue", waited)
	}
	response := router.raylineRoutesLookupRefused(routesContext(nil), context.Background(), "rte_1234abcd5678efab", waited, err)
	if code := immediateStatusCode(t, response); code != 429 {
		t.Fatalf("status = %d, want 429", code)
	}
	retry := ""
	for _, header := range response.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		if strings.EqualFold(header.GetHeader().GetKey(), "retry-after") {
			retry = string(header.GetHeader().GetRawValue())
		}
	}
	if retry != "5" {
		t.Fatalf("Retry-After = %q, want 5", retry)
	}
}

// The wait counts against the lookup's own deadline: a deadline shorter than
// the queue time ends the wait with the deadline's 504, not a 429.
func TestRoutesLookupGateWaitEndsAtTheDeadline(t *testing.T) {
	t.Parallel()
	router := lookupGatedRouter(1, 2000)
	gate := router.raylineRoutesLookups()
	held, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("first lookup refused: %v", err)
	}
	defer held()
	lookupContext, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, waited, err := gate.acquire(lookupContext)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want the deadline", err)
	}
	if waited > time.Second {
		t.Fatalf("waited %v, want the wait cut at the 50 ms deadline", waited)
	}
	response := router.raylineRoutesLookupRefused(routesContext(nil), lookupContext, "rte_1234abcd5678efab", waited, err)
	if code := immediateStatusCode(t, response); code != 504 {
		t.Fatalf("status = %d, want 504", code)
	}
}

// Beyond the queue depth, a burst is shed at once rather than queued, so a
// flood of large bodies cannot pile up in memory behind one encode.
func TestRoutesLookupGateShedsABurstBeyondTheQueue(t *testing.T) {
	t.Parallel()
	gate := lookupGatedRouter(1, 5000).raylineRoutesLookups()
	held, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("first lookup refused: %v", err)
	}
	defer held()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for range raylineRoutesLookupQueueDepth {
		go func() { _, _, _ = gate.acquire(ctx) }()
	}
	time.Sleep(100 * time.Millisecond)
	_, waited, err := gate.acquire(context.Background())
	if !errors.Is(err, errRaylineRoutesLookupShed) || waited > time.Second {
		t.Fatalf("burst lookup: waited=%v err=%v, want an immediate shed", waited, err)
	}
}
