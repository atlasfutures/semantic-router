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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
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
		// The goroutine may start a little after the 100 ms window opens.
		if waited < 50*time.Millisecond {
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

// The wait stops short of the lookup's own deadline, so a wait that runs out
// answers 429 with a Retry-After instead of racing the deadline into a 504:
// with the shipped 1500 ms deadline and 2000 ms queue, a queued lookup sheds
// at about 1000 ms.
func TestRoutesLookupGateWaitStopsShortOfTheDeadline(t *testing.T) {
	t.Parallel()
	router := lookupGatedRouter(1, 2000)
	gate := router.raylineRoutesLookups()
	held, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("first lookup refused: %v", err)
	}
	defer held()
	lookupContext, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()
	_, waited, err := gate.acquire(lookupContext)
	if !errors.Is(err, errRaylineRoutesLookupShed) {
		t.Fatalf("err = %v, want a shed", err)
	}
	if lookupContext.Err() != nil {
		t.Fatal("the wait ran into the deadline")
	}
	if waited < 800*time.Millisecond || waited > 1300*time.Millisecond {
		t.Fatalf("waited %v, want about the 1000 ms left before the decision margin", waited)
	}
	response := router.raylineRoutesLookupRefused(routesContext(nil), lookupContext, "rte_1234abcd5678efab", waited, err)
	if code := immediateStatusCode(t, response); code != 429 {
		t.Fatalf("status = %d, want 429", code)
	}
}

// A lookup whose context has already ended answers the deadline's 504, never
// a 429, whatever the queue says.
func TestRoutesLookupGateEndedContextIsTheDeadline(t *testing.T) {
	t.Parallel()
	router := lookupGatedRouter(1, 2000)
	gate := router.raylineRoutesLookups()
	held, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("first lookup refused: %v", err)
	}
	defer held()
	lookupContext, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	<-lookupContext.Done()
	_, waited, err := gate.acquire(lookupContext)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want the deadline", err)
	}
	response := router.raylineRoutesLookupRefused(routesContext(nil), lookupContext, "rte_1234abcd5678efab", waited, err)
	if code := immediateStatusCode(t, response); code != 504 {
		t.Fatalf("status = %d, want 504", code)
	}
}

// An abandoned ephemeral lookup keeps its slot until the policy service
// answers: the container is still encoding it, so the next lookup must not
// be admitted onto it at the first lookup's deadline.
func TestRoutesLookupGateHoldsTheSlotPastAnAbandonedLookup(t *testing.T) {
	t.Parallel()
	gate := lookupGatedRouter(1, 2000).raylineRoutesLookups()
	release, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("first lookup refused: %v", err)
	}
	encodeDone := make(chan struct{})
	decisionCancelled := make(chan bool, 1)
	lookupContext, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err = gate.decide(lookupContext, release, true, func(decisionContext context.Context) (routerruntime.RouteDecision, error) {
		<-encodeDone
		decisionCancelled <- decisionContext.Err() != nil
		return routerruntime.RouteDecision{}, nil
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want the caller's deadline", err)
	}
	if _, _, err := gate.acquire(context.Background()); !errors.Is(err, errRaylineRoutesLookupShed) {
		t.Fatalf("second lookup err = %v, want a shed while the first still encodes", err)
	}
	close(encodeDone)
	if <-decisionCancelled {
		t.Fatal("the abandoned decision was cancelled with its caller")
	}
	second, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("second lookup refused after the first finished: %v", err)
	}
	second()
}

// A caller already gone does not take a free slot: the semaphore admits
// without reading the context, and an admitted ephemeral lookup would start
// a detached encode nobody waits for.
func TestRoutesLookupGateRefusesAGoneCallerAFreeSlot(t *testing.T) {
	t.Parallel()
	gate := lookupGatedRouter(1, 2000).raylineRoutesLookups()
	gone, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := gate.acquire(gone); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want the caller's cancellation", err)
	}
	release, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("the refused caller kept the slot: %v", err)
	}
	release()
}

// Detaching follows the decision service's own ephemeral rule: exploration,
// caller prices and exclusive commit nothing, so they are detached too.
func TestRoutesLookupCommitsNothingMirrorsTheService(t *testing.T) {
	t.Parallel()
	for name, testCase := range map[string]struct {
		request routerruntime.RouteDecisionRequest
		want    bool
	}{
		"tracked":     {request: routerruntime.RouteDecisionRequest{}, want: false},
		"ephemeral":   {request: routerruntime.RouteDecisionRequest{Ephemeral: true}, want: true},
		"no controls": {request: routerruntime.RouteDecisionRequest{Routing: &selection.RaylineARCRouteControls{}}, want: false},
		"exploration": {request: routerruntime.RouteDecisionRequest{Routing: &selection.RaylineARCRouteControls{Exploration: &raylinearc.PolicyExplorationRequest{}}}, want: true},
		"prices":      {request: routerruntime.RouteDecisionRequest{Routing: &selection.RaylineARCRouteControls{Prices: &raylinearc.PolicyRoutingPrices{}}}, want: true},
		"exclusive":   {request: routerruntime.RouteDecisionRequest{Routing: &selection.RaylineARCRouteControls{Exclusive: true}}, want: true},
	} {
		if got := raylineRoutesCommitsNothing(testCase.request); got != testCase.want {
			t.Errorf("%s: commits nothing = %v, want %v", name, got, testCase.want)
		}
	}
}

// A tracked lookup is never detached: one its caller abandoned must not go on
// to commit its episode.
func TestRoutesLookupGateDoesNotDetachTrackedLookups(t *testing.T) {
	t.Parallel()
	gate := lookupGatedRouter(1, 2000).raylineRoutesLookups()
	release, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("lookup refused: %v", err)
	}
	lookupContext, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = gate.decide(lookupContext, release, false, func(decisionContext context.Context) (routerruntime.RouteDecision, error) {
		return routerruntime.RouteDecision{}, decisionContext.Err()
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want the tracked decision to see its caller's cancellation", err)
	}
	next, _, err := gate.acquire(context.Background())
	if err != nil {
		t.Fatalf("slot not released after a tracked lookup: %v", err)
	}
	next()
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
