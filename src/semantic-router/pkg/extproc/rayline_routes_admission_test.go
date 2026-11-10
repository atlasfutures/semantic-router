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
	"strings"
	"sync"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const routesAdmissionBody = `{"messages":[{"role":"user","content":"hi"}]}`

// routesRouterWithGate is a routes-enabled router whose default recipe is
// gated by one slot, which the test can hold.
func routesRouterWithGate(t *testing.T, wait time.Duration) (*OpenAIRouter, *raylinearc.RouteLookupGate) {
	t.Helper()
	router := routesRouter(true)
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight:    1,
		QueueWait:      wait,
		LookupDeadline: time.Second,
	})
	router.raylineRoutesAdmission = map[config.RecipeName]*raylinearc.RouteLookupGate{
		config.DefaultRecipeName: gate,
	}
	return router, gate
}

func routesRetryAfter(response *ext_proc.ProcessingResponse) string {
	for _, header := range response.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		if strings.EqualFold(header.GetHeader().GetKey(), "retry-after") {
			return string(header.GetHeader().GetRawValue())
		}
	}
	return ""
}

// A lookup that finds the cell's slot taken for its whole queue wait is
// answered 429 lookup_capacity with a Retry-After, before any encoder work.
func TestRaylineRoutesShedsALookupPastTheQueueWait(t *testing.T) {
	t.Parallel()
	router, gate := routesRouterWithGate(t, 1200*time.Millisecond)
	holder, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer holder.Release()

	response := router.handleRaylineRoutesAPI([]byte(routesAdmissionBody), routesContext(nil))
	status, errType, code := routesErrorCode(t, response)
	if status != 429 {
		t.Fatalf("status = %d, want 429", status)
	}
	if errType != "rate_limit_error" || code != raylineRoutesAdmissionCode {
		t.Fatalf("error = %s/%s, want rate_limit_error/%s", errType, code, raylineRoutesAdmissionCode)
	}
	// The queue wait rounded up: coming back sooner mostly fails again.
	if got := routesRetryAfter(response); got != "2" {
		t.Fatalf("Retry-After = %q, want 2", got)
	}
}

// The slot is held only while the lookup answers: a lookup that fails still
// gives it back, so the next one is admitted at once.
func TestRaylineRoutesReleasesTheSlotWhenTheLookupAnswers(t *testing.T) {
	t.Parallel()
	router, gate := routesRouterWithGate(t, 30*time.Millisecond)
	response := router.handleRaylineRoutesAPI([]byte(routesAdmissionBody), routesContext(nil))
	if code := immediateStatusCode(t, response); code == 429 {
		t.Fatal("a lookup on a free gate was shed")
	}
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("the slot was not released after the lookup answered: %v", err)
	}
	ticket.Release()
	if ticket.Outcome != raylinearc.RouteLookupAdmitted {
		t.Fatalf("outcome = %s, want admitted at once", ticket.Outcome)
	}
}

// A dispatched turn never passes through the gate: with every lookup slot
// held, a routed request is answered exactly as it would be without one.
func TestRaylineRoutesAdmissionLeavesDispatchedTurnsAlone(t *testing.T) {
	t.Parallel()
	router, gate := routesRouterWithGate(t, 30*time.Millisecond)
	holder, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer holder.Release()

	ctx := &RequestContext{Headers: map[string]string{":path": "/v1/messages", ":method": "POST"}}
	started := time.Now()
	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(`{"model":"rayline-router"}`)},
	}, ctx)
	if err != nil {
		t.Fatalf("HandleRequestBody() error = %v", err)
	}
	body := string(response.GetImmediateResponse().GetBody())
	if strings.Contains(body, raylineRoutesAdmissionCode) || immediateStatusCode(t, response) == 429 {
		t.Fatalf("a routed request was refused by lookup admission: %s", body)
	}
	if !strings.Contains(body, "invalid inference request") {
		t.Fatalf("routed request answered with %s, want the ingress refusal", body)
	}
	if waited := time.Since(started); waited >= 30*time.Millisecond {
		t.Fatalf("routed request took %v, as if it had queued at the gate", waited)
	}
}

// The three "come back later" answers carry different codes, so a caller can
// tell a spent deadline and a contended lease from lookup capacity.
func TestRaylineRoutesTimeoutAndContentionAreCoded(t *testing.T) {
	t.Parallel()
	router := routesRouter(true)
	expired, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	response := router.raylineRoutesFailure(nil, expired, "rte_1234abcd", context.DeadlineExceeded)
	if status, errType, code := routesErrorCode(t, response); status != 504 || errType != "timeout_error" || code != "deadline_exceeded" {
		t.Fatalf("expired lookup = %d %s/%s, want 504 timeout_error/deadline_exceeded", status, errType, code)
	}
	response = router.raylineRoutesFailure(nil, context.Background(), "rte_1234abcd", routerruntime.ErrRouteDecisionContended)
	if status, _, code := routesErrorCode(t, response); status != 429 || code != "contended" {
		t.Fatalf("contended lookup = %d %s, want 429 contended", status, code)
	}
}

// With a memory episode store there is nothing to share, so the gate is per
// process; a Redis store makes it cell-wide. Recipes calling one backend
// share a pool.
func TestRaylineRoutesAdmissionScopeFollowsTheEpisodeBackend(t *testing.T) {
	t.Parallel()
	admission := config.RaylineARCRoutesAdmissionConfig{MaxInflight: 1}
	router := routesRouter(true)
	router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.Admission = admission
	router.Config.Decisions[0].Algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"}

	memoryStore, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{
		MaxEpisodes: 8, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	router.RaylineARCEpisodeStore = memoryStore
	gates := buildRaylineRoutesAdmission(router)
	if gate := gates[config.DefaultRecipeName]; gate == nil || gate.Distributed() {
		t.Fatalf("memory backend gate = %+v, want a per-process gate", gate)
	}

	redisStore, err := raylinearc.NewRedisEpisodeStore(raylinearc.RedisEpisodeStoreConfig{
		Address: "127.0.0.1:1", KeyPrefix: "test:", LeaseTTL: time.Second, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = redisStore.Close() })
	router.RaylineARCEpisodeStore = unreadyRaylineARCEpisodeStore{redisStore}
	gates = buildRaylineRoutesAdmission(router)
	if gate := gates[config.DefaultRecipeName]; gate == nil || !gate.Distributed() {
		t.Fatal("a Redis episode store (even one unready at startup) built a per-process gate")
	}

	disabled := routesRouter(true)
	if gates := buildRaylineRoutesAdmission(disabled); len(gates) != 0 {
		t.Fatalf("no admission config built gates: %v", gates)
	}

	sameBackend := &config.RaylineARCAlgorithmConfig{PolicyService: &config.RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"}}
	otherBackend := &config.RaylineARCAlgorithmConfig{PolicyService: &config.RaylineARCPolicyServiceConfig{BaseURL: "https://other.example"}}
	if raylineRoutesAdmissionPool(router.Config.Decisions[0].Algorithm.RaylineARC) != raylineRoutesAdmissionPool(sameBackend) {
		t.Fatal("two recipes on one policy service computed two pools")
	}
	trailingSlash := &config.RaylineARCAlgorithmConfig{PolicyService: &config.RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example/"}}
	if raylineRoutesAdmissionPool(sameBackend) != raylineRoutesAdmissionPool(trailingSlash) {
		t.Fatal("two spellings of one policy service computed two pools")
	}
	oneReplica := &config.RaylineARCAlgorithmConfig{PolicyService: &config.RaylineARCPolicyServiceConfig{
		Replicas: []config.RaylineARCEncoderReplicaConfig{{ID: "a", BaseURL: "https://policy.example"}},
	}}
	if raylineRoutesAdmissionPool(sameBackend) != raylineRoutesAdmissionPool(oneReplica) {
		t.Fatal("base_url A and replicas [A] computed two pools")
	}
	fleetA := &config.RaylineARCAlgorithmConfig{Episode: config.RaylineARCEpisodeConfig{KeyPrefix: "fleet-a:"}}
	fleetB := &config.RaylineARCAlgorithmConfig{Episode: config.RaylineARCEpisodeConfig{KeyPrefix: "fleet-b:"}}
	if raylineRoutesAdmissionPool(fleetA) == raylineRoutesAdmissionPool(fleetB) {
		t.Fatal("two dynamic-membership encoder fleets share a pool")
	}
	if raylineRoutesAdmissionPool(sameBackend) == raylineRoutesAdmissionPool(otherBackend) {
		t.Fatal("two policy services share a pool")
	}
}

// An ephemeral lookup whose deadline fires mid-encode answers at its
// deadline, but its slot stays held until the backend answers: the backend
// is still encoding, and the next lookup must not be admitted onto it.
func TestRaylineRoutesDecideHoldsTheSlotUntilTheBackendAnswers(t *testing.T) {
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
	})
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	unblock := make(chan struct{})
	backendSawCancel := make(chan bool, 1)
	entered := make(chan struct{})
	decide := func(ctx context.Context, _ routerruntime.RouteDecisionRequest) (routerruntime.RouteDecision, error) {
		close(entered)
		<-unblock
		backendSawCancel <- ctx.Err() != nil
		return routerruntime.RouteDecision{}, nil
	}
	// The caller gives up only once the backend is encoding.
	lookup, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { <-entered; cancel() }()
	var background sync.WaitGroup
	if _, err := raylineRoutesDecide(lookup, &background, decide, routerruntime.RouteDecisionRequest{}, ticket, 0, true); err != context.Canceled {
		t.Fatalf("decide error = %v, want the caller's cancel", err)
	}
	// A reload's Router.Close waits on background: it must not see the
	// detached run as finished while the backend is still encoding.
	drained := make(chan struct{})
	go func() { background.Wait(); close(drained) }()
	select {
	case <-drained:
		t.Fatal("the router's background work drained while the decision still ran")
	case <-time.After(20 * time.Millisecond):
	}
	if _, err := gate.Acquire(context.Background()); err == nil {
		t.Fatal("the next lookup was admitted while the backend was still encoding")
	}
	close(unblock)
	select {
	case sawCancel := <-backendSawCancel:
		if sawCancel {
			t.Fatal("the detached decision saw the caller's cancel")
		}
	case <-time.After(time.Second):
		t.Fatal("the detached decision never returned")
	}
	select {
	case <-drained:
	case <-time.After(time.Second):
		t.Fatal("the router's background work never drained")
	}
	deadline := time.Now().Add(time.Second)
	for {
		next, err := gate.Acquire(context.Background())
		if err == nil {
			next.Release()
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("slot never freed after the backend answered: %v", err)
		}
	}
}

// A tracked lookup whose caller left mid-encode keeps its slot for the
// encode's estimated duration, then frees it; it never commits its turn.
func TestRaylineRoutesDecideHoldsAnAbandonedTrackedLookupsSlot(t *testing.T) {
	defer func(fixed time.Duration) { raylineRoutesAbandonedHoldFixed = fixed }(raylineRoutesAbandonedHoldFixed)
	raylineRoutesAbandonedHoldFixed = 50 * time.Millisecond
	lookup, cancel := context.WithCancel(context.Background())
	defer cancel()
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers(), onSelect: cancel}
	router := decisionServiceRouter(t, selector)
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
	})
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	// 4000 bytes is 1000 tokens: at 4000 tokens a second, 250 ms.
	body := anthropicBody(strings.Repeat("x", 4000-len(anthropicBody(""))))
	started := time.Now()
	if _, err := raylineRoutesDecide(lookup, &sync.WaitGroup{}, router.routeDecisionRuntimeState().RouteDecision,
		routerruntime.RouteDecisionRequest{Body: body, DecisionID: "rt_abandoned", SessionID: "abandoned-1"},
		ticket, 4000, false); err == nil {
		t.Fatal("an abandoned tracked lookup answered")
	}
	if _, err := gate.Acquire(context.Background()); err == nil {
		t.Fatal("the slot was freed at once while the encode still ran")
	}
	awaitSlotFree(t, gate)
	if held := time.Since(started); held < 250*time.Millisecond || held > 900*time.Millisecond {
		t.Fatalf("slot held %v, want about 300 ms", held)
	}
	prepared, cancelPrepare := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancelPrepare()
	lease, state, err := router.RaylineARCEpisodeStore.Prepare(prepared, raylinearc.HashEpisodeID("abandoned-1"), 2)
	if err != nil {
		t.Fatalf("the episode lease was not released: %v", err)
	}
	defer func() { _ = router.RaylineARCEpisodeStore.Abort(context.Background(), lease) }()
	if state.TurnIndex != 0 {
		t.Fatalf("turn index = %d, want 0: an abandoned lookup committed", state.TurnIndex)
	}
}

// A tracked lookup that answers frees its slot at once and commits exactly
// once.
func TestRaylineRoutesDecideFreesATrackedLookupsSlotOnAnswer(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
	})
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if _, err := raylineRoutesDecide(context.Background(), &sync.WaitGroup{}, router.routeDecisionRuntimeState().RouteDecision,
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), DecisionID: "rt_once", SessionID: "once-1"},
		ticket, 0, false); err != nil {
		t.Fatalf("decide: %v", err)
	}
	next, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("slot not freed when a tracked lookup answered: %v", err)
	}
	next.Release()
	prepared, cancelPrepare := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancelPrepare()
	lease, state, err := router.RaylineARCEpisodeStore.Prepare(prepared, raylinearc.HashEpisodeID("once-1"), 2)
	if err != nil {
		t.Fatalf("the episode lease was not released: %v", err)
	}
	defer func() { _ = router.RaylineARCEpisodeStore.Abort(context.Background(), lease) }()
	if state.TurnIndex != 1 {
		t.Fatalf("turn index = %d, want 1: a tracked lookup commits exactly once", state.TurnIndex)
	}
}

// The hold is the encode's tokens at the configured rate plus the fixed
// part, from the reported tokens when there are any and the body otherwise,
// and never longer than RouteLookupDetachBound.
func TestRaylineRoutesAbandonedHoldIsEstimatedAndCapped(t *testing.T) {
	if got := raylineRoutesAbandonedHold(70000, 35000, 0, 0, false); got != 3*time.Second {
		t.Fatalf("hold = %v, want 3s", got)
	}
	if got := raylineRoutesAbandonedHold(70000, 0, 0, 0, false); got != 3*time.Second {
		t.Fatalf("hold at the default rate = %v, want 3s", got)
	}
	if got := raylineRoutesAbandonedHold(100_000_000, 35000, 0, 0, false); got != raylinearc.RouteLookupDetachBound {
		t.Fatalf("hold = %v, want the %v cap", got, raylinearc.RouteLookupDetachBound)
	}
	// Counted from when the decision began: 3 s of encode, 2 s already spent.
	if got := raylineRoutesAbandonedHold(70000, 35000, 2*time.Second, 0, false); got != time.Second {
		t.Fatalf("hold after 2s = %v, want 1s", got)
	}
	if got := raylineRoutesAbandonedHold(100_000_000, 35000, 30*time.Second, 0, false); got != raylinearc.RouteLookupDetachBound-30*time.Second {
		t.Fatalf("capped hold after 30s = %v, want the cap less 30s", got)
	}
	if got := raylineRoutesAbandonedHold(70000, 35000, 5*time.Second, 0, false); got > 0 {
		t.Fatalf("hold after the estimate = %v, want none", got)
	}
	if got := raylineRoutesEncodeTokens(routerruntime.RouteDecision{}, make([]byte, 4000)); got != 1000 {
		t.Fatalf("tokens from the body = %d, want 1000", got)
	}
	measured := routerruntime.RouteDecision{Usage: routerruntime.RouteUsage{EncodedInputTokens: 30, CacheReadTokens: 12}}
	if got := raylineRoutesEncodeTokens(measured, make([]byte, 4000)); got != 42 {
		t.Fatalf("measured tokens = %d, want 42", got)
	}
}

// awaitSlotFree waits for gate's one slot to be free.
func awaitSlotFree(t *testing.T, gate *raylinearc.RouteLookupGate) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		next, err := gate.Acquire(context.Background())
		if err == nil {
			next.Release()
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("slot never freed: %v", err)
		}
	}
}

// A caller that left while queued starts no encode, tracked or ephemeral.
func TestRaylineRoutesDecideSkipsACancelledLookup(t *testing.T) {
	for _, ephemeral := range []bool{true, false} {
		gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
			MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
		})
		ticket, err := gate.Acquire(context.Background())
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		lookup, cancel := context.WithCancel(context.Background())
		cancel()
		called := false
		decide := func(context.Context, routerruntime.RouteDecisionRequest) (routerruntime.RouteDecision, error) {
			called = true
			return routerruntime.RouteDecision{}, nil
		}
		if _, err := raylineRoutesDecide(lookup, &sync.WaitGroup{}, decide, routerruntime.RouteDecisionRequest{}, ticket, 0, ephemeral); err != context.Canceled {
			t.Fatalf("ephemeral=%v: decide error = %v, want cancelled", ephemeral, err)
		}
		if called {
			t.Fatalf("ephemeral=%v: a cancelled lookup started an encode", ephemeral)
		}
		next, err := gate.Acquire(context.Background())
		if err != nil {
			t.Fatalf("ephemeral=%v: slot not freed for a cancelled lookup: %v", ephemeral, err)
		}
		next.Release()
	}
}

// The lookup-sessions count is the distinct episodes whose tracked lookups
// left a retained encoder session, each for the episode idle TTL: a second
// lookup on the episode is the same session, and an entry past its TTL is
// gone.
func TestRaylineRoutesLookupSessionTrackerCountsDistinctLiveSessions(t *testing.T) {
	tracker := &raylineRoutesLookupSessionTracker{expiry: map[string]time.Time{}}
	now := time.Unix(1000, 0)
	tracker.record("package:a", "ep-1", time.Minute, now)
	tracker.record("package:a", "ep-1", time.Minute, now.Add(30*time.Second))
	if held := tracker.record("package:a", "ep-2", time.Minute, now); held != 2 {
		t.Fatalf("held = %d, want 2", held)
	}
	if held := tracker.record("package:b", "ep-1", time.Minute, now); held != 3 {
		t.Fatalf("held = %d, want 3: another package holds its own session", held)
	}
	if got := tracker.count(now.Add(61 * time.Second)); got != 1 {
		t.Fatalf("count = %d, want 1: only ep-1 on package:a was renewed", got)
	}
	if got := tracker.count(now.Add(2 * time.Minute)); got != 0 {
		t.Fatalf("count = %d, want 0 past every idle TTL", got)
	}
}

// A tracked /v1/routes lookup that left a session is counted once per
// episode; an ephemeral lookup closes its own session, a decision that
// reported none left nothing, and a legacy consult is not a lookup, so none
// of those is counted.
func TestRouteDecisionCountsTheSessionsTrackedLookupsHold(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers(), sessionAction: "created"}
	router := decisionServiceRouter(t, selector)
	router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.Enabled = true
	runtime := router.routeDecisionRuntimeState()
	decide := func(sessionID string, ephemeral bool, surface routerruntime.RouteDecisionSurface) {
		t.Helper()
		_, err := runtime.RouteDecision(context.Background(), routerruntime.RouteDecisionRequest{
			Body: anthropicBody("hello"), DecisionID: "rt_" + sessionID, SessionID: sessionID,
			Ephemeral: ephemeral, Surface: surface,
		})
		if err != nil {
			t.Fatalf("RouteDecision(%s): %v", sessionID, err)
		}
	}
	routes := routerruntime.RouteDecisionSurfaceRoutes
	before := raylineRoutesLookupSessions.count(time.Now())
	decide("sessions-held-1", false, routes)
	selector.sessionAction = "appended"
	decide("sessions-held-1", false, routes)
	selector.sessionAction = "rebuilt"
	decide("sessions-held-2", false, routes)
	decide("sessions-held-3", true, routes)
	decide("sessions-held-5", false, routerruntime.RouteDecisionSurfaceConsult)
	selector.sessionAction = ""
	decide("sessions-held-4", false, routes)
	if got := raylineRoutesLookupSessions.count(time.Now()) - before; got != 2 {
		t.Fatalf("lookup sessions grew by %d, want 2", got)
	}
}

// The hold never outlives the slot's cell-wide lease: it ends a margin before
// the lease runs out, at once when that is already past.
func TestRaylineRoutesAbandonedHoldEndsBeforeTheLease(t *testing.T) {
	if got := raylineRoutesAbandonedHold(100_000_000, 35000, 30*time.Second, 60*time.Second, true); got != 59*time.Second {
		t.Fatalf("hold = %v, want 59s: the lease's 60s less the margin", got)
	}
	if got := raylineRoutesAbandonedHold(70000, 35000, 0, 10*time.Second, true); got != 3*time.Second {
		t.Fatalf("hold = %v, want the 3s estimate inside the lease", got)
	}
	if got := raylineRoutesAbandonedHold(70000, 35000, 0, 500*time.Millisecond, true); got > 0 {
		t.Fatalf("hold = %v, want none with the lease nearly out", got)
	}
}

// A delayed release is router background work: a reload's Close waits for
// it rather than closing the gate under it.
func TestRaylineRoutesAbandonedHoldIsBackgroundWork(t *testing.T) {
	defer func(fixed time.Duration) { raylineRoutesAbandonedHoldFixed = fixed }(raylineRoutesAbandonedHoldFixed)
	raylineRoutesAbandonedHoldFixed = 150 * time.Millisecond
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
	})
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	lookup, cancel := context.WithCancel(context.Background())
	decide := func(context.Context, routerruntime.RouteDecisionRequest) (routerruntime.RouteDecision, error) {
		cancel()
		return routerruntime.RouteDecision{}, context.Canceled
	}
	var background sync.WaitGroup
	if _, err := raylineRoutesDecide(lookup, &background, decide, routerruntime.RouteDecisionRequest{}, ticket, 0, false); err == nil {
		t.Fatal("an abandoned lookup answered")
	}
	drained := make(chan struct{})
	go func() { background.Wait(); close(drained) }()
	select {
	case <-drained:
		t.Fatal("background drained before the delayed release ran")
	case <-time.After(50 * time.Millisecond):
	}
	select {
	case <-drained:
	case <-time.After(time.Second):
		t.Fatal("background never drained")
	}
	next, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("slot not freed once background drained: %v", err)
	}
	next.Release()
}

// A tracked decision that panics frees its slot, and the panic goes on up.
func TestRaylineRoutesDecideReleasesATrackedLookupsSlotOnPanic(t *testing.T) {
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
	})
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	decide := func(context.Context, routerruntime.RouteDecisionRequest) (routerruntime.RouteDecision, error) {
		panic("decision failed")
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("the panic was swallowed")
			}
		}()
		_, _ = raylineRoutesDecide(context.Background(), &sync.WaitGroup{}, decide, routerruntime.RouteDecisionRequest{}, ticket, 0, false)
	}()
	next, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("slot leaked by a panicking decision: %v", err)
	}
	next.Release()
}
