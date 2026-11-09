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
	if _, err := raylineRoutesDecide(lookup, &background, decide, routerruntime.RouteDecisionRequest{}, ticket, true); err != context.Canceled {
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

// A tracked lookup that holds the slot keeps it until the backend answers,
// as an ephemeral one does: the container keeps encoding after the caller's
// 504. Its decision runs on, uncancelled, but its commit guard has seen the
// caller leave and refuses the commit.
func TestRaylineRoutesDecideHoldsATrackedLookupsSlotUntilTheBackendAnswers(t *testing.T) {
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
	})
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	lookup, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered, unblock := make(chan struct{}), make(chan struct{})
	seen := make(chan [2]bool, 1)
	decide := func(ctx context.Context, request routerruntime.RouteDecisionRequest) (routerruntime.RouteDecision, error) {
		if request.Commit == nil || request.Commit.Context() != lookup {
			t.Error("a tracked lookup's decision was not given its caller's commit guard")
		}
		close(entered)
		<-unblock
		seen <- [2]bool{ctx.Err() != nil, !request.Commit.Begin()}
		return routerruntime.RouteDecision{}, nil
	}
	go func() { <-entered; cancel() }()
	var background sync.WaitGroup
	if _, err := raylineRoutesDecide(lookup, &background, decide, routerruntime.RouteDecisionRequest{}, ticket, false); err != context.Canceled {
		t.Fatalf("decide error = %v, want the caller's cancel", err)
	}
	if _, err := gate.Acquire(context.Background()); err == nil {
		t.Fatal("the slot was freed at the caller's 504 while the backend still encoded")
	}
	close(unblock)
	background.Wait()
	got := <-seen
	if got[0] {
		t.Fatal("the tracked decision was cancelled with its caller")
	}
	if !got[1] {
		t.Fatal("the commit guard let a decision commit for a caller that left")
	}
	awaitSlotFree(t, gate)
}

// A tracked lookup decided through the detached run commits its episode turn
// exactly once, and an ephemeral one is not handed a commit context.
func TestRaylineRoutesDecideCommitsATrackedLookupOnce(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
	})
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	var background sync.WaitGroup
	decision, err := raylineRoutesDecide(context.Background(), &background,
		router.routeDecisionRuntimeState().RouteDecision,
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), DecisionID: "rt_once", SessionID: "once-1"},
		ticket, false)
	if err != nil {
		t.Fatalf("decide: %v", err)
	}
	background.Wait()
	if decision.SelectedWorker == "" {
		t.Fatal("no worker selected")
	}
	prepared, cancelPrepare := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancelPrepare()
	lease, state, err := router.RaylineARCEpisodeStore.Prepare(prepared, raylinearc.HashEpisodeID("once-1"), 2)
	if err != nil {
		t.Fatalf("the episode lease was not released: %v", err)
	}
	defer func() { _ = router.RaylineARCEpisodeStore.Abort(context.Background(), lease) }()
	if state.TurnIndex != 1 {
		t.Fatalf("turn index = %d, want 1: the tracked lookup must commit exactly once", state.TurnIndex)
	}
	awaitSlotFree(t, gate)

	ticket, err = gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	ephemeral := func(_ context.Context, request routerruntime.RouteDecisionRequest) (routerruntime.RouteDecision, error) {
		if request.Commit != nil {
			t.Error("an ephemeral lookup was handed a commit guard")
		}
		return routerruntime.RouteDecision{}, nil
	}
	if _, err := raylineRoutesDecide(context.Background(), &background, ephemeral, routerruntime.RouteDecisionRequest{}, ticket, true); err != nil {
		t.Fatalf("decide: %v", err)
	}
	background.Wait()
}

// A decision whose caller left during its encode is released, not
// committed: its episode does not advance and its lease is free for the next
// turn.
func TestRouteDecisionDoesNotCommitForACallerThatLeft(t *testing.T) {
	caller, cancel := context.WithCancel(context.Background())
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers(), onSelect: cancel}
	router := decisionServiceRouter(t, selector)
	_, err := router.routeDecisionRuntimeState().RouteDecision(context.Background(), routerruntime.RouteDecisionRequest{
		Body: anthropicBody("hello"), DecisionID: "rt_left", SessionID: "left-1",
		Commit: newRaylineRoutesCommitGuard(caller),
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RouteDecision() error = %v, want the caller's cancel", err)
	}
	prepared, cancelPrepare := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancelPrepare()
	lease, state, err := router.RaylineARCEpisodeStore.Prepare(prepared, raylinearc.HashEpisodeID("left-1"), 2)
	if err != nil {
		t.Fatalf("the episode lease was not released: %v", err)
	}
	defer func() { _ = router.RaylineARCEpisodeStore.Abort(context.Background(), lease) }()
	if state.TurnIndex != 0 {
		t.Fatalf("turn index = %d, want 0: the episode advanced for a caller that left", state.TurnIndex)
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
		if _, err := raylineRoutesDecide(lookup, &sync.WaitGroup{}, decide, routerruntime.RouteDecisionRequest{}, ticket, ephemeral); err != context.Canceled {
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

// A tracked lookup that left a session is counted once per episode, and so
// is one whose caller left (releasing the episode does not close the
// session); an ephemeral lookup closes its own session and a decision that
// reported none left nothing, so neither is counted.
func TestRouteDecisionCountsTheSessionsTrackedLookupsHold(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers(), sessionAction: "created"}
	router := decisionServiceRouter(t, selector)
	router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.Enabled = true
	runtime := router.routeDecisionRuntimeState()
	decide := func(sessionID string, ephemeral bool, surface routerruntime.RouteDecisionSurface, guard routerruntime.RouteCommitGuard) {
		t.Helper()
		_, err := runtime.RouteDecision(context.Background(), routerruntime.RouteDecisionRequest{
			Body: anthropicBody("hello"), DecisionID: "rt_" + sessionID, SessionID: sessionID,
			Ephemeral: ephemeral, Surface: surface, Commit: guard,
		})
		if err != nil && guard == nil {
			t.Fatalf("RouteDecision(%s): %v", sessionID, err)
		}
	}
	routes := routerruntime.RouteDecisionSurfaceRoutes
	before := raylineRoutesLookupSessions.count(time.Now())
	decide("sessions-held-1", false, routes, nil)
	selector.sessionAction = "appended"
	decide("sessions-held-1", false, routes, nil)
	selector.sessionAction = "rebuilt"
	left, leave := context.WithCancel(context.Background())
	selector.onSelect = leave
	decide("sessions-held-2", false, routes, newRaylineRoutesCommitGuard(left))
	selector.onSelect = nil
	decide("sessions-held-3", true, routes, nil)
	decide("sessions-held-5", false, routerruntime.RouteDecisionSurfaceConsult, nil)
	selector.sessionAction = ""
	decide("sessions-held-4", false, routes, nil)
	if got := raylineRoutesLookupSessions.count(time.Now()) - before; got != 2 {
		t.Fatalf("lookup sessions grew by %d, want 2", got)
	}
}

// A tracked decision that started committing before its caller's deadline
// is answered even if the deadline passes before the answer is built: the
// episode recorded the turn, so a 504 would leave the caller without the arm
// its next turn is scored against.
func TestRaylineRoutesDecideAnswersATrackedLookupThatCommitted(t *testing.T) {
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
	})
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	lookup, cancel := context.WithCancel(context.Background())
	defer cancel()
	decide := func(_ context.Context, request routerruntime.RouteDecisionRequest) (routerruntime.RouteDecision, error) {
		if !request.Commit.Begin() {
			t.Error("the commit was refused for a caller still waiting")
		}
		cancel()
		time.Sleep(20 * time.Millisecond)
		request.Commit.End(true)
		time.Sleep(20 * time.Millisecond)
		return routerruntime.RouteDecision{SelectedWorker: "worker-a"}, nil
	}
	var background sync.WaitGroup
	decision, err := raylineRoutesDecide(lookup, &background, decide, routerruntime.RouteDecisionRequest{}, ticket, false)
	if err != nil || decision.SelectedWorker != "worker-a" {
		t.Fatalf("decide = %+v, %v; want the committed answer", decision, err)
	}
	background.Wait()
}

// A decision whose caller leaves while it waits for the episode lease stops
// waiting with its caller rather than taking the lease and encoding for
// nobody, and a guard whose caller left refuses a later commit.
func TestRouteDecisionStopsWaitingForTheLeaseWithItsCaller(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	held, _, err := router.RaylineARCEpisodeStore.Prepare(context.Background(), raylinearc.HashEpisodeID("wait-1"), 2)
	if err != nil {
		t.Fatalf("hold the lease: %v", err)
	}
	defer func() { _ = router.RaylineARCEpisodeStore.Abort(context.Background(), held) }()
	caller, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	guard := newRaylineRoutesCommitGuard(caller)
	started := time.Now()
	if _, err := router.routeDecisionRuntimeState().RouteDecision(context.Background(), routerruntime.RouteDecisionRequest{
		Body: anthropicBody("hello"), DecisionID: "rt_wait", SessionID: "wait-1", Commit: guard,
	}); err == nil {
		t.Fatal("a decision whose caller left took a held lease")
	}
	if waited := time.Since(started); waited > time.Second {
		t.Fatalf("waited %v for the lease after the caller left at 50ms", waited)
	}
	if len(selector.contexts) != 0 {
		t.Fatal("a decision whose caller left encoded")
	}
	if guard.Begin() {
		t.Fatal("the guard let a decision commit for a caller that left")
	}
}
