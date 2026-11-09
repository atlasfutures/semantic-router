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
	decide := func(ctx context.Context, _ routerruntime.RouteDecisionRequest) (routerruntime.RouteDecision, error) {
		<-unblock
		backendSawCancel <- ctx.Err() != nil
		return routerruntime.RouteDecision{}, nil
	}
	lookup, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := raylineRoutesDecide(lookup, decide, routerruntime.RouteDecisionRequest{}, ticket, true); err != context.DeadlineExceeded {
		t.Fatalf("decide error = %v, want the lookup's deadline", err)
	}
	if _, err := gate.Acquire(context.Background()); err == nil {
		t.Fatal("the next lookup was admitted while the backend was still encoding")
	}
	close(unblock)
	if <-backendSawCancel {
		t.Fatal("the detached decision saw the caller's deadline")
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

// A tracked lookup is never detached: it decides on the lookup's own
// context, so an abandoned one cannot go on to commit its episode turn.
func TestRaylineRoutesDecideDoesNotDetachATrackedLookup(t *testing.T) {
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
	})
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	lookup, cancel := context.WithCancel(context.Background())
	decide := func(ctx context.Context, _ routerruntime.RouteDecisionRequest) (routerruntime.RouteDecision, error) {
		if ctx != lookup {
			t.Error("a tracked lookup decided on a context other than its own")
		}
		return routerruntime.RouteDecision{}, nil
	}
	if _, err := raylineRoutesDecide(lookup, decide, routerruntime.RouteDecisionRequest{}, ticket, false); err != nil {
		t.Fatalf("decide: %v", err)
	}
	cancel()
	next, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("slot not freed when a tracked lookup answered: %v", err)
	}
	next.Release()
}

// A caller that left while queued starts no encode.
func TestRaylineRoutesDecideSkipsACancelledLookup(t *testing.T) {
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
	if _, err := raylineRoutesDecide(lookup, decide, routerruntime.RouteDecisionRequest{}, ticket, true); err != context.Canceled {
		t.Fatalf("decide error = %v, want cancelled", err)
	}
	if called {
		t.Fatal("a cancelled lookup started an encode")
	}
	next, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatalf("slot not freed for a cancelled lookup: %v", err)
	}
	next.Release()
}
