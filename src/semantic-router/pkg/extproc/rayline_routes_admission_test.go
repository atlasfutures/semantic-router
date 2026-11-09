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
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
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
	if _, err := raylineRoutesDecide(lookup, &sync.WaitGroup{}, decide, routerruntime.RouteDecisionRequest{}, ticket, false); err != nil {
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
	if _, err := raylineRoutesDecide(lookup, &sync.WaitGroup{}, decide, routerruntime.RouteDecisionRequest{}, ticket, true); err != context.Canceled {
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

// appendAdmissionFirst and appendAdmissionNext are one conversation's first
// request and the next one, which extends it by a reply and a short message.
const (
	appendAdmissionFirstMessages = `[{"role":"user","content":"hi"}]`
	appendAdmissionFirst         = `{"model":"claude","max_tokens":64,"messages":` + appendAdmissionFirstMessages + `}`
	appendAdmissionNext          = `{"model":"claude","max_tokens":64,"messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":"hello"},{"role":"user","content":"next"}]}`
)

// appendAdmissionRouter is a tracked policy-service cell whose lookups are
// gated by one slot, which the test can hold, and decided by a stub selector.
// appendExempt sets routes_api.admission.append_exempt.
func appendAdmissionRouter(t *testing.T, appendExempt bool, maxSuffixTokens int) (*OpenAIRouter, *raylinearc.RouteLookupGate, *stubARCSelector) {
	t.Helper()
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	arc := router.Config.Decisions[0].Algorithm.RaylineARC
	arc.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"}
	arc.RoutesAPI = config.RaylineARCRoutesAPIConfig{
		Enabled:       true,
		EpisodeWrites: true,
		Admission: config.RaylineARCRoutesAdmissionConfig{
			MaxInflight: 1, QueueWaitMS: 40, AppendExempt: appendExempt, AppendMaxSuffixTokens: maxSuffixTokens,
		},
	}
	gate := raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
		MaxInflight: 1, QueueWait: 40 * time.Millisecond, LookupDeadline: time.Second,
	})
	router.raylineRoutesAdmission = map[config.RecipeName]*raylinearc.RouteLookupGate{config.DefaultRecipeName: gate}
	return router, gate, selector
}

// seedAppendAdmissionEpisode commits the conversation's first request as the
// episode's policy prefix and records it as the service's retained session,
// as a strict first turn this process decided leaves both.
func seedAppendAdmissionEpisode(t *testing.T, router *OpenAIRouter, session string) {
	t.Helper()
	store := router.RaylineARCEpisodeStore
	lease, state, err := store.Prepare(context.Background(), raylinearc.HashEpisodeID(session), 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := state.Commit(0, 10, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	messages, _, err := policyMessages(json.RawMessage(appendAdmissionFirstMessages))
	if err != nil {
		t.Fatal(err)
	}
	state.Policy = &raylinearc.PolicyEpisodeState{
		PrefixLen:    len(messages),
		PrefixDigest: raylinearc.NewPrefixDigests(messages).Digest(len(messages)),
	}
	if err := store.Commit(context.Background(), lease, lease.Version(), state); err != nil {
		t.Fatal(err)
	}
	scope := policyRetainedScope(router.Config.Decisions[0].Algorithm.RaylineARC)
	episode := raylinearc.HashEpisodeID(session)
	policyRetainedSessions.remember(scope, episode, raylinearc.NewPolicyPrefixShape(messages, nil, nil))
	t.Cleanup(func() { policyRetainedSessions.forget(scope, episode) })
}

// holdAppendAdmissionGate takes the gate's one slot, as a full encode would.
func holdAppendAdmissionGate(t *testing.T, gate *raylinearc.RouteLookupGate) {
	t.Helper()
	holder, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(holder.Release)
}

func trackedRoutesContext(session string) *RequestContext {
	return routesContext(map[string]string{raylineRoutesSessionHeader: session, raylineRoutesFormatHeader: "anthropic"})
}

// A tracked lookup that extends its episode's committed prefix by a short
// suffix appends to the retained session, so it is decided while a full
// encode holds the only slot.
func TestRaylineRoutesExemptsAPredictedAppendFromAHeldSlot(t *testing.T) {
	router, gate, selector := appendAdmissionRouter(t, true, 0)
	seedAppendAdmissionEpisode(t, router, t.Name())
	holdAppendAdmissionGate(t, gate)

	response := router.handleRaylineRoutesAPI([]byte(appendAdmissionNext), trackedRoutesContext(t.Name()))
	if code := immediateStatusCode(t, response); code != 200 {
		t.Fatalf("status = %d (%s), want the exempt append decided", code, response.GetImmediateResponse().GetBody())
	}
	if len(selector.contexts) != 1 {
		t.Fatalf("selector ran %d times, want once", len(selector.contexts))
	}
}

// A conversation's first tracked lookup has no retained session to append
// to: it is a full encode and waits for the slot like any other.
func TestRaylineRoutesGatesAFirstTrackedLookup(t *testing.T) {
	router, gate, selector := appendAdmissionRouter(t, true, 0)
	holdAppendAdmissionGate(t, gate)

	response := router.handleRaylineRoutesAPI([]byte(appendAdmissionFirst), trackedRoutesContext(t.Name()))
	if status, _, code := routesErrorCode(t, response); status != 429 || code != raylineRoutesAdmissionCode {
		t.Fatalf("first tracked lookup = %d %s, want 429 %s", status, code, raylineRoutesAdmissionCode)
	}
	if len(selector.contexts) != 0 {
		t.Fatal("a shed lookup reached the selector")
	}
	// The refusal released the episode lease: the conversation's next
	// lookup, once a slot is free, is not contended.
	free, _, _ := appendAdmissionRouter(t, true, 0)
	free.RaylineARCEpisodeStore = router.RaylineARCEpisodeStore
	free.ModelSelector = router.ModelSelector
	response = free.handleRaylineRoutesAPI([]byte(appendAdmissionFirst), trackedRoutesContext(t.Name()))
	if code := immediateStatusCode(t, response); code != 200 {
		t.Fatalf("lookup after a shed = %d (%s), want 200", code, response.GetImmediateResponse().GetBody())
	}
}

// An append whose suffix exceeds append_max_suffix_tokens is gated as the
// encode it is.
func TestRaylineRoutesGatesAnAppendOverTheSuffixBound(t *testing.T) {
	router, gate, _ := appendAdmissionRouter(t, true, 1)
	seedAppendAdmissionEpisode(t, router, t.Name())
	holdAppendAdmissionGate(t, gate)

	response := router.handleRaylineRoutesAPI([]byte(appendAdmissionNext), trackedRoutesContext(t.Name()))
	if status, _, code := routesErrorCode(t, response); status != 429 || code != raylineRoutesAdmissionCode {
		t.Fatalf("over-bound append = %d %s, want 429 %s", status, code, raylineRoutesAdmissionCode)
	}
}

// With append_exempt off every lookup is gated, a predicted append included.
func TestRaylineRoutesGatesEveryLookupWithAppendExemptOff(t *testing.T) {
	router, gate, selector := appendAdmissionRouter(t, false, 0)
	seedAppendAdmissionEpisode(t, router, t.Name())
	holdAppendAdmissionGate(t, gate)

	response := router.handleRaylineRoutesAPI([]byte(appendAdmissionNext), trackedRoutesContext(t.Name()))
	if status, _, code := routesErrorCode(t, response); status != 429 || code != raylineRoutesAdmissionCode {
		t.Fatalf("append with the knob off = %d %s, want 429 %s", status, code, raylineRoutesAdmissionCode)
	}
	if len(selector.contexts) != 0 {
		t.Fatal("a shed lookup reached the selector")
	}
}

// An ephemeral lookup is gated before any episode work, as before, whatever
// append_exempt says, and holds its slot until the backend answers.
func TestRaylineRoutesGatesAnEphemeralLookupWithAppendExemptOn(t *testing.T) {
	router, gate, selector := appendAdmissionRouter(t, true, 0)
	seedAppendAdmissionEpisode(t, router, t.Name())
	holdAppendAdmissionGate(t, gate)

	// No session header: ephemeral, even though the body extends conv-1.
	response := router.handleRaylineRoutesAPI([]byte(appendAdmissionNext), routesContext(map[string]string{raylineRoutesFormatHeader: "anthropic"}))
	if status, _, code := routesErrorCode(t, response); status != 429 || code != raylineRoutesAdmissionCode {
		t.Fatalf("ephemeral lookup = %d %s, want 429 %s", status, code, raylineRoutesAdmissionCode)
	}
	if len(selector.contexts) != 0 {
		t.Fatal("a shed ephemeral lookup reached the selector")
	}
}

// The prediction reads only what the router holds before the encode.
func TestPredictRaylineRoutesEncode(t *testing.T) {
	policy := &config.AlgorithmConfig{RaylineARC: &config.RaylineARCAlgorithmConfig{
		PolicyService: &config.RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"},
	}}
	first, _, err := policyMessages(json.RawMessage(appendAdmissionFirstMessages))
	if err != nil {
		t.Fatal(err)
	}
	committed := &raylinearc.EpisodeState{Policy: &raylinearc.PolicyEpisodeState{
		PrefixLen: 1, PrefixDigest: raylinearc.NewPrefixDigests(first).Digest(1),
	}}
	episode := raylinearc.HashEpisodeID("predict-" + t.Name())
	scope := policyRetainedScope(policy.RaylineARC)
	retain := func(shape *raylinearc.PolicyPrefixShape) { policyRetainedSessions.remember(scope, episode, shape) }
	t.Cleanup(func() { policyRetainedSessions.forget(scope, episode) })
	arcContext := func(body string, state *raylinearc.EpisodeState) *selection.RaylineARCSelectionContext {
		return &selection.RaylineARCSelectionContext{
			EpisodeIDHash: episode,
			RawRequest:    []byte(body),
			RequestFormat: policyFormatAnthropic,
			State:         state,
		}
	}
	// Nothing this process decided is in the session: a full encode.
	if unknown := predictRaylineRoutesEncode(policy, arcContext(appendAdmissionNext, committed), false); unknown.Append ||
		unknown.Reason != raylineRoutesPredictNotRetained {
		t.Fatalf("unrecorded session = %+v, want a full encode (%s)", unknown, raylineRoutesPredictNotRetained)
	}
	// The session holds the prefix under another system prompt: a rebuild.
	retain(raylinearc.NewPolicyPrefixShape(first, json.RawMessage(`"be brief"`), nil))
	if rebuilt := predictRaylineRoutesEncode(policy, arcContext(appendAdmissionNext, committed), false); rebuilt.Append ||
		rebuilt.Reason != raylineRoutesPredictShellChanged {
		t.Fatalf("changed system prompt = %+v, want a full encode (%s)", rebuilt, raylineRoutesPredictShellChanged)
	}
	// The committed prefix is not the request the session last retained, as
	// after a zdr turn (decided on a throwaway episode) or another decide.
	older, _, err := policyMessages(json.RawMessage(`[{"role":"user","content":"before"}]`))
	if err != nil {
		t.Fatal(err)
	}
	retain(raylinearc.NewPolicyPrefixShape(older, nil, nil))
	if stale := predictRaylineRoutesEncode(policy, arcContext(appendAdmissionNext, committed), false); stale.Append ||
		stale.Reason != raylineRoutesPredictNotRetained {
		t.Fatalf("stale retained request = %+v, want a full encode (%s)", stale, raylineRoutesPredictNotRetained)
	}
	retain(raylinearc.NewPolicyPrefixShape(first, nil, nil))
	appended := predictRaylineRoutesEncode(policy, arcContext(appendAdmissionNext, committed), false)
	if !appended.Append || appended.Reason != raylineRoutesPredictAppends || appended.SuffixTokens <= 0 {
		t.Fatalf("extending request = %+v, want an append with its suffix counted", appended)
	}
	if want := raylineRoutesSuffixTokens([]json.RawMessage{
		json.RawMessage(`{"role":"assistant","content":"hello"}`), json.RawMessage(`{"role":"user","content":"next"}`),
	}); appended.SuffixTokens != want {
		t.Fatalf("suffix tokens = %d, want %d (the two new messages)", appended.SuffixTokens, want)
	}
	broken := `{"messages":[{"role":"user","content":"other"},{"role":"user","content":"next"}]}`
	for name, testCase := range map[string]struct {
		algorithm *config.AlgorithmConfig
		context   *selection.RaylineARCSelectionContext
		ephemeral bool
		reason    string
	}{
		"no committed turn": {policy, arcContext(appendAdmissionNext, &raylinearc.EpisodeState{}), false, raylineRoutesPredictNoCommittedTurn},
		"prefix break":      {policy, arcContext(broken, committed), false, raylinearc.PolicyTransitionPrefixBreak},
		"ephemeral":         {policy, arcContext(appendAdmissionNext, committed), true, raylineRoutesPredictEphemeralRequest},
		"artifact mode": {
			&config.AlgorithmConfig{RaylineARC: &config.RaylineARCAlgorithmConfig{}},
			arcContext(appendAdmissionNext, committed), false, raylineRoutesPredictArtifactMode,
		},
		"relaxed episodes": {
			&config.AlgorithmConfig{RaylineARC: &config.RaylineARCAlgorithmConfig{
				PolicyService: policy.RaylineARC.PolicyService,
				Episode:       config.RaylineARCEpisodeConfig{Consistency: config.RaylineARCConsistencyRelaxed},
			}},
			arcContext(appendAdmissionNext, committed), false, raylineRoutesPredictRelaxed,
		},
	} {
		t.Run(name, func(t *testing.T) {
			got := predictRaylineRoutesEncode(testCase.algorithm, testCase.context, testCase.ephemeral)
			if got.Append || got.Reason != testCase.reason {
				t.Fatalf("prediction = %+v, want a full encode for %s", got, testCase.reason)
			}
		})
	}
}

// predictAfterFirstTurn runs and commits one turn on fixture, then predicts
// the lookup that follows it with a reply and a tool result.
func predictAfterFirstTurn(t *testing.T, fixture *deltaFixture) routerruntime.RouteEncodePrediction {
	t.Helper()
	scope := policyRetainedScope(fixture.decision.Algorithm.RaylineARC)
	t.Cleanup(func() { policyRetainedSessions.forget(scope, fixture.episode) })
	_, ctx := fixture.turn(t, policyTestRequest(t, deltaUser))
	complete(t, ctx, deltaReply(t))
	lease, state, err := fixture.store.Prepare(context.Background(), fixture.episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = fixture.store.Abort(context.Background(), lease) }()
	return predictRaylineRoutesEncode(fixture.decision.Algorithm, &selection.RaylineARCSelectionContext{
		EpisodeIDHash: fixture.episode,
		RawRequest:    policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult),
		RequestFormat: policyFormatAnthropic,
		State:         state,
	}, false)
}

// A strict turn this process decided leaves its request in the service's
// session, so the next request is predicted to append to it.
func TestPredictRaylineRoutesEncodeAfterAStrictTurn(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeSync, nil)
	if prediction := predictAfterFirstTurn(t, fixture); !prediction.Append {
		t.Fatalf("after a strict turn = %+v, want an append", prediction)
	}
}

// A zdr turn commits its prefix but is decided on a throwaway service
// episode, so nothing retains it: the next request is a full encode.
func TestPredictRaylineRoutesEncodeAfterAZDRTurn(t *testing.T) {
	releaseInline(t)
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeSync, nil)
	fixture.zdr = true
	fixture.fake.servesRelease = true
	if prediction := predictAfterFirstTurn(t, fixture); prediction.Append || prediction.Reason != raylineRoutesPredictNotRetained {
		t.Fatalf("after a zdr turn = %+v, want a full encode (%s)", prediction, raylineRoutesPredictNotRetained)
	}
}
