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
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// countingRetrySelector answers every decision with a fixed arm or a fixed
// error, counting the calls that reached the backend. Safe for concurrent
// lookups.
type countingRetrySelector struct {
	// stubARCSelector supplies the rest of the selector interface; Select
	// is this type's own.
	stubARCSelector
	calls atomic.Int64
	err   error
	// backend, when set, is the decision backend's session slots: a call
	// takes one for its whole decision or fails at once when none is free,
	// as a session start on a full Server does.
	backend chan struct{}
	// hold, when set, keeps a call that took a slot inside its decision
	// until it is closed, unless the call arrived after passHold was set.
	hold     chan struct{}
	passHold atomic.Bool
	entered  chan struct{}
}

var errRetryBackendFull = errors.New("backend session slots are full")

func (s *countingRetrySelector) Select(ctx context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	s.calls.Add(1)
	if s.backend != nil {
		select {
		case s.backend <- struct{}{}:
			defer func() { <-s.backend }()
		default:
			return nil, errRetryBackendFull
		}
	}
	// Decided before entry is signalled, so the test's passHold (set after
	// it saw the entries) never releases a call that already entered.
	held := s.hold != nil && !s.passHold.Load()
	if s.entered != nil {
		s.entered <- struct{}{}
	}
	if held {
		<-s.hold
	}
	if s.err != nil {
		return nil, s.err
	}
	stub := &tierStubSelector{stubARCSelector: stubARCSelector{arm: 0, workers: oneCellWorkers()}}
	return stub.Select(ctx, selCtx)
}

func retryProtectionRouter(t *testing.T, selector *countingRetrySelector) *OpenAIRouter {
	t.Helper()
	selector.workers = oneCellWorkers()
	router, _ := oneCellRouter(t, "main")
	registry := selection.NewRegistry()
	registry.Register(selection.MethodRaylineARC, selector)
	router.RecipeModelSelectors[config.DefaultRecipeName] = registry
	router.ModelSelector = registry
	return router
}

const retryProtectionModel = "claude-sonnet-5"

// statelessLookups turns episode writes off, so every lookup decides with no
// episode state.
func statelessLookups(router *OpenAIRouter) {
	router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.EpisodeWrites = false
	router.Config.Recipes[0].Profile.Decisions[0].Algorithm.RaylineARC.RoutesAPI.EpisodeWrites = false
}

func retryLookup(router *OpenAIRouter, session string, body []byte) *ext_proc.ProcessingResponse {
	headers := map[string]string{}
	if session != "" {
		headers[raylineRoutesSessionHeader] = session
	}
	ctx := routesContext(headers)
	response := router.handleRaylineRoutesAPI(body, ctx)
	// As processRequestBody does on the way out.
	stampRaylineRetryHints(response, ctx)
	return response
}

func immediateHeaderNamed(response *ext_proc.ProcessingResponse, name string) string {
	return raylineRetryHintHeader(response.GetImmediateResponse(), name)
}

// One caller past its per-minute budget is answered 429 caller_budget with an
// integer Retry-After and x-should-retry: true, before the backend is called;
// another caller is unaffected.
func TestRaylineRoutesCallerBudgetTrips(t *testing.T) {
	selector := &countingRetrySelector{}
	router := retryProtectionRouter(t, selector)
	settings := config.RaylineARCRoutesCallerBudgetConfig{LookupsPerMinute: 2}
	router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.CallerBudget = settings
	router.Config.Recipes[0].Profile.Decisions[0].Algorithm.RaylineARC.RoutesAPI.CallerBudget = settings
	router.raylineRoutesCallerBudgets = buildRaylineRoutesCallerBudgets(router)
	if router.raylineRoutesCallerBudgets[config.DefaultRecipeName] == nil {
		t.Fatal("no caller budget was built for the default recipe")
	}
	body := oneCellBody(retryProtectionModel, false)
	// The per-process counts outlive one router (a reload shares them), so
	// each run is its own caller.
	caller := "user-1:" + strconv.FormatInt(time.Now().UnixNano(), 10)
	for index := range 2 {
		if status := immediateStatusCode(t, retryLookup(router, caller, body)); status != 200 {
			t.Fatalf("lookup %d within the budget: status %d", index, status)
		}
	}
	response := retryLookup(router, caller, body)
	if status, errType, code := routesErrorCode(t, response); status != 429 || errType != "rate_limit_error" ||
		code != raylineRoutesCallerBudgetCode {
		t.Fatalf("over budget = %d %s/%s, want 429 rate_limit_error/caller_budget", status, errType, code)
	}
	retryAfter, err := strconv.Atoi(routesRetryAfter(response))
	if err != nil || retryAfter < 1 || retryAfter > 60 {
		t.Fatalf("Retry-After = %q, want integer seconds within the window", routesRetryAfter(response))
	}
	if got := immediateHeaderNamed(response, raylineShouldRetryHeader); got != "true" {
		t.Fatalf("x-should-retry = %q, want true", got)
	}
	if calls := selector.calls.Load(); calls != 2 {
		t.Fatalf("backend called %d times, want 2: the refused lookup reached it", calls)
	}
	if status := immediateStatusCode(t, retryLookup(router, caller+"-other", body)); status != 200 {
		t.Fatalf("another caller: status %d, want 200", status)
	}
}

// With the episode Redis down the budget fails open: lookups are answered,
// not refused.
func TestRaylineRoutesCallerBudgetFailsOpenWhenRedisIsDown(t *testing.T) {
	selector := &countingRetrySelector{}
	router := retryProtectionRouter(t, selector)
	redisStore, err := raylinearc.NewRedisEpisodeStore(raylinearc.RedisEpisodeStoreConfig{
		Address: "127.0.0.1:1", KeyPrefix: "test:", LeaseTTL: time.Second, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = redisStore.Close() })
	router.RaylineARCEpisodeStore = unreadyRaylineARCEpisodeStore{redisStore}
	router.raylineRoutesCallerBudgets = buildRaylineRoutesCallerBudgets(router)
	budget := router.raylineRoutesCallerBudgets[config.DefaultRecipeName]
	if budget == nil || !budget.Distributed() {
		t.Fatalf("budget = %+v, want a cell-wide budget on the Redis store", budget)
	}
	t.Cleanup(func() { _ = budget.Close() })
	// Episode writes would also need the dead Redis: decide ephemerally.
	statelessLookups(router)
	if status := immediateStatusCode(t, retryLookup(router, "user-1:conv-1", oneCellBody(retryProtectionModel, false))); status != 200 {
		t.Fatalf("Redis down: status %d, want 200 (fail open)", status)
	}
}

// The lane keeps lookups from taking the backend's session slots: with the
// lane at one, a burst of held lookups occupies one slot and the rest are
// answered 429 at once, so a live turn's decision still finds a slot. Without
// the lane the same burst takes every slot and the live turn fails.
func TestRaylineRoutesLaneProtectsALiveTurn(t *testing.T) {
	for _, lane := range []bool{true, false} {
		t.Run(map[bool]string{true: "lane", false: "no lane"}[lane], func(t *testing.T) {
			selector := &countingRetrySelector{
				backend: make(chan struct{}, 2),
				hold:    make(chan struct{}),
				entered: make(chan struct{}, 8),
			}
			router := retryProtectionRouter(t, selector)
			if lane {
				router.raylineRoutesAdmission = map[config.RecipeName]*raylinearc.RouteLookupGate{
					config.DefaultRecipeName: raylinearc.NewRouteLookupGate(raylinearc.RouteLookupGateConfig{
						MaxInflight: 1, QueueWait: 20 * time.Millisecond, LookupDeadline: time.Second,
					}),
				}
			}
			body := oneCellBody(retryProtectionModel, false)
			var shed atomic.Int64
			var lookups sync.WaitGroup
			for index := range 4 {
				lookups.Add(1)
				go func() {
					defer lookups.Done()
					response := retryLookup(router, "user-1:conv-"+strconv.Itoa(index), body)
					if immediateStatusCode(t, response) == 429 {
						shed.Add(1)
					}
				}()
			}
			// The burst is in: every slot the lane allows is held.
			want := 1
			if !lane {
				want = 2
			}
			for range want {
				select {
				case <-selector.entered:
				case <-time.After(5 * time.Second):
					t.Fatal("no lookup reached the backend")
				}
			}
			if lane {
				deadline := time.Now().Add(5 * time.Second)
				for shed.Load() < 3 && time.Now().Before(deadline) {
					time.Sleep(5 * time.Millisecond)
				}
				if shed.Load() != 3 {
					t.Fatalf("%d lookups shed, want 3 past the lane", shed.Load())
				}
			}
			// The live turn: a decision on the same backend that does not
			// pass through the lookup handler, and answers at once.
			selector.passHold.Store(true)
			_, err := router.routeDecisionRuntimeState().RouteDecision(context.Background(),
				routerruntime.RouteDecisionRequest{Body: anthropicBody("live turn"), DecisionID: "rt_live", SessionID: "live-1"})
			close(selector.hold)
			lookups.Wait()
			router.background.Wait()
			if lane && err != nil {
				t.Fatalf("live turn failed behind a full lane: %v", err)
			}
			if !lane && err == nil {
				t.Fatal("without the lane the burst left a slot: the control does not exercise the starvation")
			}
		})
	}
}

// A lookup the decision refused with a deterministic 4xx is refused again
// from memory for its repeats, without the backend; a different body is
// decided, and a 5xx is never remembered.
func TestRaylineRoutesRefusalCache(t *testing.T) {
	t.Run("4xx is replayed", func(t *testing.T) {
		selector := &countingRetrySelector{err: &raylineARCSelectionFailure{class: arcFailurePolicyUndecidable}}
		router := retryProtectionRouter(t, selector)
		statelessLookups(router)
		router.raylineRoutesRefusals = buildRaylineRoutesRefusalCache(router)
		body := oneCellBody(retryProtectionModel, false)
		first := retryLookup(router, "user-1:conv-1", body)
		if status := immediateStatusCode(t, first); status != 422 {
			t.Fatalf("first: status %d, want 422", status)
		}
		if got := immediateHeaderNamed(first, raylineShouldRetryHeader); got != "false" {
			t.Fatalf("a deterministic refusal says x-should-retry %q, want false", got)
		}
		for range 3 {
			again := retryLookup(router, "user-1:conv-1", body)
			if status := immediateStatusCode(t, again); status != 422 {
				t.Fatalf("repeat: status %d, want 422", status)
			}
			if got := immediateHeaderNamed(again, raylineRefusalCacheHeader); got != "hit" {
				t.Fatalf("repeat was not replayed from the cache (%s = %q)", raylineRefusalCacheHeader, got)
			}
			if string(again.GetImmediateResponse().GetBody()) != string(first.GetImmediateResponse().GetBody()) {
				t.Fatal("the replayed refusal differs from the first")
			}
		}
		if calls := selector.calls.Load(); calls != 1 {
			t.Fatalf("backend called %d times, want once", calls)
		}
		// Another caller, or another body, is decided.
		retryLookup(router, "user-2:conv-1", body)
		retryLookup(router, "user-1:conv-1", oneCellBody(retryProtectionModel+"-x", false))
		if calls := selector.calls.Load(); calls != 3 {
			t.Fatalf("backend called %d times, want 3", calls)
		}
		// Past the TTL the request is decided again.
		router.raylineRoutesRefusals.now = func() time.Time { return time.Now().Add(61 * time.Second) }
		retryLookup(router, "user-1:conv-1", body)
		if calls := selector.calls.Load(); calls != 4 {
			t.Fatalf("an expired refusal was replayed (backend calls %d, want 4)", calls)
		}
	})
	t.Run("a tracked lookup's 4xx is not", func(t *testing.T) {
		// A tracked lookup's refusal can depend on its episode's state at
		// that moment, so the same request may be decided differently next.
		selector := &countingRetrySelector{err: &raylineARCSelectionFailure{class: arcFailurePolicyUndecidable}}
		router := retryProtectionRouter(t, selector)
		router.raylineRoutesRefusals = buildRaylineRoutesRefusalCache(router)
		body := oneCellBody(retryProtectionModel, false)
		for range 2 {
			if status := immediateStatusCode(t, retryLookup(router, "user-1:conv-1", body)); status != 422 {
				t.Fatalf("status %d, want 422", status)
			}
		}
		if calls := selector.calls.Load(); calls != 2 {
			t.Fatalf("backend called %d times, want 2: a tracked refusal was cached", calls)
		}
	})
	t.Run("5xx is not", func(t *testing.T) {
		selector := &countingRetrySelector{err: errors.New("policy server unavailable")}
		router := retryProtectionRouter(t, selector)
		router.raylineRoutesRefusals = buildRaylineRoutesRefusalCache(router)
		body := oneCellBody(retryProtectionModel, false)
		for range 2 {
			response := retryLookup(router, "user-1:conv-1", body)
			if status := immediateStatusCode(t, response); status != 503 {
				t.Fatalf("status %d, want 503", status)
			}
			if immediateHeaderNamed(response, raylineRefusalCacheHeader) != "" {
				t.Fatal("a 503 was replayed from the cache")
			}
		}
		if calls := selector.calls.Load(); calls != 2 {
			t.Fatalf("backend called %d times, want 2: a 5xx was cached", calls)
		}
	})
	t.Run("disabled", func(t *testing.T) {
		selector := &countingRetrySelector{err: &raylineARCSelectionFailure{class: arcFailurePolicyUndecidable}}
		router := retryProtectionRouter(t, selector)
		statelessLookups(router)
		router.raylineRoutesRefusals = buildRaylineRoutesRefusalCache(router)
		router.Config.Recipes[0].Profile.Decisions[0].Algorithm.RaylineARC.RoutesAPI.RefusalCache.Disabled = true
		router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.RefusalCache.Disabled = true
		body := oneCellBody(retryProtectionModel, false)
		retryLookup(router, "user-1:conv-1", body)
		retryLookup(router, "user-1:conv-1", body)
		if calls := selector.calls.Load(); calls != 2 {
			t.Fatalf("a disabled cache replayed: backend calls %d, want 2", calls)
		}
	})
}

// The cache is bounded: past max_entries the oldest refusal is dropped.
func TestRaylineRoutesRefusalCacheIsBounded(t *testing.T) {
	t.Parallel()
	cache := newRaylineRoutesRefusalCache(2)
	response := (&OpenAIRouter{}).createRaylineRoutesError(nil, 422, "invalid_request_error", "no")
	keys := [][32]byte{{1}, {2}, {3}}
	for _, key := range keys {
		cache.store(key, response, time.Minute)
	}
	if cache.lookup(keys[0]) != nil {
		t.Fatal("the oldest refusal survived past the bound")
	}
	if cache.lookup(keys[1]) == nil || cache.lookup(keys[2]) == nil {
		t.Fatal("a refusal within the bound was dropped")
	}
	for status, cacheable := range map[int]bool{400: true, 404: true, 422: true, 408: false, 409: false, 429: false, 500: false, 503: false} {
		if raylineRoutesRefusalCacheable(status) != cacheable {
			t.Errorf("status %d cacheable = %v, want %v", status, !cacheable, cacheable)
		}
	}
}

// Every non-2xx says whether and when to retry. On /v1/routes: 429 and 503
// carry Retry-After and x-should-retry: true, other 5xx Retry-After, other
// 4xx x-should-retry: false; a producer's own value is kept. On a live ARC
// turn a 503 carries Retry-After. Other traffic is left alone.
func TestRaylineRetryHintsOnEveryStatus(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{}
	routes := func() *RequestContext { return routesContext(nil) }
	turn := func() *RequestContext {
		return &RequestContext{
			Headers:            map[string]string{":path": "/v1/messages", ":method": "POST"},
			RaylineARCIdentity: &raylineARCIdentity{},
		}
	}
	other := func() *RequestContext {
		return &RequestContext{Headers: map[string]string{":path": "/v1/messages", ":method": "POST"}}
	}
	for name, testCase := range map[string]struct {
		ctx         func() *RequestContext
		status      int
		presetRetry string
		retryAfter  string
		shouldRetry string
	}{
		"routes 200":               {ctx: routes, status: 200},
		"routes 400":               {ctx: routes, status: 400, shouldRetry: "false"},
		"routes 404":               {ctx: routes, status: 404, shouldRetry: "false"},
		"routes 413":               {ctx: routes, status: 413, shouldRetry: "false"},
		"routes 422":               {ctx: routes, status: 422, shouldRetry: "false"},
		"routes 429":               {ctx: routes, status: 429, retryAfter: "1", shouldRetry: "true"},
		"routes 429 keeps its own": {ctx: routes, status: 429, presetRetry: "7", retryAfter: "7", shouldRetry: "true"},
		"routes 502":               {ctx: routes, status: 502, retryAfter: "5"},
		"routes 503":               {ctx: routes, status: 503, retryAfter: "5", shouldRetry: "true"},
		"routes 504":               {ctx: routes, status: 504, retryAfter: "5"},
		"turn 503":                 {ctx: turn, status: 503, retryAfter: "5"},
		"turn 503 keeps its own":   {ctx: turn, status: 503, presetRetry: "30", retryAfter: "30"},
		"turn 400":                 {ctx: turn, status: 400},
		"non-ARC 503":              {ctx: other, status: 503},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			response := router.createErrorResponse(testCase.status, "x")
			if testCase.presetRetry != "" {
				appendRetryAfterHeader(response, mustAtoi(t, testCase.presetRetry))
			}
			stampRaylineRetryHints(response, testCase.ctx())
			if got := immediateHeaderNamed(response, raylineRetryAfterHeader); got != testCase.retryAfter {
				t.Errorf("Retry-After = %q, want %q", got, testCase.retryAfter)
			}
			if got := immediateHeaderNamed(response, raylineShouldRetryHeader); got != testCase.shouldRetry {
				t.Errorf("x-should-retry = %q, want %q", got, testCase.shouldRetry)
			}
			count := 0
			for _, header := range response.GetImmediateResponse().GetHeaders().GetSetHeaders() {
				if header.GetHeader().GetKey() == raylineRetryAfterHeader {
					count++
				}
			}
			if testCase.retryAfter != "" && count != 1 {
				t.Errorf("%d Retry-After headers, want exactly one", count)
			}
		})
	}
}

func mustAtoi(t *testing.T, value string) int {
	t.Helper()
	parsed, err := strconv.Atoi(value)
	if err != nil {
		t.Fatal(err)
	}
	return parsed
}
