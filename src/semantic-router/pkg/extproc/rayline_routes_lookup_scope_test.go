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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The scope is the user and the conversation, namespaced away from the
// conversation's routed episode; there is none with the setting off or no
// conversation named.
func TestRaylineRoutesLookupScopeDerivation(t *testing.T) {
	on := &config.RaylineARCRoutesAPIConfig{Enabled: true, LookupSessions: true}
	off := &config.RaylineARCRoutesAPIConfig{Enabled: true}
	if got := raylineRoutesLookupScope(on, "user-1", "conv-1"); got != "lookup:user-1:conv-1" {
		t.Fatalf("scope = %q", got)
	}
	if got := raylineRoutesLookupScope(on, "", "conv-1"); got != "lookup::conv-1" {
		t.Fatalf("scope without an authenticated user = %q", got)
	}
	for name, got := range map[string]string{
		"setting off":     raylineRoutesLookupScope(off, "user-1", "conv-1"),
		"no conversation": raylineRoutesLookupScope(on, "user-1", ""),
		"no settings":     raylineRoutesLookupScope(nil, "user-1", "conv-1"),
	} {
		if got != "" {
			t.Fatalf("%s: scope = %q, want none", name, got)
		}
	}
	scoped := raylinearc.HashEpisodeID(raylineRoutesLookupScope(on, "user-1", "conv-1"))
	if scoped == raylinearc.HashEpisodeID("conv-1") {
		t.Fatal("the lookup scope hashes to the conversation's routed episode")
	}
	if scoped == raylinearc.HashEpisodeID(raylineRoutesLookupScope(on, "user-2", "conv-1")) {
		t.Fatal("two users share a lookup scope")
	}
}

// lookupScopeSelector is a stub selector that, when failScoped is set,
// answers a lookup-scoped selection as one its scope cannot serve.
type lookupScopeSelector struct {
	*stubARCSelector
	mu         sync.Mutex
	failScoped bool
}

func (selector *lookupScopeSelector) Select(
	ctx context.Context,
	selCtx *selection.SelectionContext,
) (*selection.SelectionResult, error) {
	selector.mu.Lock()
	defer selector.mu.Unlock()
	if selector.failScoped && selCtx.RaylineARC != nil && selCtx.RaylineARC.RoutesLookupSession {
		selector.contexts = append(selector.contexts, selCtx)
		return nil, &raylineARCSelectionFailure{class: arcFailureRoutesLookupSessionFallback}
	}
	return selector.stubARCSelector.Select(ctx, selCtx)
}

type lookupScopeFixture struct {
	router   *OpenAIRouter
	selector *lookupScopeSelector
	mu       sync.Mutex
	closed   []string
}

// newLookupScopeFixture is a policy-service cell with the route lookup on and
// a session closer that records the episodes it is asked to close.
func newLookupScopeFixture(t *testing.T) *lookupScopeFixture {
	t.Helper()
	selector := &lookupScopeSelector{stubARCSelector: &stubARCSelector{
		arm: 0, workers: decisionServiceWorkers(), sessionAction: "appended",
	}}
	router := decisionServiceRouter(t, selector.stubARCSelector)
	registry := selection.NewRegistry()
	registry.Register(selection.MethodRaylineARC, selector)
	router.ModelSelector = registry
	arc := router.Config.Decisions[0].Algorithm.RaylineARC
	arc.RoutesAPI.Enabled = true
	arc.RoutesAPI.LookupSessions = true
	arc.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"}
	fixture := &lookupScopeFixture{router: router, selector: selector}
	router.raylineARCSessionClose = func(_ context.Context, episode string, _ []string) (raylinearc.EncoderCloseReport, error) {
		fixture.mu.Lock()
		defer fixture.mu.Unlock()
		fixture.closed = append(fixture.closed, episode)
		return raylinearc.EncoderCloseReport{}, nil
	}
	return fixture
}

func (fixture *lookupScopeFixture) decide(t *testing.T, request routerruntime.RouteDecisionRequest) routerruntime.RouteDecision {
	t.Helper()
	if request.Body == nil {
		request.Body = anthropicBody("hello")
	}
	if request.DecisionID == "" {
		request.DecisionID = "rt_lookup_scope"
	}
	decision, err := fixture.router.routeDecisionRuntimeState().RouteDecision(context.Background(), request)
	if err != nil {
		t.Fatalf("RouteDecision: %v", err)
	}
	// The ephemeral close runs in the router's background work.
	fixture.router.background.Wait()
	return decision
}

func (fixture *lookupScopeFixture) closedEpisodes() []string {
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	return append([]string(nil), fixture.closed...)
}

func (fixture *lookupScopeFixture) lastContext() *selection.RaylineARCSelectionContext {
	contexts := fixture.selector.contexts
	return contexts[len(contexts)-1].RaylineARC
}

const lookupScopeTestScope = "lookup:user-1:conv-1"

// Which lookups decide on their lookup scope: an ephemeral /v1/routes lookup
// on a policy-service cell that names one, including one the service makes
// ephemeral (caller prices, exclusive). The scoped lookup decides under the
// scope's stable hash and closes nothing; every other lookup is decided as
// before, under a minted id it closes.
func TestRouteDecisionClassifiesLookupScopedLookups(t *testing.T) {
	routes, consult := routerruntime.RouteDecisionSurfaceRoutes, routerruntime.RouteDecisionSurfaceConsult
	for _, tc := range []struct {
		name     string
		request  routerruntime.RouteDecisionRequest
		artifact bool
		scoped   bool
		closes   bool
	}{
		{name: "ephemeral lookup with a scope", scoped: true,
			request: routerruntime.RouteDecisionRequest{Surface: routes, Ephemeral: true, SessionID: "conv-1", LookupScope: lookupScopeTestScope}},
		{name: "lookup made ephemeral by exclusive", scoped: true,
			request: routerruntime.RouteDecisionRequest{Surface: routes, SessionID: "conv-1", LookupScope: lookupScopeTestScope,
				Routing: &selection.RaylineARCRouteControls{Exclusive: true}}},
		{name: "ephemeral lookup without a scope", closes: true,
			request: routerruntime.RouteDecisionRequest{Surface: routes, Ephemeral: true, SessionID: "conv-1"}},
		{name: "tracked lookup",
			request: routerruntime.RouteDecisionRequest{Surface: routes, SessionID: "conv-1", LookupScope: lookupScopeTestScope}},
		{name: "legacy consult", closes: true,
			request: routerruntime.RouteDecisionRequest{Surface: consult, Ephemeral: true, LookupScope: lookupScopeTestScope}},
		{name: "artifact cell", artifact: true, closes: true,
			request: routerruntime.RouteDecisionRequest{Surface: routes, Ephemeral: true, LookupScope: lookupScopeTestScope}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newLookupScopeFixture(t)
			if tc.artifact {
				fixture.router.Config.Decisions[0].Algorithm.RaylineARC.PolicyService = nil
			}
			decision := fixture.decide(t, tc.request)
			arcContext := fixture.lastContext()
			scopeHash := raylinearc.HashEpisodeID(lookupScopeTestScope)
			if arcContext.RoutesLookupSession != tc.scoped || (arcContext.EpisodeIDHash == scopeHash) != tc.scoped {
				t.Fatalf("lookup session = %t on %s, want scoped %t", arcContext.RoutesLookupSession, arcContext.EpisodeIDHash, tc.scoped)
			}
			if decision.Encode.LookupSession != tc.scoped || decision.Encode.SessionAction != "appended" {
				t.Fatalf("encode = %+v, want lookup session %t", decision.Encode, tc.scoped)
			}
			// Still ephemeral: the scoped lookup committed no episode turn.
			if tc.scoped && decision.Episode != nil {
				t.Fatalf("a scoped lookup committed an episode: %+v", decision.Episode)
			}
			closed := fixture.closedEpisodes()
			if tc.closes != (len(closed) == 1) || (tc.closes && closed[0] != arcContext.EpisodeIDHash) {
				t.Fatalf("closed %v, want a close of the minted episode %t", closed, tc.closes)
			}
			for _, episode := range closed {
				if episode == scopeHash {
					t.Fatal("a lookup closed its lookup scope's session")
				}
			}
		})
	}
}

// A lookup whose scope cannot be served is decided again one-shot, under a
// minted id it closes; the scope's session, which a concurrent lookup may
// hold, is never closed.
func TestRouteDecisionFallsBackToAOneShotLookup(t *testing.T) {
	fixture := newLookupScopeFixture(t)
	fixture.selector.failScoped = true
	decision := fixture.decide(t, routerruntime.RouteDecisionRequest{
		Surface: routerruntime.RouteDecisionSurfaceRoutes, Ephemeral: true, SessionID: "conv-1", LookupScope: lookupScopeTestScope,
	})
	contexts := fixture.selector.contexts
	if len(contexts) != 2 || !contexts[0].RaylineARC.RoutesLookupSession || contexts[1].RaylineARC.RoutesLookupSession {
		t.Fatalf("selections = %d, want a scoped attempt then a one-shot one", len(contexts))
	}
	scopeHash := raylinearc.HashEpisodeID(lookupScopeTestScope)
	oneShot := contexts[1].RaylineARC.EpisodeIDHash
	if oneShot == scopeHash || oneShot == "" {
		t.Fatalf("the fallback decided on %q, want a minted episode", oneShot)
	}
	if closed := fixture.closedEpisodes(); len(closed) != 1 || closed[0] != oneShot {
		t.Fatalf("closed %v, want only the one-shot episode", closed)
	}
	if decision.Encode.LookupSession || decision.SelectedWorker != "worker-a" {
		t.Fatalf("decision = %+v, want the one-shot answer", decision)
	}
}

// A selection the scope's own refusal ended is the only one decided again:
// any other failure of a scoped lookup is the lookup's.
func TestRouteDecisionDoesNotRetryOtherScopedFailures(t *testing.T) {
	fixture := newLookupScopeFixture(t)
	fixture.selector.err = &raylineARCSelectionFailure{class: "policy_service_transport"}
	_, err := fixture.router.routeDecisionRuntimeState().RouteDecision(context.Background(), routerruntime.RouteDecisionRequest{
		Body: anthropicBody("hello"), DecisionID: "rt_scope_err",
		Surface: routerruntime.RouteDecisionSurfaceRoutes, Ephemeral: true, LookupScope: lookupScopeTestScope,
	})
	if err == nil || len(fixture.selector.contexts) != 1 {
		t.Fatalf("err = %v after %d selections, want one failed selection", err, len(fixture.selector.contexts))
	}
}

// scopedLookupOn selects as a lookup-scoped throwaway lookup.
func (fixture *policySelectorFixture) scopedLookupOn(t *testing.T, scope string) (*selection.SelectionResult, error) {
	t.Helper()
	state, _ := raylinearc.NewEpisodeState(2)
	arcContext := &selection.RaylineARCSelectionContext{
		RequestID:       "req-lookup-scope",
		State:           state,
		RawRequest:      lookupBody(t),
		RequestFormat:   policyFormatAnthropic,
		RoutesLookup:    true,
		RoutesThrowaway: true,
	}
	if !applyRaylineRoutesLookupScope(arcContext, scope) {
		t.Fatal("the lookup was not placed on its scope")
	}
	return fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC:      arcContext,
	})
}

func advertiseLookupSessions(fixture *policySelectorFixture) {
	scorer := fixture.selector.armedComponents().scorer.(*policyServiceScorer)
	scorer.setAdvertised([]string{raylinearc.PolicyCapabilityLookupSessions})
}

// A scoped lookup is sent strict with session_scope "lookup" on the scope's
// hash, on a relaxed cell too, and records no turn for a later one to hold.
func TestScopedLookupSendsTheLookupSessionScope(t *testing.T) {
	for _, consistency := range []string{config.RaylineARCConsistencyStrict, config.RaylineARCConsistencyRelaxed} {
		fixture := newPolicySelectorFixtureWith(t, consistency)
		actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
		fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
		advertiseLookupSessions(fixture)
		result, err := fixture.scopedLookupOn(t, lookupScopeTestScope)
		if err != nil {
			t.Fatalf("%s: %v", consistency, err)
		}
		received := fixture.fake.received()
		if len(received) != 1 {
			t.Fatalf("%s: %d decides, want 1", consistency, len(received))
		}
		sent := received[0]
		if sent.SessionScope != raylinearc.PolicySessionScopeLookup || sent.EpisodeMode != "" ||
			sent.EpisodeIDHash != raylinearc.HashEpisodeID(lookupScopeTestScope) || sent.Stage != nil || sent.TurnDelta != nil {
			t.Fatalf("%s: sent scope %q mode %q episode %s stage %v", consistency, sent.SessionScope, sent.EpisodeMode, sent.EpisodeIDHash, sent.Stage)
		}
		if result.RaylineARC.PolicySideCall {
			t.Fatalf("%s: a scoped lookup was decided as a side call", consistency)
		}
	}
}

// An unscoped decide carries no session_scope member at all, so a request
// from a cell with the setting off is byte-identical to one from before it.
func TestUnscopedDecideOmitsSessionScope(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	advertiseLookupSessions(fixture)
	state, _ := raylinearc.NewEpisodeState(2)
	if _, err := fixture.lookupOn(t, state, lookupBody(t), 0, 0, nil); err != nil {
		t.Fatal(err)
	}
	for _, body := range fixture.fake.receivedBodies() {
		if strings.Contains(string(body), `"session_scope"`) {
			t.Fatalf("an unscoped decide sent session_scope: %s", body)
		}
	}
}

// Every way the scope cannot be served falls back without a 429 or 5xx: no
// capability (no decide is sent at all), a concurrent lookup holding the
// scope (one decide, never retried through the busy backoff) and a refused
// scope.
func TestScopedLookupFallsBackWhenTheScopeCannotBeServed(t *testing.T) {
	for _, tc := range []struct {
		name      string
		advertise bool
		failWith  string
		detail    map[string]any
		decides   int
	}{
		{name: "capability unadvertised", decides: 0},
		{name: "session busy", advertise: true, failWith: "session_busy", decides: 1},
		{name: "scope refused", advertise: true, failWith: "invalid_request",
			detail: map[string]any{"field": raylinearc.PolicySessionScopeField}, decides: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newPolicySelectorFixture(t, "")
			if tc.advertise {
				advertiseLookupSessions(fixture)
			}
			if tc.failWith != "" {
				fixture.fake.failDetail = tc.detail
				fixture.fake.failNext(tc.failWith)
			}
			logs := captureLogs(t)
			started := time.Now()
			_, err := fixture.scopedLookupOn(t, lookupScopeTestScope)
			if !isRaylineRoutesLookupSessionFallback(err) {
				t.Fatalf("err = %v, want the lookup-session fallback", err)
			}
			if line := findLogEvent(t, logs, "rayline_routes_lookup_session_fallback"); line["reason"] == nil {
				t.Fatalf("fallback line = %v, want its reason", line)
			}
			if got := len(fixture.fake.received()); got != tc.decides {
				t.Fatalf("%d decides, want %d", got, tc.decides)
			}
			if elapsed := time.Since(started); elapsed > time.Second {
				t.Fatalf("the fallback waited %s", elapsed)
			}
		})
	}
}

// A refusal that is not the scope's -- an invalid request naming another
// field -- is the lookup's own failure, not a fallback.
func TestScopedLookupKeepsOtherRefusals(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	advertiseLookupSessions(fixture)
	fixture.fake.failDetail = map[string]any{"field": "selection"}
	fixture.fake.failNext("invalid_request")
	_, err := fixture.scopedLookupOn(t, lookupScopeTestScope)
	if err == nil || isRaylineRoutesLookupSessionFallback(err) {
		t.Fatalf("err = %v, want the lookup's own failure", err)
	}
}

// A call classified as a side call before the selection, or a zdr turn, is
// never placed on the scope.
func TestLookupScopeSkipsSideCallsAndPrivateTurns(t *testing.T) {
	for name, arcContext := range map[string]*selection.RaylineARCSelectionContext{
		"side call": {PolicyCallKind: raylinearc.PolicyCallSide, EpisodeIDHash: "minted"},
		"zdr turn":  {EphemeralTurn: true, EpisodeIDHash: "minted"},
		"refused":   {PreparationFailure: "missing_request", EpisodeIDHash: "minted"},
	} {
		if applyRaylineRoutesLookupScope(arcContext, lookupScopeTestScope) || arcContext.RoutesLookupSession ||
			arcContext.EpisodeIDHash != "minted" {
			t.Fatalf("%s: placed on the lookup scope", name)
		}
	}
}

// End to end through the endpoint: with lookup_sessions on, a lookup that
// names a conversation decides on the scope of the authenticated user and
// that conversation, and its routing_decision event says so; with it off the
// same lookup decides one-shot, as before, and logs lookup_session false.
func TestRaylineRoutesLookupSessionsEndToEnd(t *testing.T) {
	body := []byte(`{"model":"rayline/arc-dev","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	for _, on := range []bool{true, false} {
		router, selectors := oneCellRouter(t, "main")
		arc := router.Config.Decisions[0].Algorithm.RaylineARC
		arc.RoutesAPI.EpisodeWrites = false
		arc.RoutesAPI.LookupSessions = on
		arc.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"}
		logs := captureLogs(t)
		response := router.handleRaylineRoutesAPI(body, routesContext(map[string]string{
			raylineRoutesSessionHeader: "conv 1",
			"x-authz-user-id":          "user-1",
		}))
		if code := immediateStatusCode(t, response); code != 200 {
			t.Fatalf("on=%t: status %d", on, code)
		}
		router.background.Wait()
		contexts := selectors[config.DefaultRecipeName].contexts
		arcContext := contexts[len(contexts)-1].RaylineARC
		scopeHash := raylinearc.HashEpisodeID("lookup:user-1:conv+1")
		if arcContext.RoutesLookupSession != on || (arcContext.EpisodeIDHash == scopeHash) != on {
			t.Fatalf("on=%t: lookup session %t on %s", on, arcContext.RoutesLookupSession, arcContext.EpisodeIDHash)
		}
		if line := findLogEvent(t, logs, "routing_decision"); line["lookup_session"] != on {
			t.Fatalf("on=%t: routing_decision = %v", on, line)
		}
	}
}

// A lookup with routing.execution, which is otherwise sent relaxed, is sent
// strict on its scope: the execution helpers run before the scope is set.
func TestScopedExecutionLookupIsSentStrict(t *testing.T) {
	fixture := executionFixture(t, false)
	scorer := fixture.selector.armed.Load().scorer.(*policyServiceScorer)
	scorer.setAdvertised(append(append([]string{}, *scorer.advertised.Load()...), raylinearc.PolicyCapabilityLookupSessions))
	controls := executionControls(t, fixture)
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selectOnWith(t, state, lookupBody(t), func(arcContext *selection.RaylineARCSelectionContext) {
		arcContext.RouteControls = controls
		arcContext.RoutesLookup, arcContext.RoutesThrowaway = true, true
		if !applyRaylineRoutesLookupScope(arcContext, lookupScopeTestScope) {
			t.Fatal("the lookup was not placed on its scope")
		}
	})
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := fixture.fake.received()[0]
	if sent.SessionScope != raylinearc.PolicySessionScopeLookup || sent.EpisodeMode != "" || sent.ContextEpoch != "e7" {
		t.Fatalf("sent scope %q mode %q epoch %q", sent.SessionScope, sent.EpisodeMode, sent.ContextEpoch)
	}
}

// A cell with a replica pool never sends the scope: the capability is read
// once for the cell, and a lookup may fail over to an older replica.
func TestScopedLookupFallsBackOnAReplicaPool(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	advertiseLookupSessions(fixture)
	fixture.selector.armedComponents().scorer.(*policyServiceScorer).lookupSessionsServable = false
	if _, err := fixture.scopedLookupOn(t, lookupScopeTestScope); !isRaylineRoutesLookupSessionFallback(err) {
		t.Fatalf("err = %v, want the lookup-session fallback", err)
	}
	if got := len(fixture.fake.received()); got != 0 {
		t.Fatalf("%d decides sent from a replica pool", got)
	}
	pooled := policyTestDecision("")
	pooled.Algorithm.RaylineARC.PolicyService.Replicas = []config.RaylineARCEncoderReplicaConfig{{}}
	if newPolicyServiceScorer(&config.RouterConfig{}, pooled).lookupSessionsServable {
		t.Fatal("a replica-pool cell may send session_scope")
	}
	if !newPolicyServiceScorer(&config.RouterConfig{}, policyTestDecision("")).lookupSessionsServable {
		t.Fatal("a single-service cell may not send session_scope")
	}
}

// A scoped lookup whose caller has already gone starts no one-shot encode.
func TestRouteDecisionSkipsTheFallbackPastTheCallersDeadline(t *testing.T) {
	fixture := newLookupScopeFixture(t)
	fixture.selector.failScoped = true
	_, err := fixture.router.routeDecisionRuntimeState().RouteDecision(context.Background(), routerruntime.RouteDecisionRequest{
		Body: anthropicBody("hello"), DecisionID: "rt_scope_late", Surface: routerruntime.RouteDecisionSurfaceRoutes,
		Ephemeral: true, LookupScope: lookupScopeTestScope, CallerDeadline: time.Now().Add(-time.Millisecond),
	})
	if !isRaylineRoutesLookupSessionFallback(err) || len(fixture.selector.contexts) != 1 {
		t.Fatalf("err = %v after %d selections, want the scoped attempt alone", err, len(fixture.selector.contexts))
	}
}
