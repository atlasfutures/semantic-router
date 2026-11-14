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
	"fmt"
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
// conversation's routed episode; there is none when no conversation is named.
func TestRaylineRoutesLookupScopeDerivation(t *testing.T) {
	if got := raylineRoutesLookupScope("user-1", "conv-1"); got != "lookup:user-1:conv-1" {
		t.Fatalf("scope = %q", got)
	}
	if got := raylineRoutesLookupScope("", "conv-1"); got != "lookup::conv-1" {
		t.Fatalf("scope without an authenticated user = %q", got)
	}
	if got := raylineRoutesLookupScope("user-1", ""); got != "" {
		t.Fatalf("no conversation: scope = %q, want none", got)
	}
	scoped := raylinearc.HashEpisodeID(raylineRoutesLookupScope("user-1", "conv-1"))
	if scoped == raylinearc.HashEpisodeID("conv-1") {
		t.Fatal("the lookup scope hashes to the conversation's routed episode")
	}
	if scoped == raylinearc.HashEpisodeID(raylineRoutesLookupScope("user-2", "conv-1")) {
		t.Fatal("two users share a lookup scope")
	}
}

// lookupScopeSelector is a stub selector that, when sideCall is set, reads
// a lookup-scoped selection as a side call, as the policy selection does
// on a positive reading of the materialized request.
type lookupScopeSelector struct {
	*stubARCSelector
	mu       sync.Mutex
	sideCall bool
}

func (selector *lookupScopeSelector) Select(
	ctx context.Context,
	selCtx *selection.SelectionContext,
) (*selection.SelectionResult, error) {
	selector.mu.Lock()
	defer selector.mu.Unlock()
	if selector.sideCall && selCtx.RaylineARC != nil && selCtx.RaylineARC.RoutesLookupSession {
		leaveRaylineRoutesLookupScope(selCtx.RaylineARC)
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
		replicas bool
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
		{name: "replica pool", replicas: true, closes: true,
			request: routerruntime.RouteDecisionRequest{Surface: routes, Ephemeral: true, LookupScope: lookupScopeTestScope}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newLookupScopeFixture(t)
			if tc.artifact {
				fixture.router.Config.Decisions[0].Algorithm.RaylineARC.PolicyService = nil
			}
			if tc.replicas {
				fixture.router.Config.Decisions[0].Algorithm.RaylineARC.PolicyService.Replicas =
					[]config.RaylineARCEncoderReplicaConfig{{}}
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
			if (arcContext.PolicyLookupSessions != nil) != tc.scoped {
				t.Fatalf("lookup session pins set %t, want %t", arcContext.PolicyLookupSessions != nil, tc.scoped)
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

// A lookup the selection reads as a side call leaves its scope inside the
// one selection: it is decided on its minted episode, which it closes, and
// the scope's session is never closed.
func TestRouteDecisionSideCallLeavesTheScope(t *testing.T) {
	fixture := newLookupScopeFixture(t)
	fixture.selector.sideCall = true
	decision := fixture.decide(t, routerruntime.RouteDecisionRequest{
		Surface: routerruntime.RouteDecisionSurfaceRoutes, Ephemeral: true, SessionID: "conv-1", LookupScope: lookupScopeTestScope,
	})
	contexts := fixture.selector.contexts
	if len(contexts) != 1 || contexts[0].RaylineARC.RoutesLookupSession {
		t.Fatalf("selections = %d, want one, off the scope", len(contexts))
	}
	minted := contexts[0].RaylineARC.EpisodeIDHash
	if minted == raylinearc.HashEpisodeID(lookupScopeTestScope) || minted == "" {
		t.Fatalf("the side call decided on %q, want its minted episode", minted)
	}
	if closed := fixture.closedEpisodes(); len(closed) != 1 || closed[0] != minted {
		t.Fatalf("closed %v, want only the minted episode", closed)
	}
	if decision.Encode.LookupSession {
		t.Fatalf("decision = %+v, want it off the scope", decision.Encode)
	}
}

// A scoped lookup that fails is the lookup's failure: it is not decided
// again one-shot, and closes nothing (the scope's session may be a
// concurrent lookup's).
func TestRouteDecisionDoesNotRetryAScopedFailure(t *testing.T) {
	fixture := newLookupScopeFixture(t)
	fixture.selector.err = &raylineARCSelectionFailure{class: arcFailureLookupScopeRefused}
	_, err := fixture.router.routeDecisionRuntimeState().RouteDecision(context.Background(), routerruntime.RouteDecisionRequest{
		Body: anthropicBody("hello"), DecisionID: "rt_scope_err",
		Surface: routerruntime.RouteDecisionSurfaceRoutes, Ephemeral: true, LookupScope: lookupScopeTestScope,
	})
	fixture.router.background.Wait()
	if err == nil || len(fixture.selector.contexts) != 1 {
		t.Fatalf("err = %v after %d selections, want one failed selection", err, len(fixture.selector.contexts))
	}
	if closed := fixture.closedEpisodes(); len(closed) != 0 {
		t.Fatalf("a failed scoped lookup closed %v", closed)
	}
	if errors.Is(err, routerruntime.ErrRouteDecisionContended) || errors.Is(err, routerruntime.ErrRouteDecisionInvalidBody) {
		t.Fatalf("err = %v, want a plain failure (503)", err)
	}
}

// scopedLookupOn selects as a lookup-scoped throwaway lookup.
func (fixture *policySelectorFixture) scopedLookupOn(t *testing.T, scope string) (*selection.SelectionResult, error) {
	t.Helper()
	return fixture.scopedLookupBy(t, scope, time.Time{})
}

// scopedLookupBy selects as a lookup-scoped throwaway lookup whose caller
// waits until deadline (zero: unbounded).
func (fixture *policySelectorFixture) scopedLookupBy(t *testing.T, scope string, deadline time.Time) (*selection.SelectionResult, error) {
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
	arcContext.RoutesLookupDeadline = deadline
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
// from a lookup off its scope is byte-identical to one from before it.
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

// A scope the service cannot serve fails the lookup loudly, with a logged
// class, never as a one-shot lookup: no capability (no decide is sent) and
// a refused scope.
func TestScopedLookupFailsWhenTheScopeCannotBeServed(t *testing.T) {
	for _, tc := range []struct {
		name      string
		advertise bool
		failWith  string
		detail    map[string]any
		class     string
		decides   int
	}{
		{name: "capability unadvertised", class: arcFailureLookupSessionsUnadvertised, decides: 0},
		{name: "scope refused", advertise: true, failWith: "invalid_request",
			detail: map[string]any{"field": raylinearc.PolicySessionScopeField}, class: arcFailureLookupScopeRefused, decides: 1},
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
			_, err := fixture.scopedLookupOn(t, lookupScopeTestScope)
			var failure *raylineARCSelectionFailure
			if !errors.As(err, &failure) || failure.class != tc.class {
				t.Fatalf("err = %v, want %s", err, tc.class)
			}
			if line := findLogEvent(t, logs, "rayline_routes_lookup_session_failed"); line["class"] != tc.class {
				t.Fatalf("failure line = %v", line)
			}
			if got := len(fixture.fake.received()); got != tc.decides {
				t.Fatalf("%d decides, want %d", got, tc.decides)
			}
		})
	}
}

// A concurrent lookup holding the scope (session_busy) is waited out within
// the lookup's deadline, and the lookup then appends on the scope.
func TestScopedLookupWaitsOutABusyScope(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	advertiseLookupSessions(fixture)
	fixture.selector.armedComponents().scorer.(*policyServiceScorer).busyWait = 0
	fixture.fake.failFirstCalls("session_busy", 3)
	if _, err := fixture.scopedLookupBy(t, lookupScopeTestScope, time.Now().Add(5*time.Second)); err != nil {
		t.Fatalf("a busy scope was not waited out: %v", err)
	}
	received := fixture.fake.received()
	if len(received) != 4 {
		t.Fatalf("%d decides, want 3 busy and the append", len(received))
	}
	for _, sent := range received {
		if sent.SessionScope != raylinearc.PolicySessionScopeLookup {
			t.Fatalf("a retry left the scope: %q", sent.SessionScope)
		}
	}
}

// A scope still busy at the lookup's deadline is tried up to the deadline,
// not past it, and fails the lookup there; it is never decided one-shot.
func TestScopedLookupBusyUntilItsDeadline(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	advertiseLookupSessions(fixture)
	fixture.fake.failNext("session_busy")
	deadline := time.Now().Add(300 * time.Millisecond)
	_, err := fixture.scopedLookupBy(t, lookupScopeTestScope, deadline)
	if err == nil {
		t.Fatal("a scope busy past the deadline was decided")
	}
	if late := time.Since(deadline); late < 0 || late > 250*time.Millisecond {
		t.Fatalf("the busy wait ended %s from the deadline", late)
	}
	for _, sent := range fixture.fake.received() {
		if sent.SessionScope != raylinearc.PolicySessionScopeLookup {
			t.Fatal("a busy lookup was decided off its scope")
		}
	}
}

// busyScopeSelector answers every lookup-scoped selection as the policy
// selection does when the scope stays busy: it waits for the lookup's
// deadline, then fails session_busy.
type busyScopeSelector struct {
	stubARCSelector
	deadlines []time.Time
}

func (selector *busyScopeSelector) Select(ctx context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	if !selCtx.RaylineARC.RoutesLookupSession {
		return selector.stubARCSelector.Select(ctx, selCtx)
	}
	selector.deadlines = append(selector.deadlines, selCtx.RaylineARC.RoutesLookupDeadline)
	<-ctx.Done()
	return nil, &raylineARCSelectionFailure{class: "policy_service_session_busy"}
}

// Through the endpoint, a scope busy for the whole lookup answers the same
// 504 deadline_exceeded any lookup that ran out of time answers, never the
// 429 a contended session would.
func TestRaylineRoutesBusyScopeAnswersTheDeadline(t *testing.T) {
	router, _ := oneCellRouter(t, "main")
	arc := router.Config.Decisions[0].Algorithm.RaylineARC
	arc.RoutesAPI.EpisodeWrites = false
	arc.RoutesAPI.DeadlineMS = 200
	arc.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"}
	selector := &busyScopeSelector{stubARCSelector: stubARCSelector{workers: oneCellWorkers()}}
	registry := selection.NewRegistry()
	registry.Register(selection.MethodRaylineARC, selector)
	router.RecipeModelSelectors[config.DefaultRecipeName] = registry
	router.ModelSelector = registry
	started := time.Now()
	response := router.handleRaylineRoutesAPI(
		[]byte(`{"model":"rayline/arc-dev","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`),
		routesContext(map[string]string{raylineRoutesSessionHeader: "conv-1", "x-authz-user-id": "user-1"}),
	)
	router.background.Wait()
	if code := immediateStatusCode(t, response); code != 504 {
		t.Fatalf("status %d, want 504", code)
	}
	if len(selector.deadlines) != 1 || selector.deadlines[0].Sub(started) > 250*time.Millisecond ||
		selector.deadlines[0].Before(started) {
		t.Fatalf("scoped selections %v, want one bounded by the lookup deadline", selector.deadlines)
	}
}

// A refusal that is not the scope's -- an invalid request naming another
// field -- is the lookup's own failure, as on any lookup.
func TestScopedLookupKeepsOtherRefusals(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	advertiseLookupSessions(fixture)
	fixture.fake.failDetail = map[string]any{"field": "selection"}
	fixture.fake.failNext("invalid_request")
	_, err := fixture.scopedLookupOn(t, lookupScopeTestScope)
	var failure *raylineARCSelectionFailure
	if err == nil || errors.As(err, &failure) && failure.class == arcFailureLookupScopeRefused {
		t.Fatalf("err = %v, want the lookup's own failure", err)
	}
}

// A call the selection reads as a side call after it was placed on the scope
// is sent as today's throwaway side call: on its minted episode, without
// session_scope.
func TestScopedLookupReadAsASideCallLeavesTheScope(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	advertiseLookupSessions(fixture)
	state, _ := raylinearc.NewEpisodeState(2)
	arcContext := &selection.RaylineARCSelectionContext{
		RequestID: "req-lookup-side", State: state, RawRequest: lookupBody(t), RequestFormat: policyFormatAnthropic,
		RoutesLookup: true, RoutesThrowaway: true, EpisodeIDHash: "minted",
	}
	if !applyRaylineRoutesLookupScope(arcContext, lookupScopeTestScope) {
		t.Fatal("the lookup was not placed on its scope")
	}
	arcContext.PolicyCallKind, arcContext.PolicyCallKindSource = raylinearc.PolicyCallSide, "header"
	if _, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName: fixture.decision.Name, CandidateModels: fixture.decision.ModelRefs, RaylineARC: arcContext,
	}); err != nil {
		t.Fatal(err)
	}
	sent := fixture.fake.received()[0]
	if sent.SessionScope != "" || sent.EpisodeIDHash != "minted" || arcContext.RoutesLookupSession {
		t.Fatalf("side call sent scope %q on %s", sent.SessionScope, sent.EpisodeIDHash)
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

// End to end through the endpoint: a lookup that names a conversation
// decides on the scope of the authenticated user and that conversation, and
// its routing_decision event says so; one that names none decides one-shot,
// as before, and logs lookup_session false.
func TestRaylineRoutesLookupSessionsEndToEnd(t *testing.T) {
	body := []byte(`{"model":"rayline/arc-dev","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	for _, named := range []bool{true, false} {
		router, selectors := oneCellRouter(t, "main")
		arc := router.Config.Decisions[0].Algorithm.RaylineARC
		arc.RoutesAPI.EpisodeWrites = false
		arc.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"}
		headers := map[string]string{"x-authz-user-id": "user-1"}
		if named {
			headers[raylineRoutesSessionHeader] = "conv 1"
		}
		logs := captureLogs(t)
		response := router.handleRaylineRoutesAPI(body, routesContext(headers))
		if code := immediateStatusCode(t, response); code != 200 {
			t.Fatalf("named=%t: status %d", named, code)
		}
		router.background.Wait()
		contexts := selectors[config.DefaultRecipeName].contexts
		arcContext := contexts[len(contexts)-1].RaylineARC
		scopeHash := raylinearc.HashEpisodeID("lookup:user-1:conv+1")
		if arcContext.RoutesLookupSession != named || (arcContext.EpisodeIDHash == scopeHash) != named {
			t.Fatalf("named=%t: lookup session %t on %s", named, arcContext.RoutesLookupSession, arcContext.EpisodeIDHash)
		}
		if line := findLogEvent(t, logs, "routing_decision"); line["lookup_session"] != named {
			t.Fatalf("named=%t: routing_decision = %v", named, line)
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

// Only a policy-service cell with the routes API on and a single service
// decides on lookup scopes, and only such a cell needs the service to
// advertise lookup_sessions to be ready. A replica pool keeps one-shot
// lookups: its capability list is the cell's, not each replica's.
func TestLookupSessionsServedAndRequired(t *testing.T) {
	single := policyTestDecision("").Algorithm.RaylineARC
	single.RoutesAPI.Enabled = true
	pooled := policyTestDecision("").Algorithm.RaylineARC
	pooled.RoutesAPI.Enabled = true
	pooled.PolicyService.Replicas = []config.RaylineARCEncoderReplicaConfig{{}}
	routesOff := policyTestDecision("").Algorithm.RaylineARC
	for name, tc := range map[string]struct {
		arc    *config.RaylineARCAlgorithmConfig
		served bool
	}{
		"single service": {single, true}, "replica pool": {pooled, false}, "routes off": {routesOff, false},
		"artifact cell": {&config.RaylineARCAlgorithmConfig{RoutesAPI: config.RaylineARCRoutesAPIConfig{Enabled: true}}, false},
	} {
		if got := raylineRoutesLookupSessionsServed(tc.arc); got != tc.served {
			t.Fatalf("%s: served %t, want %t", name, got, tc.served)
		}
		scorer := &policyServiceScorer{}
		scorer.setAdvertised([]string{raylinearc.PolicyCapabilityDecisionsV2})
		err := requireRaylineRoutesLookupSessions(tc.arc, scorer)
		var failure *raylinearc.PolicyServiceError
		if tc.served != (errors.As(err, &failure) && failure.Class == raylinearc.PolicyCapabilityUnsupportedClass &&
			failure.Detail["capability"] == raylinearc.PolicyCapabilityLookupSessions) {
			t.Fatalf("%s: readiness without the capability = %v", name, err)
		}
		scorer.setAdvertised([]string{raylinearc.PolicyCapabilityLookupSessions})
		if err := requireRaylineRoutesLookupSessions(tc.arc, scorer); err != nil {
			t.Fatalf("%s: readiness with the capability = %v", name, err)
		}
	}
}

// scopedLookupWithPins selects a lookup on scope with its sessions pinned in
// pins, on a sessioned fixture whose containers advertise lookup sessions.
func scopedLookupWithPins(
	t *testing.T,
	fixture *policySelectorFixture,
	pins raylinearc.PolicyLookupSessionPins,
	scope string,
) (*selection.SelectionResult, error) {
	t.Helper()
	return scopedLookupWithPinsIn(context.Background(), t, fixture, pins, scope)
}

func lookupPinKey(fixture *policySelectorFixture, scope string) string {
	return raylineRoutesLookupSessionPinKey(fixture.decision.Algorithm.RaylineARC.PolicyService, raylinearc.HashEpisodeID(scope))
}

func scopedLookupWithPinsIn(
	ctx context.Context,
	t *testing.T,
	fixture *policySelectorFixture,
	pins raylinearc.PolicyLookupSessionPins,
	scope string,
) (*selection.SelectionResult, error) {
	t.Helper()
	state, _ := raylinearc.NewEpisodeState(2)
	arcContext := &selection.RaylineARCSelectionContext{
		RequestID: "req-lookup-pin", State: state, RawRequest: lookupBody(t), RequestFormat: policyFormatAnthropic,
		RoutesLookup: true, RoutesThrowaway: true,
	}
	if !applyRaylineRoutesLookupScope(arcContext, scope) {
		t.Fatal("the lookup was not placed on its scope")
	}
	arcContext.PolicyLookupSessions = pins
	arcContext.PolicyLookupSessionKey = lookupPinKey(fixture, scope)
	return fixture.selector.Select(ctx, &selection.SelectionContext{
		DecisionName: fixture.decision.Name, CandidateModels: fixture.decision.ModelRefs, RaylineARC: arcContext,
	})
}

func newLookupSessionsFixture(t *testing.T) (*policySelectorFixture, *fakeSessionedServer) {
	t.Helper()
	fixture, front := newPolicySessionFixture(t)
	advertiseLookupSessions(fixture)
	return fixture, front
}

// On a sessioned policy Server, a lookup scope keeps one Modal session,
// pinned per scope, so every lookup on the conversation reaches the
// container that holds its warm encoder session; the session idles out with
// the service's lookup TTL, and another scope gets its own.
func TestLookupScopeKeepsItsModalSession(t *testing.T) {
	fixture, front := newLookupSessionsFixture(t)
	pins := raylinearc.NewMemoryPolicyLookupSessionPins()
	for lookup := 0; lookup < 3; lookup++ {
		result, err := scopedLookupWithPins(t, fixture, pins, lookupScopeTestScope)
		if err != nil {
			t.Fatalf("lookup %d: %v", lookup, err)
		}
		awaitPin(t, pins, lookupPinKey(fixture, lookupScopeTestScope))
		if result.RaylineARC.PolicySessionMoved {
			t.Fatalf("lookup %d moved sessions", lookup)
		}
	}
	pinned, _ := pins.LoadLookupSession(context.Background(), lookupPinKey(fixture, lookupScopeTestScope))
	if pinned == nil || front.containerOf(pinned.Token) < 0 {
		t.Fatalf("pin = %v, want the scope's live session", pinned)
	}
	front.mu.Lock()
	starts, terminates, idle := front.starts, front.terminates, append([]string(nil), front.idleTimeouts...)
	front.mu.Unlock()
	if starts != 1 || terminates != 0 {
		t.Fatalf("starts %d terminates %d, want one session kept", starts, terminates)
	}
	if want := fmt.Sprint(int(raylinearc.PolicyLookupSessionIdleTimeout / time.Second)); idle[0] != want {
		t.Fatalf("session idle timeout %q, want the lookup TTL %s", idle[0], want)
	}
	container := front.containerOf(pinned.Token)
	decides := len(front.containers[container].received())
	if decides != 3 {
		t.Fatalf("the scope's container decided %d lookups, want 3", decides)
	}
	if _, err := scopedLookupWithPins(t, fixture, pins, "lookup:user-2:conv-9"); err != nil {
		t.Fatal(err)
	}
	awaitPin(t, pins, lookupPinKey(fixture, "lookup:user-2:conv-9"))
	if other, _ := pins.LoadLookupSession(context.Background(), lookupPinKey(fixture, "lookup:user-2:conv-9")); other == nil || *other == *pinned {
		t.Fatalf("another scope shares the session: %v", other)
	}
}

// A pinned session whose container is gone moves the lookup once, to a new
// session it pins; the session it left is ended after the pin.
func TestLookupScopeMovesOffALostSessionAndRepins(t *testing.T) {
	fixture, front := newLookupSessionsFixture(t)
	pins := raylinearc.NewMemoryPolicyLookupSessionPins()
	if _, err := scopedLookupWithPins(t, fixture, pins, lookupScopeTestScope); err != nil {
		t.Fatal(err)
	}
	awaitPin(t, pins, lookupPinKey(fixture, lookupScopeTestScope))
	first, _ := pins.LoadLookupSession(context.Background(), lookupPinKey(fixture, lookupScopeTestScope))
	front.mu.Lock()
	front.dead[front.sessions[first.Token]] = true
	front.mu.Unlock()
	result, err := scopedLookupWithPins(t, fixture, pins, lookupScopeTestScope)
	if err != nil {
		t.Fatalf("moved lookup: %v", err)
	}
	var second *raylinearc.PolicySession
	for deadline := time.Now().Add(2 * time.Second); time.Now().Before(deadline); time.Sleep(5 * time.Millisecond) {
		if second, _ = pins.LoadLookupSession(context.Background(), lookupPinKey(fixture, lookupScopeTestScope)); second != nil && *second != *first {
			break
		}
	}
	if !result.RaylineARC.PolicySessionMoved || second == nil || *second == *first {
		t.Fatalf("pin %v after a move from %v, want a new session", second, first)
	}
	deadline := time.Now().Add(2 * time.Second)
	for front.containerOf(first.Token) >= 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if front.containerOf(first.Token) >= 0 {
		t.Fatal("the session the lookup moved off was not ended")
	}
}

// A lookup that fails on its scope (the service refused it) pins nothing
// and ends the session it started.
func TestLookupScopeFailureLeavesThePinAlone(t *testing.T) {
	fixture, front := newLookupSessionsFixture(t)
	pins := raylinearc.NewMemoryPolicyLookupSessionPins()
	for _, container := range front.containers {
		container.failDetail = map[string]any{"field": raylinearc.PolicySessionScopeField}
		container.failNext("invalid_request")
	}
	if _, err := scopedLookupWithPins(t, fixture, pins, lookupScopeTestScope); err == nil {
		t.Fatal("a refused scope was decided")
	}
	if pinned, _ := pins.LoadLookupSession(context.Background(), lookupPinKey(fixture, lookupScopeTestScope)); pinned != nil {
		t.Fatal("a failed lookup pinned its session")
	}
	awaitTerminates(t, front, 1)
}

// The pin lives for its idle timeout from the last lookup, and no longer.
func TestMemoryLookupSessionPinsExpire(t *testing.T) {
	pins := raylinearc.NewMemoryPolicyLookupSessionPins()
	session := raylinearc.PolicySession{ID: "sd-1", Token: "tok.1.secret"}
	if pinned, err := pins.StoreLookupSession(context.Background(), "scope", nil, session, 10*time.Millisecond); err != nil || !pinned {
		t.Fatal(pinned, err)
	}
	if got, _ := pins.LoadLookupSession(context.Background(), "scope"); got == nil || *got != session {
		t.Fatalf("pin = %v", got)
	}
	time.Sleep(20 * time.Millisecond)
	if got, _ := pins.LoadLookupSession(context.Background(), "scope"); got != nil {
		t.Fatal("the pin outlived its idle timeout")
	}
}

// A pin changes only from the session the writer read: a first pin when
// there was none, a refresh of the same session, or a move off the one it
// read. A lookup that read a session another lookup has since replaced, or
// that raced another first pin, does not overwrite it.
func TestMemoryLookupSessionPinsCompareAndSwap(t *testing.T) {
	ctx := context.Background()
	pins := raylinearc.NewMemoryPolicyLookupSessionPins()
	a := raylinearc.PolicySession{ID: "sd-a", Token: "tok.a.secret"}
	b := raylinearc.PolicySession{ID: "sd-b", Token: "tok.b.secret"}
	c := raylinearc.PolicySession{ID: "sd-c", Token: "tok.c.secret"}
	store := func(previous *raylinearc.PolicySession, session raylinearc.PolicySession) bool {
		t.Helper()
		pinned, err := pins.StoreLookupSession(ctx, "key", previous, session, time.Minute)
		if err != nil {
			t.Fatal(err)
		}
		return pinned
	}
	if !store(nil, a) {
		t.Fatal("the first pin was refused")
	}
	if store(nil, c) {
		t.Fatal("a racing first pin replaced the scope's pin")
	}
	if !store(&a, a) {
		t.Fatal("a refresh was refused")
	}
	if !store(&a, b) {
		t.Fatal("a move off the pinned session was refused")
	}
	if store(&a, a) {
		t.Fatal("a lookup finishing on a retired session restored it")
	}
	if store(&a, c) {
		t.Fatal("a stale move replaced the newer pin")
	}
	if got, _ := pins.LoadLookupSession(ctx, "key"); got == nil || *got != b {
		t.Fatalf("pin = %v, want the moved-to session", got)
	}
}

// Pins are per policy backend: recipes on different Modal apps (or another
// package) never read each other's session for one conversation.
func TestLookupSessionPinKeyIsPerBackend(t *testing.T) {
	scope := raylinearc.HashEpisodeID(lookupScopeTestScope)
	app := &config.RaylineARCPolicyServiceConfig{BaseURL: "https://a.example", PackageSHA256: "p1"}
	otherApp := &config.RaylineARCPolicyServiceConfig{BaseURL: "https://b.example", PackageSHA256: "p1"}
	otherPackage := &config.RaylineARCPolicyServiceConfig{BaseURL: "https://a.example", PackageSHA256: "p2"}
	key := raylineRoutesLookupSessionPinKey(app, scope)
	if key == raylineRoutesLookupSessionPinKey(otherApp, scope) || key == raylineRoutesLookupSessionPinKey(otherPackage, scope) ||
		key == scope || key != raylineRoutesLookupSessionPinKey(app, scope) {
		t.Fatal("the pin key does not name the scope on one backend")
	}
}

// lookupPinsStub answers every store with pinned/err (landing it when
// landed is set, as a write whose reply was lost), and every load with what
// landed; block holds a store until release closes.
type lookupPinsStub struct {
	mu      sync.Mutex
	pinned  bool
	err     error
	landed  bool
	current *raylinearc.PolicySession
	block   bool
	release chan struct{}
}

func (stub *lookupPinsStub) LoadLookupSession(context.Context, string) (*raylinearc.PolicySession, error) {
	stub.mu.Lock()
	defer stub.mu.Unlock()
	return stub.current, nil
}

func (stub *lookupPinsStub) StoreLookupSession(
	_ context.Context, _ string, _ *raylinearc.PolicySession, session raylinearc.PolicySession, _ time.Duration,
) (bool, error) {
	if stub.block {
		<-stub.release
	}
	stub.mu.Lock()
	defer stub.mu.Unlock()
	if stub.landed {
		stub.current = &session
	}
	return stub.pinned, stub.err
}

func awaitTerminates(t *testing.T, front *fakeSessionedServer, want int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		front.mu.Lock()
		terminates := front.terminates
		front.mu.Unlock()
		if terminates == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("terminates = %d, want %d", terminates, want)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// A session a decided lookup started but could not pin -- another lookup
// pinned first, or the write failed -- is ended rather than left holding a
// container's slot until its idle timeout. The lookup's answer stands.
func TestLookupSessionUnpinnedIsEnded(t *testing.T) {
	for name, stub := range map[string]*lookupPinsStub{
		"superseded":   {pinned: false},
		"write failed": {err: errors.New("redis down")},
	} {
		t.Run(name, func(t *testing.T) {
			fixture, front := newLookupSessionsFixture(t)
			if _, err := scopedLookupWithPins(t, fixture, stub, lookupScopeTestScope); err != nil {
				t.Fatalf("the lookup failed on its pin: %v", err)
			}
			awaitTerminates(t, front, 1)
		})
	}
}

// A write that failed but landed (its reply was lost) is read back, and the
// session it pinned is kept.
func TestLookupSessionLandedPinIsKept(t *testing.T) {
	fixture, front := newLookupSessionsFixture(t)
	stub := &lookupPinsStub{err: errors.New("reply lost"), landed: true}
	if _, err := scopedLookupWithPins(t, fixture, stub, lookupScopeTestScope); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	front.mu.Lock()
	terminates := front.terminates
	front.mu.Unlock()
	stub.mu.Lock()
	current := stub.current
	stub.mu.Unlock()
	if terminates != 0 || current == nil || front.containerOf(current.Token) < 0 {
		t.Fatalf("terminates %d, pin %v: the landed session was ended", terminates, current)
	}
}

// The pin is written off the answer's path: a stalled store (the episode
// store's Redis client ignores context deadlines) never holds the answer.
func TestLookupSessionPinWriteKeepsTheLookupDeadline(t *testing.T) {
	fixture, _ := newLookupSessionsFixture(t)
	stub := &lookupPinsStub{block: true, pinned: true, release: make(chan struct{})}
	defer close(stub.release)
	started := time.Now()
	if _, err := scopedLookupWithPins(t, fixture, stub, lookupScopeTestScope); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(started); elapsed >= 500*time.Millisecond {
		t.Fatalf("a stalled pin write held the lookup %s", elapsed)
	}
}

// awaitPin waits for the background pin write of a decided lookup.
func awaitPin(t *testing.T, pins raylinearc.PolicyLookupSessionPins, key string) {
	t.Helper()
	for deadline := time.Now().Add(2 * time.Second); time.Now().Before(deadline); time.Sleep(5 * time.Millisecond) {
		if pinned, _ := pins.LoadLookupSession(context.Background(), key); pinned != nil {
			return
		}
	}
	t.Fatal("the lookup's session was never pinned")
}

// busyDecider answers every decide session_busy and records when each began.
type busyDecider struct {
	mu    sync.Mutex
	calls []time.Time
}

func (decider *busyDecider) Decide(context.Context, raylinearc.PolicyDecisionRequest) (*raylinearc.PolicyDecisionResponse, error) {
	decider.mu.Lock()
	defer decider.mu.Unlock()
	decider.calls = append(decider.calls, time.Now())
	return nil, &raylinearc.PolicyServiceError{Class: "session_busy", Status: 409}
}

// On a context that outlives the caller (a detached lookup), a scope busy
// until the deadline starts no decide at or past it, and its failure comes
// back once the deadline has passed.
func TestLookupBusyWaitStartsNoDecidePastTheDeadline(t *testing.T) {
	decider := &busyDecider{}
	deadline := time.Now().Add(300 * time.Millisecond)
	admit := func() (func(), error) { return func() {}, nil }
	_, err := decidePolicyThroughBusyUntil(context.Background(), decider, raylinearc.PolicyDecisionRequest{}, deadline, true, admit, nil)
	returned := time.Now()
	var failure *raylinearc.PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != "session_busy" {
		t.Fatalf("err = %v, want session_busy", err)
	}
	if returned.Before(deadline) || returned.Sub(deadline) > 100*time.Millisecond {
		t.Fatalf("returned %s from the deadline, want just after it", returned.Sub(deadline))
	}
	decider.mu.Lock()
	defer decider.mu.Unlock()
	if len(decider.calls) < 3 {
		t.Fatalf("%d decides, want the scope retried", len(decider.calls))
	}
	if last := decider.calls[len(decider.calls)-1]; !last.Before(deadline) {
		t.Fatalf("a decide started %s past the deadline", last.Sub(deadline))
	}
	if last := decider.calls[len(decider.calls)-1]; deadline.Sub(last) > policyBusyRetryMax {
		t.Fatalf("the last decide was %s before the deadline, want the wait used", deadline.Sub(last))
	}
}
