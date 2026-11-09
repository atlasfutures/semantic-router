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
	"slices"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// lookupOn selects as a POST /v1/routes lookup whose context is estimated at
// contextTokens, under an encode limit of limit.
func (fixture *policySelectorFixture) lookupOn(
	t *testing.T,
	state *raylinearc.EpisodeState,
	body []byte,
	contextTokens, limit int,
	controls *selection.RaylineARCRouteControls,
) (*selection.SelectionResult, error) {
	t.Helper()
	return fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash:     strings.Repeat("e", 64),
			RequestID:         "req-policy-test",
			State:             state,
			RawRequest:        body,
			RequestFormat:     policyFormatAnthropic,
			RoutesLookup:      true,
			RoutesEncodeLimit: limit,
			ContextTokens:     contextTokens,
			RouteControls:     controls,
		},
	})
}

func encodeBypasses(outcome string) float64 {
	return testutil.ToFloat64(metrics.RaylineARCRoutesEncodeBypasses.WithLabelValues(outcome))
}

// A lookup at or under the limit is encoded as before; a limit of zero
// bounds nothing.
func TestALookupUnderTheEncodeLimitIsEncoded(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	for _, tc := range []struct{ tokens, limit int }{{1000, 1000}, {1_000_000, 0}} {
		state, _ := raylinearc.NewEpisodeState(2)
		before := len(fixture.fake.received())
		result, err := fixture.lookupOn(t, state, nextTurn(t), tc.tokens, tc.limit, nil)
		if err != nil {
			t.Fatalf("tokens %d limit %d: %v", tc.tokens, tc.limit, err)
		}
		if len(fixture.fake.received()) != before+1 || result.RaylineARC.OfferKind == raylineRoutesEncodeBypassOfferKind {
			t.Fatalf("tokens %d limit %d: the lookup was not encoded", tc.tokens, tc.limit)
		}
	}
}

// Over the limit, a lookup on an episode that holds an action is answered
// with it, without calling the service, and commits it as a held turn.
func TestALookupOverTheEncodeLimitServesTheHeldAction(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	state, _ := raylinearc.NewEpisodeState(2)
	servedTurn(t, fixture, state, bindings[1].ActionID)
	calls := len(fixture.fake.received())
	held := encodeBypasses(raylineRoutesEncodeBypassHeld)
	logs := captureLogs(t)
	result, err := fixture.lookupOn(t, state, nextTurn(t), 300_000, 200_000, nil)
	if err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if len(fixture.fake.received()) != calls {
		t.Fatal("a lookup over the encode limit called the service")
	}
	trace := result.RaylineARC
	if trace.PolicyActionID != bindings[1].ActionID || trace.OfferKind != raylineRoutesEncodeBypassOfferKind {
		t.Fatalf("served %s (%s), want the held %s as %s", trace.PolicyActionID, trace.OfferKind,
			bindings[1].ActionID, raylineRoutesEncodeBypassOfferKind)
	}
	if trace.RawScores != nil || !trace.TokenCountsUnknown || !trace.EncoderLatencyUnknown {
		t.Fatalf("a bypass reported scores or counts: %+v", trace)
	}
	if next := trace.PolicyNextState; next == nil || next.Ledger[len(next.Ledger)-1].ActionID != bindings[1].ActionID {
		t.Fatalf("the held turn did not commit to the ledger: %+v", next)
	}
	if got := encodeBypasses(raylineRoutesEncodeBypassHeld) - held; got != 1 {
		t.Fatalf("held bypasses counted %v, want 1", got)
	}
	if line := findLogEvent(t, logs, "rayline_routes_encode_bypass"); line["outcome"] != raylineRoutesEncodeBypassHeld {
		t.Fatalf("bypass line = %v", line)
	}
}

// Over the limit with nothing held, the package's fallback action serves,
// and the episode's policy state is left as it was; with no fallback either,
// the lookup is encoded as before.
func TestALookupOverTheEncodeLimitWithNothingHeldServesTheFallback(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	scorer := fixture.selector.armedComponents().scorer.(*policyServiceScorer)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "go"})

	state, _ := raylinearc.NewEpisodeState(2)
	encoded := encodeBypasses(raylineRoutesEncodeBypassEncoded)
	if _, err := fixture.lookupOn(t, state, body, 300_000, 200_000, nil); err != nil {
		t.Fatalf("no fallback: %v", err)
	}
	if len(fixture.fake.received()) != 1 || encodeBypasses(raylineRoutesEncodeBypassEncoded)-encoded != 1 {
		t.Fatal("a lookup with nothing to serve was not encoded and counted")
	}

	scorer.fallbackActionID = bindings[1].ActionID
	fallback := encodeBypasses(raylineRoutesEncodeBypassFallback)
	result, err := fixture.lookupOn(t, state, body, 300_000, 200_000, nil)
	if err != nil {
		t.Fatalf("fallback: %v", err)
	}
	if len(fixture.fake.received()) != 1 {
		t.Fatal("a fallback answer called the service")
	}
	trace := result.RaylineARC
	if trace.PolicyActionID != bindings[1].ActionID || trace.OfferKind != raylineRoutesEncodeBypassOfferKind {
		t.Fatalf("served %s (%s), want the fallback %s", trace.PolicyActionID, trace.OfferKind, bindings[1].ActionID)
	}
	if trace.PolicyNextState != nil {
		t.Fatalf("a fallback answer committed policy state %+v", trace.PolicyNextState)
	}
	if got := encodeBypasses(raylineRoutesEncodeBypassFallback) - fallback; got != 1 {
		t.Fatalf("fallback bypasses counted %v, want 1", got)
	}
}

// A lookup with routing controls asks for result points only the policy can
// answer, so it is encoded whatever its size.
func TestALookupWithRoutingControlsIsEncodedOverTheLimit(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	state, _ := raylinearc.NewEpisodeState(2)
	servedTurn(t, fixture, state, bindings[1].ActionID)
	calls := len(fixture.fake.received())
	_, _ = fixture.lookupOn(t, state, nextTurn(t), 300_000, 200_000, &selection.RaylineARCRouteControls{})
	if len(fixture.fake.received()) != calls+1 {
		t.Fatal("a lookup with routing controls was answered without the service")
	}
}

// A tracked conversation decides on after a bypass as though the bypassed
// turn had been held: the next lookup under the limit extends the committed
// prefix (the service appends to its session) and attributes the bypassed
// reply to the held action.
func TestATrackedLookupExtendsItsPrefixAfterABypass(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	state, _ := raylinearc.NewEpisodeState(2)
	servedTurn(t, fixture, state, bindings[1].ActionID)

	second := nextTurn(t)
	bypassed, err := fixture.lookupOn(t, state, second, 300_000, 200_000, nil)
	if err != nil || bypassed.RaylineARC.OfferKind != raylineRoutesEncodeBypassOfferKind {
		t.Fatalf("bypass: %v", err)
	}
	arm := bypassed.RaylineARC.SelectedArm
	state.PreviousArm = &arm
	state.Policy = bypassed.RaylineARC.PolicyNextState
	state.TurnIndex++

	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	third := policyTestRequest(t, map[string]any{"role": "user", "content": "go"},
		map[string]any{"role": "assistant", "content": "ok"}, map[string]any{"role": "user", "content": "more"},
		map[string]any{"role": "assistant", "content": "done"}, map[string]any{"role": "user", "content": "again"})
	logs := captureLogs(t)
	result, err := fixture.lookupOn(t, state, third, 1000, 200_000, nil)
	if err != nil {
		t.Fatalf("lookup after the bypass: %v", err)
	}
	transition := ""
	for _, entry := range logs.All() {
		if fields := entry.ContextMap(); fields["event"] == "rayline_arc_policy_turn" {
			transition, _ = fields["transition"].(string)
		}
	}
	if transition != raylinearc.PolicyTransitionExtends {
		t.Fatalf("transition after a bypass = %q, want %q", transition, raylinearc.PolicyTransitionExtends)
	}
	received := fixture.fake.received()
	decide := received[len(received)-1]
	if !slices.ContainsFunc(decide.Attribution, func(entry raylinearc.PolicyAttribution) bool {
		return entry.Message == 3 && entry.ActionID == bindings[1].ActionID
	}) {
		t.Fatalf("attribution %+v does not credit the bypassed reply (message 3) to the held action", decide.Attribution)
	}
	if next := result.RaylineARC.PolicyNextState; next == nil || next.PrefixLen != 5 {
		t.Fatalf("committed prefix after the lookup = %+v, want all 5 messages", next)
	}
}

// Only a POST /v1/routes lookup is bounded, and its answer says it was not
// scored.
func TestRouteDecisionBoundsOnlyRoutesLookupsAndWarnsOnABypass(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers(), offerKind: raylineRoutesEncodeBypassOfferKind}
	router := decisionServiceRouter(t, selector)
	arc := router.Config.Decisions[0].Algorithm.RaylineARC
	arc.RoutesAPI.Enabled = true
	arc.RoutesAPI.MaxEncodeTokens = 4096
	runtime := router.routeDecisionRuntimeState()
	for _, surface := range []routerruntime.RouteDecisionSurface{
		routerruntime.RouteDecisionSurfaceRoutes, routerruntime.RouteDecisionSurfaceConsult,
	} {
		decision, err := runtime.RouteDecision(context.Background(), routerruntime.RouteDecisionRequest{
			Body: anthropicBody("hello"), DecisionID: "rt_bypass", Surface: surface, Ephemeral: true,
		})
		if err != nil {
			t.Fatalf("%q: %v", surface, err)
		}
		limit := selector.contexts[len(selector.contexts)-1].RaylineARC.RoutesEncodeLimit
		if surface == routerruntime.RouteDecisionSurfaceRoutes && limit != 4096 ||
			surface != routerruntime.RouteDecisionSurfaceRoutes && limit != 0 {
			t.Fatalf("%q: encode limit %d", surface, limit)
		}
		if !slices.Contains(decision.Warnings, raylineRoutesEncodeBypassWarning) {
			t.Fatalf("%q: warnings %v lack the bypass", surface, decision.Warnings)
		}
		if decision.ToolsEncoded == nil || *decision.ToolsEncoded {
			t.Fatalf("%q: a bypass reported its tools encoded", surface)
		}
	}
}
