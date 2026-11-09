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
	"sync"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// lookupOn selects as a POST /v1/routes lookup whose context is estimated at
// contextTokens, under an encode limit of limit (which the decision service
// sets on throwaway lookups only).
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

func lookupBody(t *testing.T) []byte {
	t.Helper()
	return policyTestRequest(t, map[string]any{"role": "user", "content": "go"})
}

// A lookup at or under the limit is encoded as before; a limit of zero
// bounds nothing.
func TestALookupUnderTheEncodeLimitIsEncoded(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	fixture.selector.armedComponents().scorer.(*policyServiceScorer).fallbackActionID = bindings[1].ActionID
	for _, tc := range []struct{ tokens, limit int }{{1000, 1000}, {1_000_000, 0}} {
		state, _ := raylinearc.NewEpisodeState(2)
		before := len(fixture.fake.received())
		result, err := fixture.lookupOn(t, state, lookupBody(t), tc.tokens, tc.limit, nil)
		if err != nil {
			t.Fatalf("tokens %d limit %d: %v", tc.tokens, tc.limit, err)
		}
		if len(fixture.fake.received()) != before+1 || result.RaylineARC.OfferKind == raylineRoutesEncodeBypassOfferKind {
			t.Fatalf("tokens %d limit %d: the lookup was not encoded", tc.tokens, tc.limit)
		}
	}
}

// Over the limit, a throwaway lookup is answered with the package's fallback
// action without calling the service, and commits no policy state. With no
// fallback bound, it is encoded as before.
func TestALookupOverTheEncodeLimitServesTheFallback(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	scorer := fixture.selector.armedComponents().scorer.(*policyServiceScorer)

	state, _ := raylinearc.NewEpisodeState(2)
	encoded := encodeBypasses(raylineRoutesEncodeBypassEncoded)
	if _, err := fixture.lookupOn(t, state, lookupBody(t), 300_000, 200_000, nil); err != nil {
		t.Fatalf("no fallback: %v", err)
	}
	if len(fixture.fake.received()) != 1 || encodeBypasses(raylineRoutesEncodeBypassEncoded)-encoded != 1 {
		t.Fatal("a lookup with no fallback to serve was not encoded and counted")
	}

	scorer.fallbackActionID = bindings[1].ActionID
	fallback := encodeBypasses(raylineRoutesEncodeBypassFallback)
	logs := captureLogs(t)
	result, err := fixture.lookupOn(t, state, lookupBody(t), 300_000, 200_000, nil)
	if err != nil {
		t.Fatalf("fallback: %v", err)
	}
	if len(fixture.fake.received()) != 1 {
		t.Fatal("a lookup over the encode limit called the service")
	}
	trace := result.RaylineARC
	if trace.PolicyActionID != bindings[1].ActionID || trace.OfferKind != raylineRoutesEncodeBypassOfferKind {
		t.Fatalf("served %s (%s), want the fallback %s", trace.PolicyActionID, trace.OfferKind, bindings[1].ActionID)
	}
	if trace.PolicyNextState != nil {
		t.Fatalf("a bypass committed policy state %+v", trace.PolicyNextState)
	}
	if trace.RawScores != nil || !trace.TokenCountsUnknown || !trace.EncoderLatencyUnknown {
		t.Fatalf("a bypass reported scores or counts: %+v", trace)
	}
	if got := encodeBypasses(raylineRoutesEncodeBypassFallback) - fallback; got != 1 {
		t.Fatalf("fallback bypasses counted %v, want 1", got)
	}
	if line := findLogEvent(t, logs, "rayline_routes_encode_bypass"); line["outcome"] != raylineRoutesEncodeBypassFallback {
		t.Fatalf("bypass line = %v", line)
	}
}

// A lookup with routing controls asks for result points only the policy can
// answer, so it is encoded whatever its size.
func TestALookupWithRoutingControlsIsEncodedOverTheLimit(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	fixture.selector.armedComponents().scorer.(*policyServiceScorer).fallbackActionID = bindings[1].ActionID
	state, _ := raylinearc.NewEpisodeState(2)
	_, _ = fixture.lookupOn(t, state, lookupBody(t), 300_000, 200_000, &selection.RaylineARCRouteControls{})
	if len(fixture.fake.received()) != 1 {
		t.Fatal("a lookup with routing controls was answered without the service")
	}
}

// Only a throwaway POST /v1/routes lookup is bounded: a tracked lookup and
// the legacy consult get no limit. A bypassed answer says it was not scored.
func TestRouteDecisionBoundsOnlyThrowawayRoutesLookups(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers(), offerKind: raylineRoutesEncodeBypassOfferKind}
	router := decisionServiceRouter(t, selector)
	arc := router.Config.Decisions[0].Algorithm.RaylineARC
	arc.RoutesAPI.Enabled = true
	arc.RoutesAPI.MaxEncodeTokens = 4096
	runtime := router.routeDecisionRuntimeState()
	for _, tc := range []struct {
		surface   routerruntime.RouteDecisionSurface
		ephemeral bool
		limit     int
	}{
		{routerruntime.RouteDecisionSurfaceRoutes, true, 4096},
		{routerruntime.RouteDecisionSurfaceRoutes, false, 0},
		{routerruntime.RouteDecisionSurfaceConsult, true, 0},
	} {
		request := routerruntime.RouteDecisionRequest{
			Body: anthropicBody("hello"), DecisionID: "rt_bypass", Surface: tc.surface, Ephemeral: tc.ephemeral,
		}
		if !tc.ephemeral {
			request.SessionID = "conversation-1"
		}
		decision, err := runtime.RouteDecision(context.Background(), request)
		if err != nil {
			t.Fatalf("%q ephemeral=%t: %v", tc.surface, tc.ephemeral, err)
		}
		if limit := selector.contexts[len(selector.contexts)-1].RaylineARC.RoutesEncodeLimit; limit != tc.limit {
			t.Fatalf("%q ephemeral=%t: encode limit %d, want %d", tc.surface, tc.ephemeral, limit, tc.limit)
		}
		if !slices.Contains(decision.Warnings, raylineRoutesEncodeBypassWarning) {
			t.Fatalf("%q: warnings %v lack the bypass", tc.surface, decision.Warnings)
		}
		if decision.ToolsEncoded == nil || *decision.ToolsEncoded {
			t.Fatalf("%q: a bypass reported its tools encoded", tc.surface)
		}
	}
}

// A routed turn and a lookup on the same episode with the same body never
// share a decision: the consult's surface and encode limit are part of the
// coalescing identity, which is fixed before the episode is prepared. Run
// concurrently, each leads its own entry.
func TestARoutedTurnNeverSharesALookupsDecision(t *testing.T) {
	arcConfig := &config.RaylineARCAlgorithmConfig{}
	body := lookupBody(t)
	routed := &RequestContext{Headers: map[string]string{}}
	tracked := &RequestContext{Headers: map[string]string{}, RaylineRouteSurface: string(routerruntime.RouteDecisionSurfaceRoutes)}
	bounded := &RequestContext{
		Headers: map[string]string{}, RaylineRouteSurface: string(routerruntime.RouteDecisionSurfaceRoutes),
		RaylineRoutesEncodeLimit: 4096,
	}
	episode := strings.Repeat("e", 64)
	keys := make([]string, 0, 3)
	for _, reqCtx := range []*RequestContext{routed, tracked, bounded} {
		keys = append(keys, raylineARCInflightKey(nil, episode, body, raylineARCTurnInputs(arcConfig, reqCtx, nil)))
	}
	if keys[0] == keys[1] || keys[0] == keys[2] || keys[1] == keys[2] {
		t.Fatalf("coalescing keys collide: %q", keys)
	}
	registry := &raylineARCInflightRegistry{}
	leads := make([]bool, len(keys))
	var wg sync.WaitGroup
	for index, key := range keys {
		wg.Add(1)
		go func() {
			defer wg.Done()
			entry, leader := registry.join(key)
			leads[index] = leader
			t.Cleanup(func() { registry.finish(entry) })
		}()
	}
	wg.Wait()
	if !leads[0] || !leads[1] || !leads[2] {
		t.Fatalf("a request joined another surface's decision: leads %v", leads)
	}
}
