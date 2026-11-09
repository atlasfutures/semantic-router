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
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	result, err := fixture.lookupOn(t, state, nextTurn(t), 1_000_000, 200_000, nil)
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
	if _, err := fixture.lookupOn(t, state, body, 1_000_000, 200_000, nil); err != nil {
		t.Fatalf("no fallback: %v", err)
	}
	if len(fixture.fake.received()) != 1 || encodeBypasses(raylineRoutesEncodeBypassEncoded)-encoded != 1 {
		t.Fatal("a lookup with nothing to serve was not encoded and counted")
	}

	scorer.fallbackActionID = bindings[1].ActionID
	fallback := encodeBypasses(raylineRoutesEncodeBypassFallback)
	result, err := fixture.lookupOn(t, state, body, 1_000_000, 200_000, nil)
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
	_, _ = fixture.lookupOn(t, state, nextTurn(t), 1_000_000, 200_000, &selection.RaylineARCRouteControls{})
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
	bypassed, err := fixture.lookupOn(t, state, second, 1_000_000, 200_000, nil)
	if err != nil || bypassed.RaylineARC.OfferKind != raylineRoutesEncodeBypassOfferKind {
		t.Fatalf("bypass: %v", err)
	}
	if next := bypassed.RaylineARC.PolicyNextState; next.PrefixLen != 3 || next.EncodedThrough() != 1 {
		t.Fatalf("after the bypass: prefix %d, encoded through %d; want 3 and 1", next.PrefixLen, next.EncodedThrough())
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
	if next := result.RaylineARC.PolicyNextState; next == nil || next.PrefixLen != 5 || next.EncodedThrough() != 5 {
		t.Fatalf("committed prefix after the lookup = %+v, want all 5 messages, encoded", next)
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

func fiveMessageTurn(t *testing.T) []byte {
	t.Helper()
	return policyTestRequest(t, map[string]any{"role": "user", "content": "go"},
		map[string]any{"role": "assistant", "content": "ok"}, map[string]any{"role": "user", "content": "more"},
		map[string]any{"role": "assistant", "content": "done"}, map[string]any{"role": "user", "content": "again"})
}

// A tracked lookup that extends its context is bounded on its suffix, the
// messages since the service's last encode: a whole context over the limit
// whose suffix is under it is encoded.
func TestATrackedLookupIsBoundedOnItsSuffix(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	state, _ := raylinearc.NewEpisodeState(2)
	servedTurn(t, fixture, state, bindings[1].ActionID)
	calls := len(fixture.fake.received())
	// Two of three similar messages are new: about 2/3 of 260k, under 200k.
	result, err := fixture.lookupOn(t, state, nextTurn(t), 260_000, 200_000, nil)
	if err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if len(fixture.fake.received()) != calls+1 || result.RaylineARC.OfferKind == raylineRoutesEncodeBypassOfferKind {
		t.Fatal("a lookup whose suffix is under the limit was not encoded")
	}

	// The same lookup with nothing committed is encoded whole: over.
	fresh, _ := raylinearc.NewEpisodeState(2)
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[1].ActionID })
	scorer := fixture.selector.armedComponents().scorer.(*policyServiceScorer)
	scorer.fallbackActionID = bindings[0].ActionID
	logs := captureLogs(t)
	whole, err := fixture.lookupOn(t, fresh, nextTurn(t), 260_000, 200_000, nil)
	if err != nil || whole.RaylineARC.OfferKind != raylineRoutesEncodeBypassOfferKind {
		t.Fatalf("a session's first lookup over the limit was encoded: %v", err)
	}
	if line := findLogEvent(t, logs, "rayline_routes_encode_bypass"); line["span"] != raylineRoutesEncodeSpanContext {
		t.Fatalf("bypass line = %v", line)
	}
}

// After a bypass, the next lookup's suffix starts where the service's encode
// ended, so it spans the bypassed messages as well as its own: here those
// are over the limit although the messages since the bypass alone are not.
func TestASuffixAfterABypassSpansTheBypassedMessages(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	state, _ := raylinearc.NewEpisodeState(2)
	servedTurn(t, fixture, state, bindings[1].ActionID)
	bypassed, err := fixture.lookupOn(t, state, nextTurn(t), 1_000_000, 200_000, nil)
	if err != nil || bypassed.RaylineARC.OfferKind != raylineRoutesEncodeBypassOfferKind {
		t.Fatalf("bypass: %v", err)
	}
	arm := bypassed.RaylineARC.SelectedArm
	state.PreviousArm = &arm
	state.Policy = bypassed.RaylineARC.PolicyNextState
	state.TurnIndex++

	calls := len(fixture.fake.received())
	logs := captureLogs(t)
	// Four of five messages since the service's encode: about 320k. From
	// the committed prefix it would be two, about 160k.
	result, err := fixture.lookupOn(t, state, fiveMessageTurn(t), 400_000, 200_000, nil)
	if err != nil || result.RaylineARC.OfferKind != raylineRoutesEncodeBypassOfferKind {
		t.Fatalf("a suffix spanning the bypassed messages was encoded: %v", err)
	}
	if len(fixture.fake.received()) != calls {
		t.Fatal("the bypass called the service")
	}
	line := findLogEvent(t, logs, "rayline_routes_encode_bypass")
	if line["span"] != raylineRoutesEncodeSpanSuffix {
		t.Fatalf("bypass line = %v", line)
	}
	if tokens, _ := line["encode_tokens"].(int64); tokens <= 200_000 || tokens >= 400_000 {
		t.Fatalf("suffix estimate %v, want the four messages since the encode", line["encode_tokens"])
	}
	if next := result.RaylineARC.PolicyNextState; next.PrefixLen != 5 || next.EncodedThrough() != 1 {
		t.Fatalf("second bypass: prefix %d, encoded through %d; want 5 and 1", next.PrefixLen, next.EncodedThrough())
	}
}

// A decided turn's state is encoded through its prefix, whatever a hold
// before it left; a state decoded without the field is too.
func TestEncodedThroughFollowsTheDecidedTurns(t *testing.T) {
	messages := []json.RawMessage{json.RawMessage(`{"role":"user","content":"go"}`)}
	held := (&raylinearc.PolicyEpisodeState{}).NextUnencoded(messages, "a", "x")
	if held.PrefixLen != 1 || held.EncodedThrough() != 0 {
		t.Fatalf("held: %+v", held)
	}
	encoded, err := json.Marshal(held)
	if err != nil {
		t.Fatal(err)
	}
	var decoded raylinearc.PolicyEpisodeState
	if err := json.Unmarshal(encoded, &decoded); err != nil || decoded.EncodedThrough() != 0 {
		t.Fatalf("round trip lost the encode: %s", encoded)
	}
	two := append(messages, json.RawMessage(`{"role":"assistant","content":"ok"}`), json.RawMessage(`{"role":"user","content":"more"}`))
	if decided := decoded.Next(two, "a", "x"); decided.EncodedPrefixLen != nil || decided.EncodedThrough() != 3 {
		t.Fatalf("decided: %+v", decided)
	}
	if legacy := (&raylinearc.PolicyEpisodeState{PrefixLen: 4}); legacy.EncodedThrough() != 4 {
		t.Fatal("a state without the field reads as unencoded")
	}
}

// lookupTurn is a tracked lookup on the delta cell, over its encode limit or
// not, committed as the decision service commits one.
func (fixture *deltaFixture) lookupTurn(t *testing.T, body []byte, contextTokens int) *selection.SelectionResult {
	t.Helper()
	lease, state, err := fixture.store.Prepare(context.Background(), fixture.episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	result, err := fixture.selectOnWith(t, state, body, func(arcContext *selection.RaylineARCSelectionContext) {
		arcContext.RoutesLookup, arcContext.RoutesEncodeLimit, arcContext.ContextTokens = true, 200_000, contextTokens
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		TraceContext:          context.Background(),
		RaylineARCTransaction: newRaylineARCEpisodeTransaction(fixture.store, lease, state, fixture.episode, time.Minute, nil),
	}
	bindRaylineARCSelectionTransaction(ctx)
	ctx.RaylineARCTransaction.markSelection(result.RaylineARC.SelectedArm, 10)
	stageRaylineARCPolicySelection(ctx, result.RaylineARC, "")
	if err := commitDecisionOnlyEpisode(context.Background(), ctx); err != nil {
		t.Fatal(err)
	}
	return result
}

// Service-state coherence after a bypass on a delta cell: turn 1 stages and
// appends its reply, so the next turn would go as a delta on that prefix. A
// bypassed lookup commits past it without the service seeing its messages;
// the lookup after it is then sent in full, never as a delta on the ledger's
// prefix, so the service receives the bypassed messages.
func TestALookupAfterABypassSendsTheBypassedMessages(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	fixture.stagedFirstTurn(t)
	bypassed := fixture.lookupTurn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult), 1_000_000)
	if bypassed.RaylineARC.OfferKind != raylineRoutesEncodeBypassOfferKind {
		t.Fatalf("turn 2 was not bypassed: %s", bypassed.RaylineARC.OfferKind)
	}
	if sent := fixture.fake.received(); len(sent) != 1 {
		t.Fatalf("the bypass reached the service: %d decides", len(sent))
	}
	next := policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult,
		map[string]any{"role": "assistant", "content": "README.md is the only file."},
		map[string]any{"role": "user", "content": "thanks"})
	logs := captureLogs(t)
	decided := fixture.lookupTurn(t, next, 1000)
	// Turn 1's staged record belongs to the turn the bypass committed past.
	if line := findLogEvent(t, logs, "rayline_arc_policy_staged_turn"); line["delta_mismatch_reason"] != "stale_turn" {
		t.Fatalf("staged-turn line = %v", line)
	}
	if decided.RaylineARC.OfferKind == raylineRoutesEncodeBypassOfferKind {
		t.Fatal("the lookup under the limit was bypassed")
	}
	sent := fixture.fake.received()
	last := sent[len(sent)-1]
	var messages []json.RawMessage
	if err := json.Unmarshal(last.Request.Messages, &messages); err != nil {
		t.Fatalf("decide messages: %v", err)
	}
	if last.TurnDelta != nil || last.Request.Shell != "" || len(messages) != 5 {
		t.Fatalf("decide after the bypass: delta %+v, shell %q, %d messages; want the whole request",
			last.TurnDelta, last.Request.Shell, len(messages))
	}
	if !slices.ContainsFunc(last.Attribution, func(entry raylinearc.PolicyAttribution) bool {
		return entry.Message == 1 && entry.ActionID == fixture.decided
	}) {
		t.Fatalf("attribution %+v lost turn 1's reply", last.Attribution)
	}
}
