package extproc

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// startFallbackFixture is a policy cell whose package names bindings[1] as
// its fallback action, with a start deadline short enough for a test.
func startFallbackFixture(t *testing.T) (*policySelectorFixture, []config.RaylineARCPolicyBinding) {
	t.Helper()
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	scorer := fixture.selector.armedComponents().scorer.(*policyServiceScorer)
	scorer.fallbackActionID = bindings[1].ActionID
	scorer.startDeadline = 100 * time.Millisecond
	return fixture, bindings
}

// A start the service has not decided by the start deadline is served the
// package's fallback action and commits no policy state, so the episode's
// next turn starts it again; a start decided in time is unchanged.
func TestASlowStartIsServedThePackagesFallback(t *testing.T) {
	fixture, bindings := startFallbackFixture(t)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "go"})

	state, _ := raylinearc.NewEpisodeState(2)
	fast, err := fixture.selectOn(t, state, body)
	if err != nil || fast.RaylineARC.PolicyActionID != bindings[0].ActionID ||
		fast.RaylineARC.OfferKind == policyStartFallbackOfferKind {
		t.Fatalf("a fast start = %+v, %v; want the policy's decision", fast, err)
	}

	fixture.fake.decideDelay = 400 * time.Millisecond
	logs := captureLogs(t)
	started := time.Now()
	slow, err := fixture.selectOn(t, state, body)
	if err != nil {
		t.Fatalf("a slow start failed: %v", err)
	}
	if elapsed := time.Since(started); elapsed >= 400*time.Millisecond {
		t.Fatalf("the slow start waited %v, past its deadline", elapsed)
	}
	trace := slow.RaylineARC
	if trace.PolicyActionID != bindings[1].ActionID || trace.OfferKind != policyStartFallbackOfferKind {
		t.Fatalf("served %s (%s), want the fallback %s", trace.PolicyActionID, trace.OfferKind, bindings[1].ActionID)
	}
	if trace.PolicyNextState != nil || trace.RawScores != nil || !trace.TokenCountsUnknown {
		t.Fatalf("a fallback committed policy state or reported scores: %+v", trace)
	}
	line := findLogEvent(t, logs, "rayline_arc_start_deadline_fallback")
	if line["reason"] != policyStartLapseDeadline || line["action_id"] != bindings[1].ActionID {
		t.Fatalf("fallback line = %v", line)
	}
}

// Without a fallback action to serve, a slow start waits for its decision as
// before.
func TestASlowStartWithNoFallbackWaitsForTheDecision(t *testing.T) {
	fixture, bindings := startFallbackFixture(t)
	fixture.selector.armedComponents().scorer.(*policyServiceScorer).fallbackActionID = ""
	fixture.fake.decideDelay = 300 * time.Millisecond
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil || result.RaylineARC.PolicyActionID != bindings[0].ActionID {
		t.Fatalf("select = %+v, %v; want the policy's decision", result, err)
	}
}

// A turn that extends a committed episode is not a start: it keeps the
// turn's own deadline however long the service takes.
func TestAnExtendingTurnIsNotBoundByTheStartDeadline(t *testing.T) {
	fixture, bindings := startFallbackFixture(t)
	state, _ := raylinearc.NewEpisodeState(2)
	first, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
	if err != nil {
		t.Fatal(err)
	}
	state.Policy = first.RaylineARC.PolicyNextState
	state.TurnIndex = 1
	fixture.fake.decideDelay = 300 * time.Millisecond
	next, err := fixture.selectOn(t, state, policyTestRequest(t,
		map[string]any{"role": "user", "content": "fix the bug"},
		map[string]any{"role": "assistant", "content": "fixed"},
		map[string]any{"role": "user", "content": "now test it"}))
	if err != nil || next.RaylineARC.PolicyActionID != bindings[0].ActionID ||
		next.RaylineARC.OfferKind == policyStartFallbackOfferKind {
		t.Fatalf("extending turn = %+v, %v; want the policy's decision", next, err)
	}
}

// A /v1/routes lookup keeps its own semantics: it is never answered with the
// fallback for a slow start.
func TestALookupIsNotBoundByTheStartDeadline(t *testing.T) {
	fixture, bindings := startFallbackFixture(t)
	fixture.fake.decideDelay = 300 * time.Millisecond
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.lookupOn(t, state, lookupBody(t), 1000, 0, nil)
	if err != nil || result.RaylineARC.PolicyActionID != bindings[0].ActionID {
		t.Fatalf("lookup = %+v, %v; want the policy's decision", result, err)
	}
}

// On a sessioned Server, a start that lapses ends the session it started (no
// commit records it) and records none, so the episode's next turn starts a
// new one.
func TestASessionedStartThatLapsesEndsItsSession(t *testing.T) {
	fixture, front := newPolicySessionFixture(t)
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	scorer := fixture.selector.armedComponents().scorer.(*policyServiceScorer)
	scorer.fallbackActionID = bindings[1].ActionID
	scorer.startDeadline = 100 * time.Millisecond
	for _, container := range front.containers {
		container.decideDelay = 400 * time.Millisecond
	}
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatalf("a lapsed sessioned start failed: %v", err)
	}
	if result.RaylineARC.PolicyActionID != bindings[1].ActionID || result.RaylineARC.PolicySession != nil {
		t.Fatalf("lapsed start trace %+v", result.RaylineARC)
	}
	waitForTerminates(t, front, 1)
}

func TestPolicyStartLapseReason(t *testing.T) {
	live := context.Background()
	lapsed, cancel := context.WithTimeout(live, time.Nanosecond)
	defer cancel()
	<-lapsed.Done()
	sessionStart := &raylinearc.PolicyServiceError{Class: raylinearc.PolicySessionStartClass}
	for _, tc := range []struct {
		name      string
		decideCtx context.Context
		err       error
		want      string
	}{
		{"deadline", lapsed, errors.New("context deadline exceeded"), policyStartLapseDeadline},
		{"session start", live, sessionStart, policyStartLapseSessionStart},
		{"another failure", live, &raylinearc.PolicyServiceError{Class: "backend_unavailable"}, ""},
	} {
		if got := policyStartLapseReason(tc.decideCtx, tc.err); got != tc.want {
			t.Fatalf("%s: reason %q, want %q", tc.name, got, tc.want)
		}
	}
	// The turn's own deadline is the turn's outcome, not a lapsed start.
	if policyStartLapsed(lapsed, lapsed, sessionStart) {
		t.Fatal("a turn out of its own time counted as a lapsed start")
	}
}

func TestPolicyStartDeadlineConfig(t *testing.T) {
	for _, tc := range []struct {
		configured, total int
		want              time.Duration
	}{
		{0, 30, 8 * time.Second},
		{3, 30, 3 * time.Second},
		{0, 5, 5 * time.Second},
	} {
		cfg := &config.RaylineARCPolicyServiceConfig{StartDeadlineSeconds: tc.configured, TotalTimeoutSeconds: tc.total}
		if got := cfg.StartDeadline(); got != tc.want {
			t.Fatalf("start_deadline_seconds %d total %d: %v, want %v", tc.configured, tc.total, got, tc.want)
		}
	}
}
