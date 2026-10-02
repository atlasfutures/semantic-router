package extproc

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// selectCall runs one selection of the given call kind on the fixture.
func (fixture *policySelectorFixture) selectCall(
	t *testing.T,
	kind raylinearc.PolicyCallKind,
) (*selection.SelectionResult, error) {
	t.Helper()
	state, _ := raylinearc.NewEpisodeState(2)
	return fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash:        strings.Repeat("e", 64),
			State:                state,
			RawRequest:           policyTestRequest(t, map[string]any{"role": "user", "content": "go"}),
			RequestFormat:        policyFormatAnthropic,
			PolicyCallKind:       kind,
			PolicyCallKindSource: "header",
		},
	})
}

func (fixture *policySelectorFixture) lastEpisodeMode(t *testing.T) string {
	t.Helper()
	sent := fixture.fake.received()
	if len(sent) == 0 {
		t.Fatal("no decide request reached the service")
	}
	return sent[len(sent)-1].EpisodeMode
}

func newSideCallFixture(t *testing.T, consistency string) *policySelectorFixture {
	t.Helper()
	fixture := newPolicySelectorFixtureWith(t, consistency)
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	return fixture
}

// A side call commits nothing and holds no lease, so even a strict cell asks
// for it relaxed: the service locks no session and leaves the main
// conversation's untouched. The cell's own turns stay strict.
func TestStrictPolicyCellDecidesSideCallsRelaxed(t *testing.T) {
	fixture := newSideCallFixture(t, config.RaylineARCConsistencyStrict)
	if _, err := fixture.selectCall(t, raylinearc.PolicyCallSide); err != nil {
		t.Fatalf("side call: %v", err)
	}
	if mode := fixture.lastEpisodeMode(t); mode != raylinearc.PolicyEpisodeModeRelaxed {
		t.Fatalf("side call sent episode_mode %q, want relaxed", mode)
	}
	if _, err := fixture.selectCall(t, raylinearc.PolicyCallMain); err != nil {
		t.Fatalf("main turn: %v", err)
	}
	if mode := fixture.lastEpisodeMode(t); mode != "" {
		t.Fatalf("main turn sent episode_mode %q, want strict (absent)", mode)
	}
}

// A service that predates episode_mode answers a relaxed side call strict,
// with a revision. That is how side calls were served before, so the call is
// served, and side calls go strict until the retry window passes.
func TestSideCallOnAServiceIgnoringTheModeIsServedThenSentStrict(t *testing.T) {
	fixture := newSideCallFixture(t, config.RaylineARCConsistencyStrict)
	clock := time.Now()
	fixture.selector.now = func() time.Time { return clock }
	fixture.fake.mu.Lock()
	fixture.fake.ignoreEpisodeMode = true
	fixture.fake.mu.Unlock()

	if _, err := fixture.selectCall(t, raylinearc.PolicyCallSide); err != nil {
		t.Fatalf("side call answered strict: %v", err)
	}
	if _, err := fixture.selectCall(t, raylinearc.PolicyCallSide); err != nil {
		t.Fatalf("second side call: %v", err)
	}
	if mode := fixture.lastEpisodeMode(t); mode != "" {
		t.Fatalf("side call after a strict answer sent %q, want strict", mode)
	}
	clock = clock.Add(policySideCallStrictFor)
	if _, err := fixture.selectCall(t, raylinearc.PolicyCallSide); err != nil {
		t.Fatalf("side call after the window: %v", err)
	}
	if mode := fixture.lastEpisodeMode(t); mode != raylinearc.PolicyEpisodeModeRelaxed {
		t.Fatalf("side call after the window sent %q, want relaxed again", mode)
	}
}

// A service whose package cannot serve relaxed refuses the side call; the
// router decides it strict instead, and sends the next side calls strict
// without asking relaxed first.
func TestSideCallRefusedRelaxedIsDecidedStrict(t *testing.T) {
	fixture := newSideCallFixture(t, config.RaylineARCConsistencyStrict)
	fixture.fake.refuseRelaxed("relaxed_unsupported_by_encoder")

	if _, err := fixture.selectCall(t, raylinearc.PolicyCallSide); err != nil {
		t.Fatalf("refused side call: %v", err)
	}
	sent := fixture.fake.received()
	if len(sent) != 2 || sent[0].EpisodeMode != raylinearc.PolicyEpisodeModeRelaxed || sent[1].EpisodeMode != "" {
		t.Fatalf("decide calls = %d (%v), want a relaxed refusal then a strict retry", len(sent), sent)
	}
	if _, err := fixture.selectCall(t, raylinearc.PolicyCallSide); err != nil {
		t.Fatalf("second side call: %v", err)
	}
	if sent := fixture.fake.received(); len(sent) != 3 || sent[2].EpisodeMode != "" {
		t.Fatalf("second side call made %d calls in all, want one strict call", len(sent))
	}
}

// A relaxed cell's side calls were already relaxed, and a strict answer to
// one still means the service ignored the cell's mode.
func TestRelaxedPolicyCellSideCallStaysRelaxed(t *testing.T) {
	fixture := newSideCallFixture(t, config.RaylineARCConsistencyRelaxed)
	if _, err := fixture.selectCall(t, raylinearc.PolicyCallSide); err != nil {
		t.Fatalf("side call: %v", err)
	}
	if mode := fixture.lastEpisodeMode(t); mode != raylinearc.PolicyEpisodeModeRelaxed {
		t.Fatalf("side call sent %q, want relaxed", mode)
	}
	fixture.fake.mu.Lock()
	fixture.fake.ignoreEpisodeMode = true
	fixture.fake.mu.Unlock()
	if _, err := fixture.selectCall(t, raylinearc.PolicyCallSide); err == nil ||
		!strings.Contains(err.Error(), "policy_session_revision") {
		t.Fatalf("relaxed cell side call answered strict = %v", err)
	}
}

// End to end: a strict policy cell built from config, trusting the gateway's
// turn-signal headers, decides a side call relaxed and a main turn strict.
func TestStrictPolicyCellSendsSideCallsRelaxedEndToEnd(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	fake := newRelaxedPolicyFake(t)
	path := writeConsistentPolicyConfig(t, fake.URL(), config.RaylineARCConsistencyStrict)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	anchor := "          policy_service:\n"
	if !strings.Contains(string(raw), anchor) {
		t.Fatal("the policy e2e config no longer carries the policy_service anchor")
	}
	rendered := strings.Replace(string(raw), anchor, anchor+"            trust_turn_signal_headers: true\n", 1)
	if writeErr := os.WriteFile(path, []byte(rendered), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	for _, tc := range []struct {
		kind string
		want string
	}{
		{"side", raylinearc.PolicyEpisodeModeRelaxed},
		{"main", ""},
	} {
		status, err := policyTurnStatusWithHeaders(router, "episode-side-e2e", "turn "+tc.kind,
			map[string]string{raylineARCCallKindHeader: tc.kind})
		if err != nil || status != 0 {
			t.Fatalf("%s call = %d, %v; want forwarded", tc.kind, status, err)
		}
		sent := fake.received()
		if got := sent[len(sent)-1].EpisodeMode; got != tc.want {
			t.Fatalf("%s call sent episode_mode %q, want %q", tc.kind, got, tc.want)
		}
	}
}
