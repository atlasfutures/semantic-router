package extproc

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// productionPolicyStartRetryDelay is the backoff the package ships, kept
// before init shortens it for every test in the package.
var productionPolicyStartRetryDelay = policyStartRetryDelay

func init() {
	policyStartRetryDelay = func() time.Duration { return 10 * time.Millisecond }
}

func startRetries(outcome string) float64 {
	return testutil.ToFloat64(metrics.RaylineARCPolicySessionStartRetries.WithLabelValues(outcome))
}

func TestPolicyStartRetryDelayIsJitteredAroundASecond(t *testing.T) {
	for range 200 {
		if delay := productionPolicyStartRetryDelay(); delay < policyStartRetryDelayMin || delay > policyStartRetryDelayMax {
			t.Fatalf("delay %v outside [%v, %v]", delay, policyStartRetryDelayMin, policyStartRetryDelayMax)
		}
	}
}

// An episode's first decide answered 503 backend_unavailable is asked once
// more: a second 503 fails the turn closed with the class it always had, and
// any other failure is not asked again.
func TestPolicyStartRetriesA503BackendUnavailableOnce(t *testing.T) {
	for _, tc := range []struct {
		name      string
		configure func(*fakePolicyService)
		wantClass string
		wantCalls int
		outcome   string
	}{
		{"recovers", func(fake *fakePolicyService) { fake.failFirstCalls("backend_unavailable", 1) }, "", 2, "recovered"},
		{"still down", func(fake *fakePolicyService) { fake.failNext("backend_unavailable") }, "policy_service_backend_unavailable", 2, "failed"},
		{"another refusal", func(fake *fakePolicyService) { fake.failFirstCalls("session_capacity", 1) }, "policy_service_session_capacity", 1, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newPolicySelectorFixture(t, "")
			bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
			fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
			tc.configure(fixture.fake)
			before := map[string]float64{"recovered": startRetries("recovered"), "failed": startRetries("failed")}
			state, _ := raylinearc.NewEpisodeState(2)
			result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
			if tc.wantClass == "" {
				if err != nil || result.RaylineARC.PolicyActionID != bindings[0].ActionID {
					t.Fatalf("select = %v, %v; want the decision", result, err)
				}
			} else if class := selectionFailureClassOf(t, err); class != tc.wantClass {
				t.Fatalf("class %s, want %s", class, tc.wantClass)
			}
			if calls := len(fixture.fake.received()); calls != tc.wantCalls {
				t.Fatalf("decides = %d, want %d", calls, tc.wantCalls)
			}
			for outcome, count := range before {
				want := count
				if outcome == tc.outcome {
					want++
				}
				if got := startRetries(outcome); got != want {
					t.Fatalf("retry %s = %v, want %v", outcome, got, want)
				}
			}
		})
	}
}

// A turn that extends a committed episode, sent as a delta, and its append
// may already have advanced the service: a 503 on either is not asked again.
func TestPolicyStartRetryLeavesExtendingTurnsAndAppends(t *testing.T) {
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	fixture.fake.appendStatus = http.StatusServiceUnavailable
	fixture.stagedFirstTurn(t)
	if appends := fixture.fake.receivedAppends(); len(appends) != 1 {
		t.Fatalf("appends = %d after a 503, want 1", len(appends))
	}

	fixture.fake.failNext("backend_unavailable")
	before := len(fixture.fake.received())
	_, state, err := fixture.store.Prepare(context.Background(), fixture.episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	_, err = fixture.selectOnWith(t, state, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult),
		func(*selection.RaylineARCSelectionContext) {})
	if class := selectionFailureClassOf(t, err); class != "policy_service_backend_unavailable" {
		t.Fatalf("class %s, want policy_service_backend_unavailable", class)
	}
	if calls := len(fixture.fake.received()) - before; calls != 1 {
		t.Fatalf("extending turn decides = %d, want 1", calls)
	}
}

func TestRetryPolicyStartOnceBounds(t *testing.T) {
	unavailable := &raylinearc.PolicyServiceError{Class: "backend_unavailable", Status: http.StatusServiceUnavailable}
	for _, tc := range []struct {
		name       string
		transition string
		err        error
		timeout    time.Duration
		wantCalls  int
	}{
		{"start", raylinearc.PolicyTransitionStart, unavailable, 5 * time.Second, 1},
		{"no deadline", raylinearc.PolicyTransitionStart, unavailable, 0, 1},
		{"short deadline", raylinearc.PolicyTransitionStart, unavailable, 100 * time.Millisecond, 0},
		{"extends", raylinearc.PolicyTransitionExtends, unavailable, 0, 0},
		{"prefix break", raylinearc.PolicyTransitionPrefixBreak, unavailable, 0, 0},
		{"side call", raylinearc.PolicyTransitionSideCall, unavailable, 0, 0},
		{"a 500", raylinearc.PolicyTransitionStart, &raylinearc.PolicyServiceError{Class: "backend_unavailable", Status: http.StatusInternalServerError}, 0, 0},
		{"transport", raylinearc.PolicyTransitionStart, &raylinearc.PolicyServiceError{Class: "transport", Status: http.StatusServiceUnavailable}, 0, 0},
		{"timeout", raylinearc.PolicyTransitionStart, &raylinearc.PolicyServiceError{Class: "timeout"}, 0, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			if tc.timeout > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, tc.timeout)
				defer cancel()
			}
			calls := 0
			_, err := retryPolicyStartOnce(ctx, tc.transition, "e", nil, tc.err, func() (*raylinearc.PolicyDecisionResponse, error) {
				calls++
				return &raylinearc.PolicyDecisionResponse{}, nil
			})
			if calls != tc.wantCalls {
				t.Fatalf("retries = %d, want %d", calls, tc.wantCalls)
			}
			if tc.wantCalls == 0 && !errors.Is(err, tc.err) {
				t.Fatalf("err = %v, want the first failure", err)
			}
		})
	}
}
