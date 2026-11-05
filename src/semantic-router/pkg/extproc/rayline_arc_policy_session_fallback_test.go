package extproc

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// newSaturatedSessionFixture is a sessioned cell whose Server refuses every
// session start, as one whose session slots are full does, with a
// sessionless fallback deployment when withFallback is set.
func newSaturatedSessionFixture(t *testing.T, withFallback bool) (*policySelectorFixture, *fakePolicyService) {
	t.Helper()
	decision := policyTestDecision("")
	catalog := policyTestActionIDs(decision.Algorithm.RaylineARC.PolicyService.Bindings)
	saturated := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(saturated.Close)
	clientFor := func(baseURL string) *raylinearc.PolicyServiceClient {
		return raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
			BaseURL: baseURL, ModalKey: "wk-test", ModalSecret: "ws-test", TotalTimeout: 5 * time.Second,
		})
	}
	sessionConfig := raylinearc.PolicySessionConfig{StartTimeout: 2 * time.Second, ReplacedGrace: time.Millisecond}
	var sessionless *fakePolicyService
	if withFallback {
		sessionless = newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
		sessionless.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return catalog[0] })
		sessionConfig.Sessionless = clientFor(sessionless.URL())
	}
	sessions, err := raylinearc.NewPolicySessionBackend(clientFor(saturated.URL), sessionConfig)
	if err != nil {
		t.Fatal(err)
	}
	selector := newRaylineARCSelector(nil, nil, nil, policyTestPackage)
	selector.arm(&raylineARCArmedComponents{
		scorer:         newPolicyServiceScorer(&config.RouterConfig{}, decision),
		admission:      raylinearc.NewAdmissionGate(0),
		policySessions: sessions,
	})
	return &policySelectorFixture{selector: selector, decision: decision}, sessionless
}

// A turn whose session cannot start is decided on the sessionless fallback
// and dispatched; its trace labels it and records no session, so the episode
// starts one on its next turn.
func TestPolicySessionStartFailureDecidesSessionless(t *testing.T) {
	fixture, sessionless := newSaturatedSessionFixture(t, true)
	state, _ := raylinearc.NewEpisodeState(2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})

	result, err := fixture.selectOn(t, state, body)
	if err != nil {
		t.Fatalf("the turn failed: %v", err)
	}
	trace := result.RaylineARC
	if trace.PolicySessionFallback != raylinearc.PolicySessionFallbackSessionless ||
		trace.PolicySessionFallbackReason != raylinearc.PolicySessionFallbackStartRefused {
		t.Fatalf("fallback %q reason %q", trace.PolicySessionFallback, trace.PolicySessionFallbackReason)
	}
	if trace.PolicySession != nil || trace.PolicySessionReplaced != nil || trace.PolicyStaging != nil {
		t.Fatalf("a sessionless turn recorded a session or staged: %+v", trace)
	}
	if trace.PolicyActionID == "" || len(sessionless.received()) != 1 {
		t.Fatalf("action %q, fallback decides %d", trace.PolicyActionID, len(sessionless.received()))
	}
}

// With no fallback configured (the default) the turn fails closed as before:
// policy_service_session_start.
func TestPolicySessionStartFailureWithoutFallbackFailsClosed(t *testing.T) {
	fixture, _ := newSaturatedSessionFixture(t, false)
	state, _ := raylinearc.NewEpisodeState(2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})

	_, err := fixture.selectOn(t, state, body)
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "policy_service_"+raylinearc.PolicySessionStartClass {
		t.Fatalf("err = %v, want policy_service_session_start", err)
	}
}

// llm_usage labels a turn decided on the sessionless fallback, so evaluations
// can score it apart, and leaves both fields null on every other turn.
func TestUsageRecordLabelsSessionlessFallback(t *testing.T) {
	cases := []struct {
		name       string
		trace      *selection.RaylineARCTrace
		wantLabel  interface{}
		wantReason interface{}
	}{
		{"sessionless", &selection.RaylineARCTrace{
			PolicySessionFallback:       raylinearc.PolicySessionFallbackSessionless,
			PolicySessionFallbackReason: raylinearc.PolicySessionFallbackStartTimeout,
		}, "sessionless", "session_start_timeout"},
		{"sessioned", &selection.RaylineARCTrace{}, nil, nil},
		{"not ARC", nil, nil, nil},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			logs := captureLogs(t)
			ctx := &RequestContext{RequestID: "rt_fallback-000", RequestModel: "priced-model", VSRRaylineARC: test.trace}
			usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, responseUsageMetrics{
				promptTokens: 10, promptTokensReported: true, completionTokens: 5, completionTokensReported: true,
			})
			fields := findLogEvent(t, logs, "llm_usage")
			if value, present := fields["policy_session_fallback"]; !present || value != test.wantLabel {
				t.Fatalf("policy_session_fallback = %v (present %v), want %v", value, present, test.wantLabel)
			}
			if value, present := fields["policy_session_fallback_reason"]; !present || value != test.wantReason {
				t.Fatalf("policy_session_fallback_reason = %v (present %v), want %v", value, present, test.wantReason)
			}
		})
	}
}
