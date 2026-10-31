package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A policy-service selection logs the decide call's httptrace as numeric
// policy_http_ fields; one with no traced call carries none.
func TestRaylineARCSelectionLogCarriesThePolicyHTTPTiming(t *testing.T) {
	logs := captureLogs(t)
	trace := &selection.RaylineARCTrace{
		SelectedArm:    0,
		PolicyActionID: "action-0",
		PolicyHTTP: &raylinearc.PolicyHTTPTiming{
			Reused: true, WasIdle: true, IdleMillis: 1200,
			GotConnMillis: 0.2, WroteMillis: 0.5, FirstByteMillis: 61,
		},
	}
	observeRaylineARCSelection(&RequestContext{RequestID: "r"}, trace)
	fields := findLogEvent(t, logs, "rayline_arc_selection")
	want := map[string]interface{}{
		"policy_http_reused":           int64(1),
		"policy_http_was_idle":         int64(1),
		"policy_http_idle_ms":          1200.0,
		"policy_http_got_conn_ms":      0.2,
		"policy_http_dns_ms":           0.0,
		"policy_http_connect_ms":       0.0,
		"policy_http_tls_ms":           0.0,
		"policy_http_wrote_request_ms": 0.5,
		"policy_http_first_byte_ms":    61.0,
	}
	for key, value := range want {
		if fields[key] != value {
			t.Fatalf("%s = %#v, want %#v", key, fields[key], value)
		}
	}
	logs.TakeAll()
	trace.PolicyHTTP = nil
	observeRaylineARCSelection(&RequestContext{RequestID: "r"}, trace)
	if _, present := findLogEvent(t, logs, "rayline_arc_selection")["policy_http_reused"]; present {
		t.Fatal("policy_http_reused present without a traced call")
	}
}

// The timing reaches the selection trace from a real decide exchange.
func TestRaylineARCPolicySelectionCarriesTheDecideHTTPTiming(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	state, _ := raylinearc.NewEpisodeState(2)
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatal(err)
	}
	timing := result.RaylineARC.PolicyHTTP
	if timing == nil || timing.FirstByteMillis <= 0 {
		t.Fatalf("selection trace PolicyHTTP = %+v, want the decide exchange's timing", timing)
	}
}

// A held selection built from an answered capacity refusal carries the
// refused exchange's timing: the decide call was made and answered.
func TestRaylineARCCapacityHoldCarriesTheDecideHTTPTiming(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	state, _ := raylinearc.NewEpisodeState(2)
	servedTurn(t, fixture, state, bindings[1].ActionID)
	fixture.fake.mu.Lock()
	fixture.fake.failWith = "context_exceeds_encoder_capacity"
	fixture.fake.failDetail = map[string]any{"token_count": 300000, "max_tokens": 262144}
	fixture.fake.mu.Unlock()
	result, err := fixture.selectOn(t, state, nextTurn(t))
	if err != nil {
		t.Fatal(err)
	}
	if timing := result.RaylineARC.PolicyHTTP; timing == nil || timing.FirstByteMillis <= 0 {
		t.Fatalf("capacity hold PolicyHTTP = %+v, want the refused exchange's timing", timing)
	}
}
