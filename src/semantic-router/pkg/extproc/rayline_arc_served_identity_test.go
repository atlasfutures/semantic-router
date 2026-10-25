package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func servedIdentityHeaders(ctx *RequestContext, successful bool) map[string]string {
	set := map[string]string{}
	mutation := buildResponseHeaderMutation(ctx, successful)
	if mutation == nil {
		return set
	}
	for _, option := range mutation.SetHeaders {
		set[option.Header.Key] = string(option.Header.RawValue)
	}
	return set
}

func servedIdentityContext() *RequestContext {
	weight := 1 / 4.5
	return &RequestContext{
		RequestID:          "rt_served-000",
		RequestModel:       "priced-model",
		VSRSelectedModel:   "kimi-k3@thinking-off",
		DispatchedOutbound: &dispatchedOutboundParams{Model: "moonshotai/kimi-k3"},
		VSRRaylineARC: &selection.RaylineARCTrace{
			EpisodeIDHash: "episode-hash", SelectedArm: 2,
			PolicyActionID: "action-1", PolicyArmID: "arm-1",
			PolicyPackageSHA256: "package-digest", PolicyCostWeight: &weight,
		},
	}
}

// A served ARC turn names the provider model it was dispatched to, the
// package that decided it and the cost weight it was decided at, so a
// gateway can bill configurable ARC at the served model's own card. The arm
// name stays on x-vsr-selected-model: it is the package's name for the arm,
// not a rate-card id.
func TestARCTurnPublishesItsServedIdentity(t *testing.T) {
	for _, successful := range []bool{true, false} {
		set := servedIdentityHeaders(servedIdentityContext(), successful)
		want := map[string]string{
			headers.VSRServedModel:      "moonshotai/kimi-k3",
			headers.VSRPolicyPackage:    "package-digest",
			headers.VSRPolicyCostWeight: "0.2222222222222222",
			headers.VSRSelectedModel:    "kimi-k3@thinking-off",
		}
		for key, value := range want {
			if set[key] != value {
				t.Errorf("successful=%v: %s = %q, want %q", successful, key, set[key], value)
			}
		}
	}
}

// Nothing is guessed: a turn that never dispatched a body names no served
// model, a decision echoed without a cost weight sends none, and a turn
// outside ARC carries none of the three.
func TestServedIdentityIsNeverGuessed(t *testing.T) {
	undispatched := servedIdentityContext()
	undispatched.DispatchedOutbound = nil
	unweighted := servedIdentityContext()
	unweighted.VSRRaylineARC.PolicyCostWeight = nil
	outside := servedIdentityContext()
	outside.VSRRaylineARC = nil

	if set := servedIdentityHeaders(undispatched, true); set[headers.VSRServedModel] != "" ||
		set[headers.VSRPolicyPackage] != "" || set[headers.VSRPolicyCostWeight] != "" {
		t.Errorf("an undispatched turn published a served identity: %v", set)
	}
	if set := servedIdentityHeaders(unweighted, true); set[headers.VSRPolicyCostWeight] != "" ||
		set[headers.VSRServedModel] != "moonshotai/kimi-k3" {
		t.Errorf("a decision with no echoed cost weight published %v", set)
	}
	if set := servedIdentityHeaders(outside, true); set[headers.VSRServedModel] != "" {
		t.Errorf("a turn outside ARC published a served model: %v", set)
	}
	// A cache hit served nothing upstream.
	cached := servedIdentityContext()
	cached.VSRCacheHit = true
	if set := servedIdentityHeaders(cached, true); set[headers.VSRServedModel] != "" {
		t.Errorf("a cache hit published a served model: %v", set)
	}
}

// The llm_usage record carries the same identity, and null where it is
// unknown.
func TestUsageRecordCarriesTheServedIdentity(t *testing.T) {
	logs := captureLogs(t)
	usageRecordRouter().reportNonStreamingUsage(servedIdentityContext(), time.Second, streamingFlagUsage())
	fields := findLogEvent(t, logs, "llm_usage")
	if fields["served_model"] != "moonshotai/kimi-k3" || fields["policy_cost_weight"] != 1/4.5 {
		t.Errorf("served_model = %v, policy_cost_weight = %v", fields["served_model"], fields["policy_cost_weight"])
	}

	logs = captureLogs(t)
	undispatched := servedIdentityContext()
	undispatched.DispatchedOutbound = nil
	usageRecordRouter().reportNonStreamingUsage(undispatched, time.Second, streamingFlagUsage())
	fields = findLogEvent(t, logs, "llm_usage")
	for _, key := range []string{"served_model", "policy_cost_weight"} {
		if value, present := fields[key]; !present || value != nil {
			t.Errorf("%s = %v (present %v), want null", key, value, present)
		}
	}
}
