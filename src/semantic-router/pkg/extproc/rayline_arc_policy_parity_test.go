package extproc

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A format the policy service cannot read is refused before the episode lease,
// so the episode is untouched and nothing waits on a service call.
func TestPolicyModeRefusesAnUnreadableFormatBeforeTheLease(t *testing.T) {
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{RaylineARCEpisodeStore: store}
	requestContext := &RequestContext{
		Headers:      map[string]string{"x-rayline-session": t.Name()},
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "go"}}},
		}},
		TraceContext: context.Background(),
	}
	algorithm := &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType, RaylineARC: &config.RaylineARCAlgorithmConfig{
		PolicyService: &config.RaylineARCPolicyServiceConfig{},
		Episode:       config.RaylineARCEpisodeConfig{IDHeader: "x-rayline-session", AcquireTimeoutSeconds: 1, LeaseTTLSeconds: 30},
	}}
	selectionContext := router.buildRaylineARCSelectionContext(algorithm, requestContext,
		[]config.ModelRef{{Model: "arm-0"}, {Model: "arm-1"}}, raylineARCEpisodeRequired)
	if selectionContext.PreparationFailure != arcFailurePolicyRequestFormat || requestContext.RaylineARCTransaction != nil {
		t.Fatalf("failure %q, transaction %v", selectionContext.PreparationFailure, requestContext.RaylineARCTransaction)
	}
	// The episode was never leased: a second request acquires it at once.
	lease, _, err := store.Prepare(context.Background(), raylinearc.HashEpisodeID(t.Name()), 2)
	if err != nil {
		t.Fatalf("the refused request left the episode leased: %v", err)
	}
	_ = store.Abort(context.Background(), lease)
}

// A service error code becomes a metric label, so only the contract's codes
// pass through, and the service's own back-pressure answers 429.
func TestPolicyServiceFailureClassesAreBounded(t *testing.T) {
	cases := map[string]string{
		"session_busy":                     "policy_service_session_busy",
		"backend_unavailable":              "policy_service_backend_unavailable",
		"a-new-code\nwith text":            "policy_service_service_error",
		"context_exceeds_encoder_capacity": "policy_service_context_exceeds_encoder_capacity",
	}
	for code, want := range cases {
		fixture := newPolicySelectorFixture(t, "")
		bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
		fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
		fixture.fake.failNext(code)
		state, _ := raylinearc.NewEpisodeState(2)
		_, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
		var failure *raylineARCSelectionFailure
		if !errors.As(err, &failure) || failure.class != want {
			t.Fatalf("%q: error = %v, want class %s", code, err, want)
		}
	}
	for class, contended := range map[string]bool{
		"policy_service_session_busy":        true,
		"policy_service_session_capacity":    true,
		"policy_service_backend_unavailable": false,
	} {
		if selectionFailureIsContended(class) != contended {
			t.Fatalf("%s contended = %v", class, !contended)
		}
	}
}

func TestPolicySessionActionIsBounded(t *testing.T) {
	for action, want := range map[string]string{"appended": "appended", "reused": "reused", "evicted-by-gc": ""} {
		if got := boundedPolicySessionAction(action); got != want {
			t.Fatalf("%q -> %q, want %q", action, got, want)
		}
	}
}
