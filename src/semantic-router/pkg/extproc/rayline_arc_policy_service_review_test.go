package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The shared fixture's decision selects this action, the second of its three.
const reviewFixtureSelected = "be83da527f7beb71e86f23fdcac94c8365ba49df2ae09050ba7e74958c6b1f90"

// A service that returns an action the router did not offer -- here one on an
// operator-disabled arm -- is refused rather than dispatched.
func TestPolicySelectorRefusesAnActionItDidNotOffer(t *testing.T) {
	runUnofferedActionCase(t, nil, []bool{false, true, false})
}

// A response that selects an action it marks unavailable contradicts itself.
func TestPolicySelectorRefusesASelectionMarkedUnavailable(t *testing.T) {
	runUnofferedActionCase(t, func(response map[string]any) {
		for _, action := range response["actions"].([]any) {
			if entry := action.(map[string]any); entry["action_id"] == reviewFixtureSelected {
				entry["available"] = false
			}
		}
	}, []bool{false, false, false})
}

func runUnofferedActionCase(t *testing.T, mutate func(map[string]any), disabled []bool) {
	t.Helper()
	fixture, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/decision_response.v1.json")
	if err != nil {
		t.Fatal(err)
	}
	if mutate != nil {
		var response map[string]any
		if err = json.Unmarshal(fixture, &response); err != nil {
			t.Fatal(err)
		}
		mutate(response)
		fixture, _ = json.Marshal(response)
	}
	service := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write(fixture)
	}))
	t.Cleanup(service.Close)
	decision := &config.Decision{
		Name:      "arc-policy",
		ModelRefs: []config.ModelRef{{Model: "w0"}, {Model: "w1"}, {Model: "w2"}},
		Algorithm: &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType, RaylineARC: &config.RaylineARCAlgorithmConfig{
			PolicyService: &config.RaylineARCPolicyServiceConfig{
				// The shared fixture's own package, so the decision is this package's.
				PackageAlias: "rayline/arc-example-fold0", PackageSHA256: "81e4a7af2c28c3be9ecbe850888c6773d196d58cad100474cbfb48622af7abe4",
				Bindings: []config.RaylineARCPolicyBinding{
					{ActionID: "b779e919d4709c8bc766f1cc7b4228ed1cd8a5d1f230acfb2453e4f37df6e1f1", Worker: "w0"},
					{ActionID: reviewFixtureSelected, Worker: "w1"},
					{ActionID: "569bf3701d341b1945954bf38e0686b23e404ab2a857983402706a40b2a0fad9", Worker: "w2"},
				},
			},
		}},
	}
	selector := newRaylineARCSelector(nil, nil, nil, "sha")
	selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: service.URL, TotalTimeout: 5 * time.Second}),
	})
	state, _ := raylinearc.NewEpisodeState(3)
	body, _ := json.Marshal(map[string]any{"messages": []map[string]any{{"role": "user", "content": "go"}}})
	_, err = selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName: decision.Name, CandidateModels: decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64), State: state, RawRequest: body,
			RequestFormat: policyFormatAnthropic, DisabledArms: disabled,
		},
	})
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "policy_action_not_offered" {
		t.Fatalf("error = %v, want class policy_action_not_offered", err)
	}
}

// /v1/routes callers execute the route themselves, so the policy action
// travels with it; outside the policy mode the fields stay absent.
func TestRaylineRoutesPayloadCarriesThePolicyAction(t *testing.T) {
	payload := raylineRoutesPayload("route", "", routerruntime.RouteDecision{
		ThinkingLevel: "up", PolicyActionID: reviewFixtureSelected,
	}, nil, time.Millisecond)
	encoded, _ := json.Marshal(payload)
	if !strings.Contains(string(encoded), `"thinking_level":"up"`) ||
		!strings.Contains(string(encoded), `"policy_action_id":"`+reviewFixtureSelected+`"`) {
		t.Fatalf("payload = %s", encoded)
	}
	encoded, _ = json.Marshal(raylineRoutesPayload("route", "", routerruntime.RouteDecision{}, nil, time.Millisecond))
	if strings.Contains(string(encoded), "thinking_level") || strings.Contains(string(encoded), "policy_action_id") {
		t.Fatalf("artifact-mode payload carries policy fields: %s", encoded)
	}
}

func TestRaylineARCDecisionsShareWorkers(t *testing.T) {
	decision := func(models ...string) *config.Decision {
		refs := make([]config.ModelRef, len(models))
		for index, model := range models {
			refs[index] = config.ModelRef{Model: model}
		}
		return &config.Decision{ModelRefs: refs}
	}
	if !raylineARCDecisionsShareWorkers([]*config.Decision{decision("a", "b"), decision("a", "b")}) {
		t.Fatal("identical worker lists refused")
	}
	if raylineARCDecisionsShareWorkers([]*config.Decision{decision("a", "b"), decision("b", "a")}) {
		t.Fatal("a reordered worker list accepted")
	}
	if raylineARCDecisionsShareWorkers([]*config.Decision{decision("a", "b"), decision("a")}) {
		t.Fatal("a shorter worker list accepted")
	}
	reasoning := decision("a", "b")
	on := true
	reasoning.ModelRefs[1].UseReasoning = &on
	if raylineARCDecisionsShareWorkers([]*config.Decision{decision("a", "b"), reasoning}) {
		t.Fatal("a worker list that reasons differently accepted")
	}
}
