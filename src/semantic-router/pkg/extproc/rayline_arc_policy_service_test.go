package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const (
	policyTestAlias   = "rayline/test-fold0"
	policyTestUp      = "Until the next steering instruction, reason more thoroughly before acting."
	policyTestPackage = "7938aa0be9d0bb6bc0ae14b64a5c319905e12c34b53c0bab9c940a24d35443f0"
)

func policyTestEffort(value string) *string { return &value }

// policyTestBindings are three actions over two workers: the thinking worker
// at its neutral level and steered up, and the thinking-off worker.
func policyTestBindings() []config.RaylineARCPolicyBinding {
	bind := func(worker, level, model string, effort *string, suffix string) config.RaylineARCPolicyBinding {
		return config.RaylineARCPolicyBinding{
			ActionID: config.RaylineARCPolicyActionID(model, effort, nil, suffix),
			Worker:   worker, Level: level, Model: model, Effort: effort,
		}
	}
	return []config.RaylineARCPolicyBinding{
		bind("think", "none", "vendor/think", policyTestEffort("high"), ""),
		bind("think", "up", "vendor/think", policyTestEffort("high"), policyTestUp),
		bind("off", "none", "vendor/off", policyTestEffort("none"), ""),
	}
}

func policyTestActionIDs(bindings []config.RaylineARCPolicyBinding) []string {
	ids := make([]string, len(bindings))
	for index, binding := range bindings {
		ids[index] = binding.ActionID
	}
	return ids
}

func policyTestDecision(schedule string) *config.Decision {
	on, off := true, false
	return &config.Decision{
		Name: "arc-policy",
		ModelRefs: []config.ModelRef{
			{Model: "think", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &on}},
			{Model: "off", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &off}},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: config.RaylineARCAlgorithmType, OnError: "fail_closed",
			RaylineARC: &config.RaylineARCAlgorithmConfig{PolicyService: &config.RaylineARCPolicyServiceConfig{
				PackageAlias: policyTestAlias, PackageSHA256: policyTestPackage,
				Bindings: policyTestBindings(), ModelSchedule: schedule,
			}},
		},
	}
}

type policySelectorFixture struct {
	fake     *fakePolicyService
	selector *raylineARCSelector
	decision *config.Decision
}

func newPolicySelectorFixture(t *testing.T, schedule string) *policySelectorFixture {
	t.Helper()
	decision := policyTestDecision(schedule)
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, policyTestActionIDs(decision.Algorithm.RaylineARC.PolicyService.Bindings))
	selector := newRaylineARCSelector(nil, nil, nil, policyTestPackage)
	selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fake.URL(), TotalTimeout: 5 * time.Second}),
	})
	return &policySelectorFixture{fake: fake, selector: selector, decision: decision}
}

func policyTestRequest(t *testing.T, messages ...map[string]any) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{"model": "auto", "max_tokens": 1024, "system": "be brief", "messages": messages})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func (fixture *policySelectorFixture) selectOn(
	t *testing.T,
	state *raylinearc.EpisodeState,
	body []byte,
) (*selection.SelectionResult, error) {
	t.Helper()
	return fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64),
			State:         state,
			RawRequest:    body,
			RequestFormat: policyFormatAnthropic,
		},
	})
}

func TestPolicySelectorMapsTheDecisionToAWorkerAndLevel(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[1].ActionID })
	state, _ := raylinearc.NewEpisodeState(2)

	turn0 := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	result, err := fixture.selectOn(t, state, turn0)
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	if result.SelectedModel != "think" || result.RaylineARC.ThinkingLevel != "up" ||
		result.RaylineARC.PolicyActionID != bindings[1].ActionID {
		t.Fatalf("decision = model %q level %q action %q", result.SelectedModel,
			result.RaylineARC.ThinkingLevel, result.RaylineARC.PolicyActionID)
	}
	sent := fixture.fake.received()[0]
	if sent.ContextEpoch != "0" || len(sent.Attribution) != 0 || len(sent.Selection.AvailableActionIDs) != 3 ||
		sent.Package.Alias != policyTestAlias || sent.RequestFormat != policyFormatAnthropic {
		t.Fatalf("first decide request = %+v", sent)
	}

	// The committed turn's reply comes back attributed to the action that
	// produced it, in the same epoch.
	state.Policy = result.RaylineARC.PolicyNextState
	state.TurnIndex = 1
	turn1 := policyTestRequest(t,
		map[string]any{"role": "user", "content": "fix the bug"},
		map[string]any{"role": "assistant", "content": "fixed"},
		map[string]any{"role": "user", "content": "now test it"})
	if _, err := fixture.selectOn(t, state, turn1); err != nil {
		t.Fatalf("second select: %v", err)
	}
	second := fixture.fake.received()[1]
	if second.ContextEpoch != "0" || len(second.Attribution) != 1 ||
		second.Attribution[0].Message != 1 || second.Attribution[0].ActionID != bindings[1].ActionID {
		t.Fatalf("second decide request attribution = %+v epoch %q", second.Attribution, second.ContextEpoch)
	}

	// A client that rewrote history starts a new epoch with no attribution.
	rewritten := policyTestRequest(t, map[string]any{"role": "user", "content": "summary of earlier work"})
	if _, err := fixture.selectOn(t, state, rewritten); err != nil {
		t.Fatalf("third select: %v", err)
	}
	if third := fixture.fake.received()[2]; third.ContextEpoch != "1" || len(third.Attribution) != 0 {
		t.Fatalf("rewritten history: epoch %q attribution %+v", third.ContextEpoch, third.Attribution)
	}
}

// Between schedule boundaries the model is held: only the held worker's
// actions -- its levels -- are offered.
func TestPolicySelectorScheduleOffersOnlyTheHeldModelsLevels(t *testing.T) {
	fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	state, _ := raylinearc.NewEpisodeState(2)
	held := 0
	state.PreviousArm, state.TurnIndex = &held, 2
	// The episode began at turn 0 and has not been compacted since; with no
	// stored epoch, this turn would itself start one, which is a boundary.
	state.Policy = &raylinearc.PolicyEpisodeState{PrefixDigest: raylinearc.MessagesDigest(nil, 0)}
	if _, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"})); err != nil {
		t.Fatalf("select: %v", err)
	}
	offered := fixture.fake.received()[0].Selection.AvailableActionIDs
	if len(offered) != 2 || offered[0] != bindings[0].ActionID || offered[1] != bindings[1].ActionID {
		t.Fatalf("held turn offered %v, want the think worker's two levels", offered)
	}

	// At a boundary every action is offered again.
	state.TurnIndex = 5
	if _, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"})); err != nil {
		t.Fatalf("select at boundary: %v", err)
	}
	if offered := fixture.fake.received()[1].Selection.AvailableActionIDs; len(offered) != 3 {
		t.Fatalf("boundary turn offered %v", offered)
	}
}

func TestPolicySelectorFailsClosed(t *testing.T) {
	cases := []struct {
		name      string
		configure func(*policySelectorFixture)
		wantClass string
	}{
		{"a service error", func(f *policySelectorFixture) {
			f.fake.failNext("backend_unavailable")
		}, "policy_service_backend_unavailable"},
		{"a selected action with no binding", func(f *policySelectorFixture) {
			f.fake.catalog = append(f.fake.catalog, strings.Repeat("f", 64))
			f.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return strings.Repeat("f", 64) })
		}, "policy_catalog_unbound"},
		{"a catalog action with no binding", func(f *policySelectorFixture) {
			f.fake.catalog = append(f.fake.catalog, strings.Repeat("f", 64))
		}, "policy_catalog_unbound"},
		{"an unreachable service", func(f *policySelectorFixture) {
			f.fake.server.Close()
		}, "policy_service_transport"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			fixture := newPolicySelectorFixture(t, "")
			bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
			fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
			test.configure(fixture)
			state, _ := raylinearc.NewEpisodeState(2)
			_, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
			var failure *raylineARCSelectionFailure
			if !errors.As(err, &failure) || failure.class != test.wantClass {
				t.Fatalf("error = %v, want class %s", err, test.wantClass)
			}
		})
	}
}
