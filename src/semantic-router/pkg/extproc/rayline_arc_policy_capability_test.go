//go:build !windows && cgo

package extproc

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// On the policy-service path, a turn that needs a capability (an image in a
// tool result) is offered only the arms whose model card holds it, and
// served by one of them; when no arm holds it the turn fails as
// no_capable_arm, the replayable class, before any decide call.
func TestPolicySelectorRoutesCapabilityTurnsToCapableArms(t *testing.T) {
	for name, incapable := range map[string][]bool{
		"think holds it":  {false, true},
		"no arm holds it": {true, true},
	} {
		fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
		fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
			return request.Selection.AvailableActionIDs[0]
		})
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
			DecisionName:    fixture.decision.Name,
			CandidateModels: fixture.decision.ModelRefs,
			RaylineARC: &selection.RaylineARCSelectionContext{
				EpisodeIDHash: strings.Repeat("e", 64),
				State:         state,
				RawRequest:    policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}),
				RequestFormat: policyFormatAnthropic,
				IncapableArms: incapable,
			},
		})
		if !slices.Contains(incapable, false) {
			var failure *raylineARCSelectionFailure
			if !errors.As(err, &failure) || failure.class != arcFailureNoCapableArm || len(fixture.fake.received()) != 0 {
				t.Fatalf("%s: error %v after %d decide calls, want no_capable_arm before any", name, err, len(fixture.fake.received()))
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		offerArms := map[int]bool{}
		for _, actionID := range offeredActions(fixture, 0) {
			for _, binding := range fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings {
				if binding.ActionID == actionID {
					offerArms[map[string]int{"think": 0, "off": 1}[binding.Worker]] = true
				}
			}
		}
		if offerArms[1] || !offerArms[0] || result.RaylineARC.SelectedArm != 0 {
			t.Fatalf("%s: offered arms %v, served arm %d", name, offerArms, result.RaylineARC.SelectedArm)
		}
	}
}
