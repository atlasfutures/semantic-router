/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// servedTurn serves one turn on the fixture's fake and commits it to state, as
// the transaction does after a delivered turn.
func servedTurn(t *testing.T, fixture *policySelectorFixture, state *raylinearc.EpisodeState, actionID string) {
	t.Helper()
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	result, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}
	arm := result.RaylineARC.SelectedArm
	state.PreviousArm = &arm
	state.Policy = result.RaylineARC.PolicyNextState
	state.TurnIndex++
}

func nextTurn(t *testing.T) []byte {
	t.Helper()
	return policyTestRequest(t, map[string]any{"role": "user", "content": "go"},
		map[string]any{"role": "assistant", "content": "ok"}, map[string]any{"role": "user", "content": "more"})
}

// pathfinder#3329: a context the policy service's encoder cannot hold keeps
// the episode's last served action instead of failing the turn, and says so.
func TestCapacityRefusalHoldsTheEpisodesServedAction(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	state, _ := raylinearc.NewEpisodeState(2)
	servedTurn(t, fixture, state, bindings[1].ActionID)
	fixture.fake.mu.Lock()
	fixture.fake.failWith = "context_exceeds_encoder_capacity"
	fixture.fake.failDetail = map[string]any{"token_count": 300000, "max_tokens": 262144, "note": "dropped"}
	fixture.fake.mu.Unlock()
	result, err := fixture.selectOn(t, state, nextTurn(t))
	if err != nil {
		t.Fatalf("capacity refusal with a held action failed the turn: %v", err)
	}
	if result.RaylineARC.PolicyActionID != bindings[1].ActionID || !strings.Contains(result.Reasoning, "encoder_capacity_hold") {
		t.Fatalf("served %s (%s), want the held %s", result.RaylineARC.PolicyActionID, result.Reasoning, bindings[1].ActionID)
	}
	if next := result.RaylineARC.PolicyNextState; next == nil || next.Ledger[len(next.Ledger)-1].ActionID != bindings[1].ActionID {
		t.Fatalf("the held turn did not commit to the ledger: %+v", next)
	}
}

// With nothing served yet the refusal still fails the turn closed: the
// package fallback is not known to this router for v4 packages.
func TestCapacityRefusalWithNothingHeldFailsClosed(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	fixture.fake.failNext("context_exceeds_encoder_capacity")
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selectOn(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "policy_service_context_exceeds_encoder_capacity" {
		t.Fatalf("first call: error = %v, want a fail-closed capacity class", err)
	}

}
