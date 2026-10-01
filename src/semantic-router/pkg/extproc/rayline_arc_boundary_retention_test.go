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
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// boundaryFixture is a scheduled policy-service selector over an in-memory
// episode store, whose service picks the "off" worker's action whenever it
// is offered and the first offered action otherwise.
func boundaryFixture(t *testing.T) (*policySelectorFixture, *raylinearc.MemoryEpisodeStore, string) {
	t.Helper()
	fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
	offAction := ""
	for _, binding := range fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings {
		if binding.Worker == "off" {
			offAction = binding.ActionID
		}
	}
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		for _, action := range request.Selection.AvailableActionIDs {
			if action == offAction {
				return action
			}
		}
		return request.Selection.AvailableActionIDs[0]
	})
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	return fixture, store, strings.Repeat("e", 64)
}

// boundaryAttempt prepares the strict episode, decides, and stages the
// decision as the request path does before dispatch.
func boundaryAttempt(
	t *testing.T,
	fixture *policySelectorFixture,
	store *raylinearc.MemoryEpisodeStore,
	episode string,
	body []byte,
) (*RequestContext, *selection.SelectionResult) {
	t.Helper()
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	result, err := fixture.selectOn(t, state, body)
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		TraceContext:          context.Background(),
		RaylineARCTransaction: newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil),
	}
	ctx.RaylineARCTransaction.markSelection(result.RaylineARC.SelectedArm, 10)
	stageRaylineARCPolicySelection(ctx, result.RaylineARC)
	return ctx, result
}

// A boundary request that fails after its decision is stored is retried on
// the arm that boundary decided, not decided again; the successful retry's
// commit clears the stored decision.
func TestRaylineARCRetryAtABoundaryKeepsTheDecidedArm(t *testing.T) {
	fixture, store, episode := boundaryFixture(t)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})

	first, decided := boundaryAttempt(t, fixture, store, episode, body)
	if decided.RaylineARC.PolicyBoundary == nil || decided.RaylineARC.SelectedArm != 1 {
		t.Fatalf("session start: boundary %+v arm %d", decided.RaylineARC.PolicyBoundary, decided.RaylineARC.SelectedArm)
	}
	// The upstream answered 5xx: the turn commits nothing.
	if err := first.RaylineARCTransaction.abort(context.Background(), "upstream_status"); err != nil {
		t.Fatal(err)
	}
	// The service would now pick another worker if it were offered one.
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	retry, reused := boundaryAttempt(t, fixture, store, episode, body)
	if offered := offeredActions(fixture, 1); len(offered) != 1 || reused.RaylineARC.SelectedArm != 1 {
		t.Fatalf("retry offered %v and chose arm %d; want only the decided arm 1", offered, reused.RaylineARC.SelectedArm)
	}
	if reused.RaylineARC.PolicyBoundary != nil {
		t.Fatal("a retry that reused the boundary decision stored a new one")
	}
	retry.RaylineARCTransaction.markPolicyState(reused.RaylineARC.PolicyNextState, false)
	if err := retry.RaylineARCTransaction.commit(context.Background(), retry); err != nil {
		t.Fatal(err)
	}
	after := readLedgerTestStateNoWait(store, episode)
	if after == nil || after.TurnIndex != 1 || after.PolicyBoundary != nil {
		t.Fatalf("after the committed retry: %+v", after)
	}
}

// A different request at the same boundary is not a retry: it decides
// afresh and replaces the stored decision.
func TestRaylineARCBoundaryDecisionExpiresOnAPrefixChange(t *testing.T) {
	fixture, store, episode := boundaryFixture(t)
	first, _ := boundaryAttempt(t, fixture, store, episode,
		policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
	if err := first.RaylineARCTransaction.abort(context.Background(), "upstream_status"); err != nil {
		t.Fatal(err)
	}
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	other, decided := boundaryAttempt(t, fixture, store, episode,
		policyTestRequest(t, map[string]any{"role": "user", "content": "a different task"}))
	if offered := offeredActions(fixture, 1); len(offered) != 3 || decided.RaylineARC.SelectedArm != 0 {
		t.Fatalf("a new prefix offered %v and chose arm %d; want a fresh decision", offered, decided.RaylineARC.SelectedArm)
	}
	if boundary := decided.RaylineARC.PolicyBoundary; boundary == nil || boundary.Arm != 0 {
		t.Fatalf("the fresh decision was not stored: %+v", boundary)
	}
	_ = other.RaylineARCTransaction.abort(context.Background(), "test")
}

// Between boundaries nothing is stored: the held arm already covers a retry.
func TestRaylineARCNoBoundaryDecisionBetweenBoundaries(t *testing.T) {
	fixture, _, _ := boundaryFixture(t)
	state := heldEpisode(t)
	task := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"},
		map[string]any{"role": "assistant", "content": "done"}, map[string]any{"role": "user", "content": "next"})
	result, err := fixture.selectOn(t, state, task)
	if err != nil {
		t.Fatal(err)
	}
	if result.RaylineARC.PolicyBoundary != nil {
		t.Fatalf("a held turn stored a boundary decision: %+v", result.RaylineARC.PolicyBoundary)
	}
}

// On a relaxed episode the decision is stored with a conditional commit and
// the turn's own commit still lands on top of it; a failed turn leaves it
// for the retry.
func TestRaylineARCRelaxedBoundaryDecisionIsStoredAndCommitted(t *testing.T) {
	fixture, store, episode := boundaryFixture(t)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	attempt := func() (*RequestContext, *selection.SelectionResult) {
		state, read, err := store.Snapshot(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		result, err := fixture.selectOn(t, state, body)
		if err != nil {
			t.Fatal(err)
		}
		ctx := &RequestContext{
			TraceContext:          context.Background(),
			RaylineARCTransaction: newRelaxedRaylineARCEpisodeTransaction(store, state, read, episode, false),
		}
		ctx.RaylineARCTransaction.markSelection(result.RaylineARC.SelectedArm, 10)
		stageRaylineARCPolicySelection(ctx, result.RaylineARC)
		return ctx, result
	}
	// The first attempt fails after staging: nothing else is written.
	attempt()
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	retry, reused := attempt()
	if offered := offeredActions(fixture, 1); len(offered) != 1 || reused.RaylineARC.SelectedArm != 1 {
		t.Fatalf("relaxed retry offered %v and chose arm %d; want only the decided arm 1", offered, reused.RaylineARC.SelectedArm)
	}
	retry.RaylineARCTransaction.markPolicyState(reused.RaylineARC.PolicyNextState, false)
	if err := retry.RaylineARCTransaction.commit(context.Background(), retry); err != nil {
		t.Fatal(err)
	}
	state, _, err := store.Snapshot(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	if state.TurnIndex != 1 || state.PolicyBoundary != nil {
		t.Fatalf("relaxed retry did not commit over the stored decision: turn %d boundary %+v", state.TurnIndex, state.PolicyBoundary)
	}

	// A turn that stages a boundary decision and succeeds commits too: its
	// read was moved to the staged version.
	fresh, freshStore, freshEpisode := boundaryFixture(t)
	fixture, store, episode = fresh, freshStore, freshEpisode
	turn, decided := attempt()
	if decided.RaylineARC.PolicyBoundary == nil {
		t.Fatal("session start stored no boundary decision")
	}
	turn.RaylineARCTransaction.markPolicyState(decided.RaylineARC.PolicyNextState, false)
	if err := turn.RaylineARCTransaction.commit(context.Background(), turn); err != nil {
		t.Fatal(err)
	}
	if state, _, _ := store.Snapshot(context.Background(), episode, 2); state.TurnIndex != 1 || state.PolicyBoundary != nil {
		t.Fatalf("the staging turn's own commit was dropped: turn %d boundary %+v", state.TurnIndex, state.PolicyBoundary)
	}
}
