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
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// asTwoStage re-arms fixture's selector with a scorer for a two-stage package
// (one whose manifest carries stage_one), as newPolicyServiceScorer builds it
// from such a manifest.
func asTwoStage(fixture *policySelectorFixture) {
	scorer := newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
	scorer.twoStage = true
	scorer.attestable = true // a model-unit stage one, as 0.6.3's
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    scorer,
		admission: raylinearc.NewAdmissionGate(0),
		policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second}),
	})
}

// selectWith is selectOn with the selection context edited first.
func (fixture *policySelectorFixture) selectWith(
	t *testing.T,
	state *raylinearc.EpisodeState,
	body []byte,
	edit func(*selection.RaylineARCSelectionContext),
) (*selection.SelectionResult, error) {
	t.Helper()
	arc := &selection.RaylineARCSelectionContext{
		EpisodeIDHash: strings.Repeat("e", 64), RequestID: "req-two-stage-test",
		State: state, RawRequest: body, RequestFormat: policyFormatAnthropic,
	}
	edit(arc)
	return fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName: fixture.decision.Name, CandidateModels: fixture.decision.ModelRefs, RaylineARC: arc,
	})
}

// A typed follow-up two completed turns in: between task_turn_compaction_v1's
// boundaries, where VSR holds a single-stage package's model.
func typedFollowUp(t *testing.T) []byte {
	t.Helper()
	return policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"},
		map[string]any{"role": "assistant", "content": "done"},
		map[string]any{"role": "user", "content": "now also handle the empty input"})
}

// semantic-router#265: a two-stage package's service decides the model at its
// own stage-one boundaries, a typed user message among them, so VSR holds
// nothing on its main turns: the turn the schedule would hold is offered every
// model, with no held action, and stores its decision for a retry as a
// boundary does. Control: a single-stage package on the same turn is offered
// only the held model's levels, as before.
func TestTwoStagePackageIsOfferedEveryModelBetweenScheduleBoundaries(t *testing.T) {
	for _, twoStage := range []bool{true, false} {
		fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
		if twoStage {
			asTwoStage(fixture)
		}
		fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
			return request.Selection.AvailableActionIDs[0]
		})
		result, err := fixture.selectOn(t, heldEpisode(t), typedFollowUp(t))
		if err != nil {
			t.Fatalf("twoStage=%v: select: %v", twoStage, err)
		}
		sent := fixture.fake.received()[0]
		trace := result.RaylineARC
		if twoStage {
			if len(sent.Selection.AvailableActionIDs) != 3 || sent.Selection.HeldActionID != nil ||
				trace.OfferKind != "fresh" || trace.PolicyBoundary == nil {
				t.Fatalf("two-stage: offered %v held %v, offer_kind %q, boundary %+v; want every model, nothing held, the decision stored",
					sent.Selection.AvailableActionIDs, sent.Selection.HeldActionID, trace.OfferKind, trace.PolicyBoundary)
			}
			continue
		}
		if len(sent.Selection.AvailableActionIDs) != 2 || trace.OfferKind != "held" || trace.PolicyBoundary != nil {
			t.Fatalf("single-stage: offered %v, offer_kind %q, boundary %+v; want the held model's two levels",
				sent.Selection.AvailableActionIDs, trace.OfferKind, trace.PolicyBoundary)
		}
	}
}

// A side call keeps the held arm whatever the package: it commits nothing,
// and the service must not decide the main conversation's model on it.
func TestTwoStageSideCallStaysHeld(t *testing.T) {
	fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
	asTwoStage(fixture)
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	_, err := fixture.selectWith(t, heldEpisode(t), typedFollowUp(t), func(arc *selection.RaylineARCSelectionContext) {
		arc.PolicyCallKind, arc.PolicyCallKindSource = raylinearc.PolicyCallSide, callKindSourceHeader
	})
	if err != nil {
		t.Fatal(err)
	}
	if offered := offeredActions(fixture, 0); len(offered) != 2 {
		t.Fatalf("side call offered %v, want only the held model's levels", offered)
	}
}

// The retained-boundary rule covers every main turn of a two-stage package:
// a mid-conversation turn the schedule would have held, which fails after its
// decision, is retried on the arm it decided, not decided again.
func TestTwoStageRetryBetweenScheduleBoundariesKeepsTheDecidedArm(t *testing.T) {
	fixture, store, episode := boundaryFixture(t)
	asTwoStage(fixture)
	start := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	first, decided := boundaryAttempt(t, fixture, store, episode, start)
	first.RaylineARCTransaction.markPolicyState(decided.RaylineARC.PolicyNextState, false)
	if err := first.RaylineARCTransaction.commit(context.Background(), first); err != nil {
		t.Fatal(err)
	}

	// Turn 1 (one completed turn): held under the schedule, decided here.
	// The service now prefers the think worker.
	thinkAction := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return thinkAction })
	follow := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"},
		map[string]any{"role": "assistant", "content": "dispatched"},
		map[string]any{"role": "user", "content": "and the docs"})
	failed, mid := boundaryAttempt(t, fixture, store, episode, follow)
	if offered := offeredActions(fixture, 1); len(offered) != 3 || mid.RaylineARC.SelectedArm != 0 ||
		mid.RaylineARC.PolicyBoundary == nil {
		t.Fatalf("turn 1 offered %v, chose arm %d, boundary %+v", offered, mid.RaylineARC.SelectedArm, mid.RaylineARC.PolicyBoundary)
	}
	if err := failed.RaylineARCTransaction.abort(context.Background(), "upstream_status"); err != nil {
		t.Fatal(err)
	}

	// The retry is offered only the arm turn 1 decided, though the service
	// would now pick the other worker.
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[len(request.Selection.AvailableActionIDs)-1]
	})
	retry, reused := boundaryAttempt(t, fixture, store, episode, follow)
	if offered := offeredActions(fixture, 2); len(offered) != 2 || reused.RaylineARC.SelectedArm != 0 ||
		reused.RaylineARC.PolicyBoundary != nil {
		t.Fatalf("retry offered %v, chose arm %d, boundary %+v; want only the decided arm 0",
			offered, reused.RaylineARC.SelectedArm, reused.RaylineARC.PolicyBoundary)
	}
	_ = retry.RaylineARCTransaction.abort(context.Background(), "test")
}

// A replica without episode memory: VSR no longer holds a mid-conversation
// turn whose history the service cannot attribute, so the first ask carries
// the session record's model attested. A package that predates the field
// refuses it (stage_one_held_unknown), and the turn is asked once more
// narrowed to that model. Control: a single-stage package holds the turn
// itself and is never attested or refused.
func TestTwoStageColdMidConversationTurnTakesTheDerivedHold(t *testing.T) {
	for _, twoStage := range []bool{true, false} {
		fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
		if twoStage {
			asTwoStage(fixture)
		}
		bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
		model := map[string]string{}
		for _, binding := range bindings {
			model[binding.ActionID] = binding.Model
		}
		fixture.fake.refuseCold = func(request raylinearc.PolicyDecisionRequest) bool {
			models := map[string]bool{}
			for _, action := range request.Selection.AvailableActionIDs {
				models[model[action]] = true
			}
			return len(models) > 1 && len(request.Attribution) == 0
		}
		fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
			return request.Selection.AvailableActionIDs[0]
		})
		state, _ := raylinearc.NewEpisodeState(2)
		held := 0
		state.PreviousArm, state.TurnIndex = &held, 2
		// The episode's prefix is known but its ledger is empty: nothing in
		// the history is attributed.
		state.Policy = &raylinearc.PolicyEpisodeState{PrefixDigest: raylinearc.MessagesDigest(nil, 0)}
		result, err := fixture.selectWith(t, state, typedFollowUp(t), func(arc *selection.RaylineARCSelectionContext) {
			arc.ServedModels = []selection.RaylineARCServedModel{{Source: derivedHoldSessionRecord, Model: "vendor/off"}}
		})
		if err != nil {
			t.Fatalf("twoStage=%v: select: %v", twoStage, err)
		}
		calls := fixture.fake.received()
		if !twoStage {
			if len(calls) != 1 || len(calls[0].Selection.AvailableActionIDs) != 2 {
				t.Fatalf("single-stage: %d calls, offer %v; want one held call", len(calls), calls[0].Selection.AvailableActionIDs)
			}
			continue
		}
		// A package that predates attested_held_action_id refuses the attested
		// full offer, and the turn is then held on the derived model.
		if len(calls) != 2 || calls[0].Selection.AttestedHeldActionID == nil || len(calls[0].Selection.AvailableActionIDs) != 3 ||
			calls[1].Selection.AttestedHeldActionID != nil ||
			!slices.Equal(calls[1].Selection.AvailableActionIDs, []string{bindings[2].ActionID}) {
			t.Fatalf("two-stage: %d calls, offers %v; want the full offer attested and refused, then the derived model", len(calls), offersOf(calls))
		}
		if result.RaylineARC.PolicyActionID != bindings[2].ActionID {
			t.Fatalf("two-stage: served %s, want the derived model's action", result.RaylineARC.PolicyActionID)
		}
	}
}

// ADR 0120 on a turn VSR no longer holds: a model the context excluded stays
// out of the offer, the turn is not marked forced, and the rest are offered.
func TestTwoStageFallbackExclusionStaysOutOfAnUnheldTurn(t *testing.T) {
	fixture, store, episode, offAction := fallbackFixture(t, true)
	asTwoStage(fixture)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	first, decided := boundaryAttempt(t, fixture, store, episode, body)
	if decided.RaylineARC.PolicyActionID != offAction {
		t.Fatalf("session start chose %s", decided.RaylineARC.PolicyActionID)
	}
	first.VSRRaylineARC, first.VSRSelectedDecision = decided.RaylineARC, fixture.decision
	declineRefusedTurn(first)
	if err := first.RaylineARCTransaction.abort(context.Background(), "test"); err != nil {
		t.Fatal(err)
	}
	retry, retried := boundaryAttempt(t, fixture, store, episode, body)
	retry.RaylineARCTransaction.markPolicyState(retried.RaylineARC.PolicyNextState, false)
	if err := retry.RaylineARCTransaction.commit(context.Background(), retry); err != nil {
		t.Fatal(err)
	}

	follow := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"},
		map[string]any{"role": "assistant", "content": "dispatched"},
		map[string]any{"role": "user", "content": "and the docs"})
	next, result := boundaryAttempt(t, fixture, store, episode, follow)
	offered := offeredActions(fixture, 2)
	if len(offered) != 2 || slices.Contains(offered, offAction) || result.RaylineARC.OfferKind != "fresh" {
		t.Fatalf("turn 1 offered %v (offer_kind %q); want both think levels, the excluded model left out",
			offered, result.RaylineARC.OfferKind)
	}
	_ = next.RaylineARCTransaction.abort(context.Background(), "test")
}

func offersOf(calls []raylinearc.PolicyDecisionRequest) [][]string {
	offers := make([][]string, len(calls))
	for index, call := range calls {
		offers[index] = call.Selection.AvailableActionIDs
	}
	return offers
}

// The scorer reads two-stage from the manifest: a v5 manifest carrying
// stage_one is two-stage; the same manifest without it is not.
func TestPolicyScorerReadsTwoStageFromTheManifest(t *testing.T) {
	raw, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v5.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, stageOne := range []bool{false, true} {
		body := raw
		if stageOne {
			body = []byte(strings.Replace(string(raw), "{", `{"stage_one": {"boundaries": ["task_start", "user_message"]},`, 1))
		}
		path := filepath.Join(t.TempDir(), "package.json")
		if err := os.WriteFile(path, body, 0o600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(body)
		policy := &config.RaylineARCPolicyServiceConfig{
			PackageAlias: policyTestAlias, PackageSHA256: hex.EncodeToString(sum[:]), PackageManifest: path,
			Bindings: policyTestBindings(), ModelSchedule: config.RaylineARCModelScheduleTaskTurnCompaction,
		}
		if got := policy.PackageTwoStage(); got != stageOne {
			t.Fatalf("stage_one=%v: PackageTwoStage() = %v", stageOne, got)
		}
		decision := policyTestDecision(config.RaylineARCModelScheduleTaskTurnCompaction)
		decision.Algorithm.RaylineARC.PolicyService = policy
		if scorer := newPolicyServiceScorer(&config.RouterConfig{}, decision); scorer.twoStage != stageOne {
			t.Fatalf("stage_one=%v: scorer.twoStage = %v", stageOne, scorer.twoStage)
		}
	}
}

// A package that reads attested_held_action_id (pathfinder c0dce29a) decides a
// cold typed user turn itself: the first ask keeps every action offered with
// the session record's model attested, and the service's choice is served --
// here another model -- rather than VSR narrowing the offer (memex-desktop
// #7417). One call: nothing is refused first.
func TestTwoStageColdTypedTurnIsDecidedWithTheServedModelAttested(t *testing.T) {
	fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
	asTwoStage(fixture)
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.refuseCold = func(request raylinearc.PolicyDecisionRequest) bool {
		return request.Selection.AttestedHeldActionID == nil && len(request.Attribution) == 0
	}
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0] // a user-message decision that switches
	})
	state, _ := raylinearc.NewEpisodeState(2)
	held := 0
	state.PreviousArm, state.TurnIndex = &held, 2
	state.Policy = &raylinearc.PolicyEpisodeState{PrefixDigest: raylinearc.MessagesDigest(nil, 0)}
	result, err := fixture.selectWith(t, state, typedFollowUp(t), func(arc *selection.RaylineARCSelectionContext) {
		arc.ServedModels = []selection.RaylineARCServedModel{{Source: derivedHoldSessionRecord, Model: "vendor/off"}}
	})
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	calls := fixture.fake.received()
	if len(calls) != 1 || len(calls[0].Selection.AvailableActionIDs) != 3 ||
		calls[0].Selection.AttestedHeldActionID == nil || *calls[0].Selection.AttestedHeldActionID != bindings[2].ActionID {
		t.Fatalf("%d calls, offers %v; want one ask, the full offer with the served model attested", len(calls), offersOf(calls))
	}
	if result.RaylineARC.PolicyActionID != calls[0].Selection.AvailableActionIDs[0] {
		t.Fatalf("served %s, want the service's decision", result.RaylineARC.PolicyActionID)
	}
}

// Any refusal of the attested first ask -- here invalid_request from a service
// that rejects the field -- falls through to the narrowed re-ask: the
// attested ask never serves worse than the old hold (SwiftWolf's #272 review).
func TestTwoStageAttestedAskRefusedForAnyReasonFallsBackToTheNarrowedHold(t *testing.T) {
	fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
	asTwoStage(fixture)
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	model := map[string]string{}
	for _, binding := range bindings {
		model[binding.ActionID] = binding.Model
	}
	fixture.fake.rejectAttested = true
	fixture.fake.refuseCold = func(request raylinearc.PolicyDecisionRequest) bool {
		models := map[string]bool{}
		for _, action := range request.Selection.AvailableActionIDs {
			models[model[action]] = true
		}
		return len(models) > 1 && len(request.Attribution) == 0
	}
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	state, _ := raylinearc.NewEpisodeState(2)
	held := 0
	state.PreviousArm, state.TurnIndex = &held, 2
	state.Policy = &raylinearc.PolicyEpisodeState{PrefixDigest: raylinearc.MessagesDigest(nil, 0)}
	result, err := fixture.selectWith(t, state, typedFollowUp(t), func(arc *selection.RaylineARCSelectionContext) {
		arc.ServedModels = []selection.RaylineARCServedModel{{Source: derivedHoldSessionRecord, Model: "vendor/off"}}
	})
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	calls := fixture.fake.received()
	if len(calls) != 2 || calls[0].Selection.AttestedHeldActionID == nil || calls[1].Selection.AttestedHeldActionID != nil ||
		!slices.Equal(calls[1].Selection.AvailableActionIDs, []string{bindings[2].ActionID}) {
		t.Fatalf("%d calls, offers %v; want attested (rejected), then narrowed to the derived model", len(calls), offersOf(calls))
	}
	if result.RaylineARC.PolicyActionID != bindings[2].ActionID {
		t.Fatalf("served %s, want the derived model's action", result.RaylineARC.PolicyActionID)
	}
}

// An attested first ask the service never answered -- the connection lost
// mid-call -- is the turn's failure as it stands: it is not asked again
// narrowed, which on an outage would pay a second timeout for nothing and
// could force the recorded model after a transient network error.
func TestTwoStageAttestedFirstAskLostInTransportIsNotReaskedNarrowed(t *testing.T) {
	fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
	asTwoStage(fixture)
	fixture.fake.dropAttested = true
	state, _ := raylinearc.NewEpisodeState(2)
	held := 0
	state.PreviousArm, state.TurnIndex = &held, 2
	state.Policy = &raylinearc.PolicyEpisodeState{PrefixDigest: raylinearc.MessagesDigest(nil, 0)}
	_, err := fixture.selectWith(t, state, typedFollowUp(t), func(arc *selection.RaylineARCSelectionContext) {
		arc.ServedModels = []selection.RaylineARCServedModel{{Source: derivedHoldSessionRecord, Model: "vendor/off"}}
	})
	if class := selectionFailureClassOf(t, err); class != "policy_service_transport" {
		t.Fatalf("class %s, want policy_service_transport", class)
	}
	calls := fixture.fake.received()
	if len(calls) != 1 || calls[0].Selection.AttestedHeldActionID == nil {
		t.Fatalf("%d calls, offers %v; want the one attested ask and no narrowed re-ask", len(calls), offersOf(calls))
	}
}
