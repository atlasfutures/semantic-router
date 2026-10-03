//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// fallbackFixture is boundaryFixture with ADR 0120's fallback switched to
// enabled, and the "off" worker's action, which its service prefers.
func fallbackFixture(t *testing.T, enabled bool) (*policySelectorFixture, *raylinearc.MemoryEpisodeStore, string, string) {
	t.Helper()
	fixture, store, episode := boundaryFixture(t)
	fixture.decision.Algorithm.RaylineARC.PolicyService.Fallback = &config.RaylineARCPolicyFallbackConfig{Enabled: enabled}
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second}),
	})
	offAction := ""
	for _, binding := range fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings {
		if binding.Worker == "off" {
			offAction = binding.ActionID
		}
	}
	return fixture, store, episode, offAction
}

// With the fallback on, a refused turn excludes its model for the rest of
// the context: the retry is offered only the other model, and the
// exclusion is committed with the turn that served, so later turns keep it.
// Control: with the fallback off, the retry is offered the refusing model
// again, and the service picks it again.
func TestRaylineARCFallbackExcludesTheRefusingModel(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		fixture, store, episode, offAction := fallbackFixture(t, enabled)
		body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
		first, decided := boundaryAttempt(t, fixture, store, episode, body)
		if decided.RaylineARC.PolicyActionID != offAction {
			t.Fatalf("enabled=%v: session start chose %s", enabled, decided.RaylineARC.PolicyActionID)
		}
		first.VSRRaylineARC, first.VSRSelectedDecision = decided.RaylineARC, fixture.decision
		declineRefusedTurn(first)
		if err := first.RaylineARCTransaction.abort(context.Background(), "test"); err != nil {
			t.Fatal(err)
		}
		if excluded := readLedgerTestState(t, store, episode).Policy.Excludes("vendor/off"); excluded != enabled {
			t.Fatalf("enabled=%v: stored exclusion=%v", enabled, excluded)
		}

		retry, retried := boundaryAttempt(t, fixture, store, episode, body)
		offered := offeredActions(fixture, 1)
		if slices.Contains(offered, offAction) == enabled || (retried.RaylineARC.SelectedArm == 1) == enabled {
			t.Fatalf("enabled=%v: retry offered %v, chose arm %d", enabled, offered, retried.RaylineARC.SelectedArm)
		}
		if !enabled {
			continue
		}
		retry.RaylineARCTransaction.markPolicyState(retried.RaylineARC.PolicyNextState, false)
		if err := retry.RaylineARCTransaction.commit(context.Background(), retry); err != nil {
			t.Fatal(err)
		}
		committed := readLedgerTestStateNoWait(store, episode)
		if committed == nil || committed.TurnIndex != 1 || !committed.Policy.Excludes("vendor/off") {
			t.Fatalf("after the served retry: %+v", committed)
		}
	}
}

// A held model that the context excluded is not held: its turn decides
// again among the rest, stores that as a boundary decision, and logs the
// re-decision with what Phase 2 learns from. Control: with the fallback
// off the held model is offered alone, excluded or not.
func TestRaylineARCFallbackRedecidesOffAnExcludedHeldModel(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		fixture, _, _, offAction := fallbackFixture(t, enabled)
		fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
			return request.Selection.AvailableActionIDs[0]
		})
		state := heldEpisode(t)
		off := 1
		state.PreviousArm = &off
		state.Policy = state.Policy.WithExclusion("vendor/off", "refusal")
		logs := captureLogs(t)
		next := policyTestRequest(t,
			map[string]any{"role": "user", "content": "fix the bug"},
			map[string]any{"role": "assistant", "content": "done"},
			map[string]any{"role": "user", "content": "now the tests"})
		result, err := fixture.selectOn(t, state, next)
		if err != nil {
			t.Fatal(err)
		}
		offered := offeredActions(fixture, 0)
		if enabled {
			if slices.Contains(offered, offAction) || len(offered) != 2 || result.RaylineARC.SelectedArm != 0 ||
				result.RaylineARC.PolicyBoundary == nil || fixture.fake.received()[0].Selection.HeldActionID != nil {
				t.Fatalf("offered %v, arm %d, boundary %+v", offered, result.RaylineARC.SelectedArm, result.RaylineARC.PolicyBoundary)
			}
			event := findLogEvent(t, logs, "rayline_arc_fallback_decision")
			raw, _ := json.Marshal(event)
			for _, want := range []string{`"reason":"fallback_redecide"`, `"model":"vendor/off"`, `"class":"refusal"`, `"turn_index":2`, `"scores":[`, `"request_id":"req-policy-test"`, `"selected_action_id":"` + offered[0] + `"`} {
				if !strings.Contains(string(raw), want) {
					t.Fatalf("the re-decision log lacks %s: %s", want, raw)
				}
			}
			continue
		}
		if len(offered) != 1 || offered[0] != offAction || result.RaylineARC.PolicyBoundary != nil {
			t.Fatalf("fallback off: offered %v, boundary %+v", offered, result.RaylineARC.PolicyBoundary)
		}
	}
}

// A refusal on the turn that compacted excludes the model in the compacted
// context the retry continues, not in the one the compaction left. Control:
// when the store moved on since the turn read it, the exclusion is recorded
// in what the store now holds.
func TestRaylineARCRefusalExcludesInTheTurnsContext(t *testing.T) {
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	stored := (&raylinearc.PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a")
	compacted := &raylinearc.PolicyEpisodeState{Epoch: 1, EpochStartTurn: 3, CompactionCount: 1}
	state, _ := raylinearc.NewEpisodeState(2)
	state.Policy = stored
	for _, moved := range []bool{false, true} {
		decidedFrom := stored
		if moved {
			longer := append(append([]json.RawMessage(nil), task...), json.RawMessage(`{"role":"assistant","content":"done"}`))
			decidedFrom = stored.Next(longer, strings.Repeat("b", 64), "arm-b")
		}
		next, changed := refusedTurn{arm: -1, exclude: "vendor/off", context: compacted, decidedFrom: decidedFrom}.apply(state)
		if !changed || !next.Policy.Excludes("vendor/off") || (next.Policy.CompactionCount == 1) == moved {
			t.Fatalf("moved=%v: stored policy %+v", moved, next.Policy)
		}
	}
}

// A late refusal keeps what the store excluded since its turn read it: it
// lands in the turn's context without lifting a newer exclusion.
func TestRaylineARCLateRefusalKeepsNewerExclusions(t *testing.T) {
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	read := (&raylinearc.PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a")
	state, _ := raylinearc.NewEpisodeState(2)
	state.Policy = read.WithExclusion("vendor/think", turnFailureRefusal)
	next, changed := refusedTurn{arm: -1, exclude: "vendor/off", context: read.Clone(), decidedFrom: read}.apply(state)
	if !changed || !next.Policy.Excludes("vendor/off") || !next.Policy.Excludes("vendor/think") {
		t.Fatalf("exclusions after the late refusal: %+v", next.Policy.Exclusions)
	}
}

// A coalesced resend refused while the leader holds the lease hands its
// refusal over; when the leader's own turn succeeds, the exclusion is
// committed with it. Control: with no refusal handed over, the commit
// excludes nothing.
func TestRaylineARCHandedOverRefusalCommitsWithTheLeader(t *testing.T) {
	for _, refusal := range []bool{true, false} {
		fixture, store, episode, _ := fallbackFixture(t, true)
		leader, decided := boundaryAttempt(t, fixture, store, episode,
			policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"}))
		entry := &raylineARCInflightEntry{done: make(chan struct{}), finished: make(chan struct{})}
		leader.RaylineARCTransaction.inflight = entry
		if refusal {
			declineRefusedTurn(&RequestContext{
				RaylineARCTransaction: newBorrowedRaylineARCEpisodeTransaction(store, leader.RaylineARCTransaction.state, episode, entry),
				VSRRaylineARC:         decided.RaylineARC,
				VSRSelectedDecision:   fixture.decision,
			})
		}
		leader.RaylineARCTransaction.markPolicyState(decided.RaylineARC.PolicyNextState, false)
		if err := leader.RaylineARCTransaction.commit(context.Background(), leader); err != nil {
			t.Fatal(err)
		}
		committed := readLedgerTestStateNoWait(store, episode)
		if committed == nil || committed.TurnIndex != 1 || committed.Policy.Excludes("vendor/off") != refusal {
			t.Fatalf("refusal=%v: committed %+v", refusal, committed)
		}
	}
}

// The loader's model bound is the episode's exclusion bound: every model a
// fallback-enabled package serves can be excluded.
func TestRaylineARCFallbackModelBoundMatchesTheEpisode(t *testing.T) {
	if config.RaylineARCMaxFallbackModels != raylinearc.MaxPolicyExclusions {
		t.Fatalf("loader bound %d, episode bound %d", config.RaylineARCMaxFallbackModels, raylinearc.MaxPolicyExclusions)
	}
}
