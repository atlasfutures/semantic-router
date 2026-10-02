package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const (
	subagentSystem = `[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.280; cc_entrypoint=sdk-cli; cc_is_subagent=true;"},{"type":"text","text":"You are a search agent."}]`
	titleSystem    = `[{"type":"text","text":"Generate a concise, sentence-case title (3-7 words) that captures the main topic or goal of this coding session. The title should be clear enough that the user recognizes the session in a list. Use sentence case: capitalize only the first word and proper nouns.\n\nReturn JSON with a single \"title\" field."}]`
)

// selectSideCall runs one selection of a request whose body Claude Code sent
// with the given system blocks, on an episode holding state.
func (fixture *policySelectorFixture) selectSideCall(
	t *testing.T,
	system string,
	state *raylinearc.EpisodeState,
) (*selection.SelectionResult, error) {
	t.Helper()
	body := `{"model":"auto","max_tokens":1024,"system":` + system +
		`,"messages":[{"role":"user","content":"find the config loader"}]}`
	if !json.Valid([]byte(body)) {
		t.Fatal("invalid test body")
	}
	return fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64),
			State:         state,
			RawRequest:    []byte(body),
			RequestFormat: policyFormatAnthropic,
		},
	})
}

// heldEpisode is an episode whose main conversation committed one turn on
// action.
func subagentHeldEpisode(t *testing.T, action string, arm int) *raylinearc.EpisodeState {
	t.Helper()
	state, _ := raylinearc.NewEpisodeState(2)
	state.TurnIndex = 3
	state.PreviousArm = &arm
	state.Policy = &raylinearc.PolicyEpisodeState{
		PrefixLen: 2, PrefixDigest: strings.Repeat("a", 64),
		Ledger: []raylinearc.PolicyLedgerEntry{{Message: 1, ActionID: action, ArmID: "arm-x"}},
	}
	return state
}

// A Claude Code subagent call runs on the parent's held action, the model
// and its control, with no decide and nothing for the episode to commit.
func TestSubagentCallDispatchesTheParentsHeldAction(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "task_turn_compaction_v1")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	held := bindings[1]
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	state := subagentHeldEpisode(t, held.ActionID, 0)
	ledgerBefore := len(state.Policy.Ledger)

	result, err := fixture.selectSideCall(t, subagentSystem, state)
	if err != nil {
		t.Fatalf("subagent call: %v", err)
	}
	if got := result.RaylineARC.PolicyActionID; got != held.ActionID {
		t.Fatalf("subagent dispatched action %s, want the parent's %s", got, held.ActionID)
	}
	if result.SelectedModel != held.Worker || result.RaylineARC.ThinkingLevel != held.Level {
		t.Fatalf("subagent got model %q level %q, want %q %q", result.SelectedModel, result.RaylineARC.ThinkingLevel, held.Worker, held.Level)
	}
	if calls := len(fixture.fake.received()); calls != 0 {
		t.Fatalf("subagent call sent %d decide calls, want 0", calls)
	}
	if result.RaylineARC.PolicyNextState != nil || result.RaylineARC.PolicyBoundary != nil ||
		!result.RaylineARC.PolicySideCall || len(state.Policy.Ledger) != ledgerBefore {
		t.Fatal("a subagent call left something for the episode to commit")
	}
}

// While a boundary turn is in flight, the subagent holds the action that
// boundary decided, not the previous one.
func TestSubagentCallHoldsTheInFlightBoundaryAction(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "task_turn_compaction_v1")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	state := subagentHeldEpisode(t, bindings[0].ActionID, 0)
	state.PolicyBoundary = &raylinearc.PolicyBoundaryDecision{Arm: 0, ActionID: bindings[2].ActionID, TurnIndex: state.TurnIndex}

	result, err := fixture.selectSideCall(t, subagentSystem, state)
	if err != nil {
		t.Fatalf("subagent call: %v", err)
	}
	if got := result.RaylineARC.PolicyActionID; got != bindings[2].ActionID {
		t.Fatalf("subagent dispatched %s, want the boundary's %s", got, bindings[2].ActionID)
	}
	if calls := len(fixture.fake.received()); calls != 0 {
		t.Fatalf("subagent call sent %d decide calls, want 0", calls)
	}
}

// With no held action (nothing committed in this context yet) the subagent
// is decided as any side call is; a title call still decides as before.
func TestOtherSideCallsStillDecide(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "task_turn_compaction_v1")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })

	fresh, _ := raylinearc.NewEpisodeState(2)
	if _, err := fixture.selectSideCall(t, subagentSystem, fresh); err != nil {
		t.Fatalf("subagent call without a held action: %v", err)
	}
	if calls := len(fixture.fake.received()); calls != 1 {
		t.Fatalf("subagent with nothing held sent %d decide calls, want 1", calls)
	}
	if _, err := fixture.selectSideCall(t, titleSystem, subagentHeldEpisode(t, bindings[1].ActionID, 0)); err != nil {
		t.Fatalf("title call: %v", err)
	}
	if calls := len(fixture.fake.received()); calls != 2 {
		t.Fatalf("title call sent %d decide calls in all, want 2", calls)
	}
}
