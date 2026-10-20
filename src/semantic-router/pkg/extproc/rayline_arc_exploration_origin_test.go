package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// relaxedExplorationFixture is an exploration cell whose episodes are relaxed.
func relaxedExplorationFixture(t *testing.T) *policySelectorFixture {
	t.Helper()
	fixture := newExplorationFixture(t, fixtureExploration(), nil)
	fixture.decision.Algorithm.RaylineARC.Episode.Consistency = config.RaylineARCConsistencyRelaxed
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy: raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
			BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
		}),
	})
	return fixture
}

// answerExploration sets how the fake answers the next decide's exploration.
func (fixture *policySelectorFixture) answerExploration(held, omit bool, chosen raylinearc.PolicyStageOneCandidate) {
	fixture.fake.mu.Lock()
	defer fixture.fake.mu.Unlock()
	fixture.fake.explorationHeld, fixture.fake.omitExploration, fixture.fake.explorationChosen = held, omit, chosen
}

// explorationTurn decides one main turn on state and returns its trace and
// the state the turn commits.
func explorationTurn(
	t *testing.T,
	fixture *policySelectorFixture,
	state *raylinearc.EpisodeState,
	headers map[string]string,
	messages ...map[string]any,
) (*selection.RaylineARCTrace, *raylinearc.EpisodeState) {
	t.Helper()
	result, err := fixture.selectWithHeaders(t, state, policyTestRequest(t, messages...), headers)
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	committed := cloneARCState(state)
	committed.Policy = result.RaylineARC.PolicyNextState.Clone()
	return result.RaylineARC, committed
}

var (
	explorationTask     = map[string]any{"role": "user", "content": "go"}
	explorationReply    = map[string]any{"role": "assistant", "content": "done"}
	explorationFollowUp = map[string]any{"role": "user", "content": "and the tests"}
)

// boundaryKey is the boundary turn's own decision key.
func boundaryKey(t *testing.T, trace *selection.RaylineARCTrace) string {
	t.Helper()
	if trace.PolicyExploration == nil || trace.PolicyExploration.DecisionKey == nil {
		t.Fatalf("boundary turn has no decision key: %+v", trace.PolicyExploration)
	}
	return *trace.PolicyExploration.DecisionKey
}

func otherCandidate() raylinearc.PolicyStageOneCandidate {
	model, effort := "claude-opus-5", "high"
	return raylinearc.PolicyStageOneCandidate{"model": &model, "effort": &effort}
}

// A relaxed held turn, whose record states no origin, joins the latest
// boundary through VSR's own record, but only while the held model is the
// one that boundary chose; the record is kept for the next held turn.
func TestRelaxedHeldTurnFillsTheLatestBoundarysKey(t *testing.T) {
	fixture := relaxedExplorationFixture(t)
	state, _ := raylinearc.NewEpisodeState(2)
	boundary, afterBoundary := explorationTurn(t, fixture, state, eligibleHeaders(), explorationTask)
	key := boundaryKey(t, boundary)
	origin := afterBoundary.Policy.Exploration
	if origin == nil || origin.DecisionKey != key || *origin.Chosen["model"] != "glm-5.3-flash" || origin.Chosen["effort"] != nil {
		t.Fatalf("boundary stored %+v, want key %s and the chosen candidate", origin, key)
	}
	if line := usageLine(t, boundary); line["exploration_decision_key"] != key {
		t.Fatalf("boundary llm_usage key %v", line["exploration_decision_key"])
	}

	fixture.answerExploration(true, false, nil)
	held, afterHeld := explorationTurn(t, fixture, afterBoundary, eligibleHeaders(), explorationTask, explorationReply, explorationFollowUp)
	if held.PolicyExploration == nil || !held.PolicyExploration.Held || held.PolicyExploration.OriginDecisionKey != nil {
		t.Fatalf("relaxed held record %+v, want held with no origin", held.PolicyExploration)
	}
	if line := usageLine(t, held); line["exploration_decision_key"] != key || line["router_propensity"] != 1.0 {
		t.Fatalf("held llm_usage key %v propensity %v, want %s", line["exploration_decision_key"], line["router_propensity"], key)
	}
	if kept := afterHeld.Policy.Exploration; kept == nil || kept.DecisionKey != key {
		t.Fatalf("held turn left %+v, want the boundary's record kept", kept)
	}
}

// Every other relaxed held turn keeps the key null, never an older
// boundary's: one holding a model the boundary did not choose, one after a
// later boundary that carried no exploration record (exploration off for it,
// or the turn excluded), and one in an episode with no recorded boundary.
func TestRelaxedHeldTurnKeepsTheKeyNullWithoutTheLatestBoundary(t *testing.T) {
	fixture := relaxedExplorationFixture(t)
	fresh, _ := raylinearc.NewEpisodeState(2)
	_, afterBoundary := explorationTurn(t, fixture, fresh, eligibleHeaders(), explorationTask)
	if afterBoundary.Policy.Exploration == nil {
		t.Fatal("the boundary stored no record")
	}
	conversation := []map[string]any{explorationTask, explorationReply, explorationFollowUp}

	t.Run("different held model", func(t *testing.T) {
		fixture.answerExploration(true, false, otherCandidate())
		held, after := explorationTurn(t, fixture, afterBoundary, eligibleHeaders(), conversation...)
		assertHeldKeyNull(t, held)
		if after.Policy.Exploration != nil {
			t.Fatalf("a held turn on another model kept %+v", after.Policy.Exploration)
		}
	})

	privacy := eligibleHeaders()
	privacy[raylineARCPrivacyModeHeader] = "true"
	for name, later := range map[string]struct {
		omit    bool
		headers map[string]string
	}{
		"later boundary without a record": {omit: true, headers: eligibleHeaders()},
		"later turn excluded":             {headers: privacy},
	} {
		t.Run(name, func(t *testing.T) {
			fixture.answerExploration(false, later.omit, nil)
			recordless, afterLater := explorationTurn(t, fixture, afterBoundary, later.headers, conversation...)
			if recordless.PolicyExploration != nil {
				t.Fatalf("the later turn carried a record %+v", recordless.PolicyExploration)
			}
			if afterLater.Policy.Exploration != nil {
				t.Fatalf("a turn without a record kept %+v", afterLater.Policy.Exploration)
			}
			fixture.answerExploration(true, false, nil)
			next := append(append([]map[string]any(nil), conversation...), explorationReply, explorationFollowUp)
			held, _ := explorationTurn(t, fixture, afterLater, eligibleHeaders(), next...)
			assertHeldKeyNull(t, held)
		})
	}

	t.Run("no recorded boundary", func(t *testing.T) {
		fixture.answerExploration(true, false, nil)
		empty, _ := raylinearc.NewEpisodeState(2)
		held, after := explorationTurn(t, fixture, empty, eligibleHeaders(), explorationTask)
		assertHeldKeyNull(t, held)
		if after.Policy.Exploration != nil {
			t.Fatalf("a held turn with no boundary stored %+v", after.Policy.Exploration)
		}
	})
}

func assertHeldKeyNull(t *testing.T, trace *selection.RaylineARCTrace) {
	t.Helper()
	if trace.PolicyExploration == nil || !trace.PolicyExploration.Held {
		t.Fatalf("not a held record: %+v", trace.PolicyExploration)
	}
	if trace.PolicyExplorationHeldKey != nil {
		t.Fatalf("filled key %q", *trace.PolicyExplorationHeldKey)
	}
	if line := usageLine(t, trace); line["exploration_decision_key"] != nil {
		t.Fatalf("llm_usage key %v, want null", line["exploration_decision_key"])
	}
}

// A strict call's origin is the service's: a strict episode keeps no record
// of its own, and its held turn joins through origin_decision_key.
func TestStrictEpisodeKeepsNoExplorationRecord(t *testing.T) {
	fixture := newExplorationFixture(t, fixtureExploration(), nil)
	state, _ := raylinearc.NewEpisodeState(2)
	_, afterBoundary := explorationTurn(t, fixture, state, eligibleHeaders(), explorationTask)
	if afterBoundary.Policy.Exploration != nil {
		t.Fatalf("a strict boundary stored %+v", afterBoundary.Policy.Exploration)
	}
	fixture.answerExploration(true, false, nil)
	held, _ := explorationTurn(t, fixture, afterBoundary, eligibleHeaders(), explorationTask, explorationReply, explorationFollowUp)
	origin := held.PolicyExploration.OriginDecisionKey
	if origin == nil || held.PolicyExplorationHeldKey != nil || usageLine(t, held)["exploration_decision_key"] != *origin {
		t.Fatalf("strict held turn: origin %v filled %v", origin, held.PolicyExplorationHeldKey)
	}
}

// A later boundary decided without an exploration record retires the older
// origin even when its turn never commits: its dispatch failed, or the
// provider refused it (with or without the fallback's exclusion). The next
// held turn on the older boundary's candidate keeps the key null.
func TestRecordlessBoundaryThatFailsStillRetiresTheOlderKey(t *testing.T) {
	for name, outcome := range map[string]struct{ refused, fallback bool }{
		"upstream failure":          {},
		"provider refusal":          {refused: true},
		"refusal with an exclusion": {refused: true, fallback: true},
	} {
		t.Run(name, func(t *testing.T) {
			fixture := relaxedExplorationFixture(t)
			fixture.decision.Algorithm.RaylineARC.PolicyService.ModelSchedule = config.RaylineARCModelScheduleTaskTurnCompaction
			if outcome.fallback {
				fixture.decision.Algorithm.RaylineARC.PolicyService.Fallback = &config.RaylineARCPolicyFallbackConfig{Enabled: true}
			}
			asTwoStage(fixture)
			// The refused model is excluded, so the fake serves what is offered.
			fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
				return request.Selection.AvailableActionIDs[0]
			})
			store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
			if err != nil {
				t.Fatal(err)
			}
			episode := "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
			attempt := func(messages ...map[string]any) (*RequestContext, *selection.RaylineARCTrace) {
				t.Helper()
				state, read, err := store.Snapshot(context.Background(), episode, 2)
				if err != nil {
					t.Fatal(err)
				}
				result, err := fixture.selectWithHeaders(t, state, policyTestRequest(t, messages...), eligibleHeaders())
				if err != nil {
					t.Fatal(err)
				}
				ctx := &RequestContext{
					TraceContext: context.Background(), VSRRaylineARC: result.RaylineARC, VSRSelectedDecision: fixture.decision,
					RaylineARCTransaction: newRelaxedRaylineARCEpisodeTransaction(store, state, read, episode, false),
				}
				ctx.RaylineARCTransaction.markSelection(result.RaylineARC.SelectedArm, 10)
				stageRaylineARCPolicySelection(ctx, result.RaylineARC, "")
				return ctx, result.RaylineARC
			}
			first, boundary := attempt(explorationTask)
			boundaryKey(t, boundary)
			if err := first.RaylineARCTransaction.commit(context.Background(), first); err != nil {
				t.Fatal(err)
			}
			toolUse := map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "tool_use", "id": "t1", "name": "test", "input": map[string]any{}}}}
			fixture.answerExploration(false, true, nil)
			failed, later := attempt(explorationTask, toolUse, map[string]any{"role": "user", "content": "new task"})
			if later.PolicyBoundary == nil || later.PolicyExploration != nil {
				t.Fatal("not a recordless later boundary")
			}
			if outcome.refused {
				declineRefusedTurn(failed)
			}
			if err := failed.RaylineARCTransaction.abort(context.Background(), "upstream_status"); err != nil {
				t.Fatal(err)
			}
			fixture.answerExploration(true, false, nil)
			toolResult := map[string]any{"role": "user", "content": []any{map[string]any{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}}}
			_, held := attempt(explorationTask, toolUse, toolResult)
			if held.OfferKind == "retained" {
				t.Fatal("the held turn reused the failed boundary")
			}
			assertHeldKeyNull(t, held)
		})
	}
}
