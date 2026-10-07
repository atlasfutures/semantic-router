package extproc

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// The request's token count has to reach the selection context, or the mask
// is built from zero and excludes nothing. This is the one wiring point
// between the routed path's estimate and the selector.
func TestRaylineARCSelectionContextCarriesTheContextMask(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"arm-0": {ContextWindowSize: 1048576},
				"arm-1": {ContextWindowSize: 262144},
			},
		},
	}}
	algorithm := &config.AlgorithmConfig{
		Type: config.RaylineARCAlgorithmType,
		RaylineARC: &config.RaylineARCAlgorithmConfig{
			Episode: config.RaylineARCEpisodeConfig{IDHeader: "x-rayline-session"},
		},
	}
	requestContext := &RequestContext{
		Headers:              map[string]string{},
		VSRContextTokenCount: 300000,
		SemanticRequest:      testNeutralRequest("auto", "a long prompt"),
	}
	selectionContext := router.buildRaylineARCSelectionContext(
		algorithm, requestContext, []config.ModelRef{{Model: "arm-0"}, {Model: "arm-1"}}, raylineARCEpisodeEphemeral,
	)
	if selectionContext.PreparationFailure != "" {
		t.Fatalf("preparation failed: %s", selectionContext.PreparationFailure)
	}
	if selectionContext.ContextTokens != 300000 || !reflect.DeepEqual(selectionContext.OverContextArms, []bool{false, true}) {
		t.Fatalf("context = %d over %v, want 300000 and the 262144 arm marked", selectionContext.ContextTokens, selectionContext.OverContextArms)
	}
}

// The route lookup counts the consult body the way the routed path counts a
// request, so a lookup masks the arms a routed turn of the same body would.
func TestDecisionOnlyRequestContextCountsTheConsultBody(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	service := &raylineARCDecisionService{router: router}
	algorithm := decisionServiceConfig().Decisions[0].Algorithm
	long := strings.Repeat("the quick brown fox jumps over the lazy dog ", 2000)
	requestContext, err := service.decisionOnlyRequestContext(
		context.Background(), algorithm,
		routerruntime.RouteDecisionRequest{Body: anthropicBody(long), DecisionID: "rt_0a1b", SessionID: "session-1"},
	)
	if err != nil {
		t.Fatalf("decisionOnlyRequestContext() error = %v", err)
	}
	// The routed path's number for the same text, by the same rule.
	snapshot := extractSemanticRequestSignals(requestContext.SemanticRequest)
	input := router.prepareSignalEvaluationInput(signalConversationHistoryFromSnapshot(snapshot))
	routed := &RequestContext{Headers: map[string]string{}}
	applyRequestContextEstimate(snapshot, routed)
	ensureContextTokenCount(routed, input)
	if requestContext.VSRContextTokenCount <= 0 || requestContext.VSRContextTokenCount != routed.VSRContextTokenCount {
		t.Fatalf("consult count = %d, routed count = %d, want the same positive number", requestContext.VSRContextTokenCount, routed.VSRContextTokenCount)
	}
}

// A mask that does not line up with the arm list is a programming error, not
// a routing decision, and fails the turn with its own class.
func TestRaylineARCSelectorRefusesAMisalignedContextMask(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	selector := armedVisionSelector(visionScorer(), visionEncoder())
	_, err = selector.Select(context.Background(), contextSelectionContext(state, []bool{true}))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "context_arm_mapping" {
		t.Fatalf("Select() error = %v, want class context_arm_mapping", err)
	}
}

// Eval refuses what the router would refuse: a turn no arm holds is
// unavailable, a turn some arm holds is execution_required.
func TestEvalRefusesAnARCTurnNoArmHolds(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
		"glm": {ContextWindowSize: 1048576}, "qwen": {ContextWindowSize: 262144},
	}}}}
	arc := &config.Decision{
		Name:      "rayline-arc",
		ModelRefs: []config.ModelRef{{Model: "glm"}, {Model: "qwen"}},
		Algorithm: &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType, OnError: "fail_closed", RaylineARC: &config.RaylineARCAlgorithmConfig{}},
	}
	if result := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: arc, ContextTokenCount: 1100000}); result.Status != services.EvalSelectionUnavailable {
		t.Fatalf("eval status = %q, want %s when no arm holds the turn", result.Status, services.EvalSelectionUnavailable)
	}
	if result := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: arc, ContextTokenCount: 300000}); result.Status != services.EvalSelectionExecutionRequired {
		t.Fatalf("eval status = %q, want %s when one arm holds the turn", result.Status, services.EvalSelectionExecutionRequired)
	}
}

func (fixture *policySelectorFixture) selectOnWith(
	t *testing.T,
	state *raylinearc.EpisodeState,
	body []byte,
	shape func(*selection.RaylineARCSelectionContext),
) (*selection.SelectionResult, error) {
	t.Helper()
	arcContext := &selection.RaylineARCSelectionContext{
		EpisodeIDHash: strings.Repeat("e", 64),
		RequestID:     "req-policy-test",
		State:         state,
		RawRequest:    body,
		RequestFormat: policyFormatAnthropic,
	}
	shape(arcContext)
	return fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC:      arcContext,
	})
}

// A conversation held on an arm between schedule boundaries is released from
// it when the prompt outgrows that arm, the way it is released from an arm
// the context excluded: the turn decides again among the arms that fit
// rather than failing every turn until the next boundary.
func TestPolicySelectorReleasesAHeldArmTheTurnDoesNotFit(t *testing.T) {
	fixture := newPolicySelectorFixture(t, config.RaylineARCModelScheduleTaskTurnCompaction)
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	state, _ := raylinearc.NewEpisodeState(2)
	held := 0
	state.PreviousArm, state.TurnIndex = &held, 2
	state.Policy = &raylinearc.PolicyEpisodeState{PrefixDigest: raylinearc.MessagesDigest(nil, 0)}
	result, err := fixture.selectOnWith(t, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}),
		func(arcContext *selection.RaylineARCSelectionContext) {
			arcContext.ContextTokens = 300000
			arcContext.OverContextArms = []bool{true, false}
		})
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	offered := fixture.fake.received()[0].Selection.AvailableActionIDs
	for _, actionID := range offered {
		if fixture.selector.armedComponents().scorer.(*policyServiceScorer).bindings[actionID].arm == held {
			t.Fatalf("held turn offered %v, which includes the arm the prompt outgrew", offered)
		}
	}
	if len(offered) == 0 || result.RaylineARC.SelectedArm == held {
		t.Fatalf("offered %v, selected arm %d, want the turn moved off arm %d", offered, result.RaylineARC.SelectedArm, held)
	}
}
