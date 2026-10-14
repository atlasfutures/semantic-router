package extproc

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func contextLimitAlgorithm(limit int) *config.AlgorithmConfig {
	return &config.AlgorithmConfig{
		Type: config.RaylineARCAlgorithmType,
		RaylineARC: &config.RaylineARCAlgorithmConfig{
			Episode:          config.RaylineARCEpisodeConfig{IDHeader: "x-rayline-session"},
			MaxContextTokens: limit,
		},
	}
}

func contextLimitRequest(tokens int) *RequestContext {
	return &RequestContext{
		Headers:              map[string]string{},
		VSRContextTokenCount: tokens,
		SemanticRequest:      testNeutralRequest("auto", "a long prompt"),
	}
}

// A turn longer than the cell's declared limit is refused while preparing
// the selection context, before the episode is prepared and before any arm
// mask is built, so no lease is taken and no encoder is consulted for it. A
// turn within the limit, and every turn on a cell that declares none, goes
// on exactly as before.
func TestRaylineARCSelectionContextRefusesATurnOverTheContextLimit(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"arm-0": {ContextWindowSize: 1048576},
				"arm-1": {ContextWindowSize: 262144},
			},
		},
	}}
	refs := []config.ModelRef{{Model: "arm-0"}, {Model: "arm-1"}}
	for _, tc := range []struct {
		name   string
		limit  int
		tokens int
		refuse bool
	}{
		{name: "undeclared limit admits every length", limit: 0, tokens: 2_000_000},
		{name: "under the limit", limit: 1_000_000, tokens: 300_000},
		{name: "at the limit", limit: 1_000_000, tokens: 1_000_000},
		{name: "no estimate", limit: 1_000_000, tokens: 0},
		{name: "over the limit", limit: 1_000_000, tokens: 1_000_001, refuse: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			arcContext := router.buildRaylineARCSelectionContext(
				contextLimitAlgorithm(tc.limit), contextLimitRequest(tc.tokens), refs, raylineARCEpisodeEphemeral,
			)
			if !tc.refuse {
				if arcContext.PreparationFailure != "" {
					t.Fatalf("preparation failed: %s", arcContext.PreparationFailure)
				}
				if arcContext.ContextTokens != tc.tokens {
					t.Fatalf("context tokens = %d, want %d", arcContext.ContextTokens, tc.tokens)
				}
				return
			}
			if arcContext.PreparationFailure != arcFailureOverContextLimit {
				t.Fatalf("preparation failure = %q, want %s", arcContext.PreparationFailure, arcFailureOverContextLimit)
			}
			if arcContext.Turns != nil || arcContext.OverContextArms != nil || arcContext.State != nil {
				t.Fatalf("a refused turn was still prepared: turns=%v masks=%v state=%v",
					arcContext.Turns, arcContext.OverContextArms, arcContext.State)
			}
		})
	}
}

// Eval refuses what the routed path refuses: in artifact mode a turn over
// the cell's limit is unavailable even when an arm's window would hold it,
// a turn within it is execution_required, and a policy-service decision is
// not refused on the limit, as the routed path is not.
func TestEvalRefusesAnARCTurnOverTheContextLimit(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
		"arm-0": {ContextWindowSize: 1048576}, "arm-1": {ContextWindowSize: 262144},
	}}}}
	decision := func(limit int, policy bool) *config.Decision {
		algorithm := contextLimitAlgorithm(limit)
		algorithm.OnError = "fail_closed"
		if policy {
			algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "http://policy-service:8000"}
		}
		return &config.Decision{Name: "rayline-arc", ModelRefs: []config.ModelRef{{Model: "arm-0"}, {Model: "arm-1"}}, Algorithm: algorithm}
	}
	eval := func(decision *config.Decision, tokens int) services.EvalModelSelection {
		return router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision, ContextTokenCount: tokens})
	}
	if result := eval(decision(500_000, false), 600_000); result.Status != services.EvalSelectionUnavailable {
		t.Fatalf("eval status = %q (%s), want %s over the cell limit, though arm-0 would hold it", result.Status, result.Reason, services.EvalSelectionUnavailable)
	}
	if result := eval(decision(500_000, false), 300_000); result.Status != services.EvalSelectionExecutionRequired {
		t.Fatalf("eval status = %q (%s), want %s within the cell limit", result.Status, result.Reason, services.EvalSelectionExecutionRequired)
	}
	if result := eval(decision(0, false), 600_000); result.Status != services.EvalSelectionExecutionRequired {
		t.Fatalf("eval status = %q (%s), want %s with no limit declared", result.Status, result.Reason, services.EvalSelectionExecutionRequired)
	}
	if result := eval(decision(500_000, true), 600_000); result.Status == services.EvalSelectionUnavailable &&
		strings.Contains(result.Reason, "max_context_tokens") {
		t.Fatalf("eval refused a policy-service decision on the cell limit: %s", result.Reason)
	}
}

// A recipe's ARC decisions share one selector, so two that declare
// different limits are one of the conflicting configs startup refuses, as
// the listing's one-limit reading relies on; the same limit, or none on
// both, shares the selector as before.
func TestDifferingContextLimitsAreConflictingSelectionConfigs(t *testing.T) {
	t.Parallel()
	base := *raylineARCAlgorithmConfigForTest().RaylineARC
	same, differing := base, base
	same.MaxContextTokens, differing.MaxContextTokens = 262144, 1_000_000
	declared := base
	declared.MaxContextTokens = 262144
	if !sameRaylineARCSelectionConfig(&base, &base) || !sameRaylineARCSelectionConfig(&declared, &same) {
		t.Fatal("equal limits read as conflicting selector configs")
	}
	if sameRaylineARCSelectionConfig(&same, &differing) || sameRaylineARCSelectionConfig(&base, &declared) {
		t.Fatal("different limits read as one selector config")
	}
}

// A policy-service cell is not refused on its limit here: the service
// refuses a context its encoder cannot hold on its own count, and the router
// keeps the episode on its held arm through that refusal, which a pre-lease
// 503 would bypass. The limit is still published and still bounded at load.
func TestRaylineARCSelectionContextDoesNotRefuseOnTheLimitInPolicyServiceMode(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	algorithm := contextLimitAlgorithm(1_000_000)
	algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{BaseURL: "http://policy-service:8000"}
	request := contextLimitRequest(1_000_001)
	request.SourceFormat = llmprotocol.OpenAIChatV1
	arcContext := router.buildRaylineARCSelectionContext(
		algorithm, request, []config.ModelRef{{Model: "arm-0"}}, raylineARCEpisodeEphemeral,
	)
	if arcContext.PreparationFailure == arcFailureOverContextLimit {
		t.Fatalf("a policy-service turn was refused %s; the service's own refusal and the capacity hold govern there",
			arcFailureOverContextLimit)
	}
	if arcContext.ContextTokens != 1_000_001 {
		t.Fatalf("context tokens = %d, want the estimate carried for the masks", arcContext.ContextTokens)
	}
}

// The class is the router's to answer for, on both entrypoints: not the
// caller's omission (no 400 that ends the turn), not back-pressure (no 429
// that asks the caller to wait), and never a request-shape refusal. It lands
// on the public `unavailable`, the 503 a gateway replays to its fallback.
func TestOverContextLimitIsAReplayableSelectionFailure(t *testing.T) {
	t.Parallel()
	if selectionFailureIsContended(arcFailureOverContextLimit) {
		t.Fatalf("%s must not answer 429: waiting does not shorten the prompt", arcFailureOverContextLimit)
	}
	if selectionFailureIsCallerError(arcFailureOverContextLimit) {
		t.Fatalf("%s must not answer 400: the gateway replays a 503 and ends the turn on a 400", arcFailureOverContextLimit)
	}
	if got := publicSelectionFailureClass(arcFailureOverContextLimit); got != selectionFailureUnavailable {
		t.Fatalf("public class = %q, want %s", got, selectionFailureUnavailable)
	}
	err := prepareFailureError(arcFailureOverContextLimit)
	if errors.Is(err, routerruntime.ErrRouteDecisionContended) || errors.Is(err, routerruntime.ErrRouteDecisionInvalidBody) {
		t.Fatalf("decision-only error = %v, want neither contended nor an invalid body", err)
	}
	if err == nil {
		t.Fatal("decision-only preparation failure produced no error")
	}
	var failure *raylineARCSelectionFailure
	selector := armedVisionSelector(visionScorer(), visionEncoder())
	selectionContext := validARCSelectionContext(nil)
	selectionContext.RaylineARC.PreparationFailure = arcFailureOverContextLimit
	_, selectErr := selector.Select(context.Background(), selectionContext)
	if !errors.As(selectErr, &failure) || failure.class != arcFailureOverContextLimit || failure.contended() {
		t.Fatalf("Select() error = %v, want the %s class as a 503", selectErr, arcFailureOverContextLimit)
	}
}
