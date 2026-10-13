package extproc

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
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

// The declared limit may not exceed what the mounted artifact's encoder
// manifest serves: a cell that promised more would publish a window whose
// longest turns the encoder refuses after admission. Less is a choice;
// undeclared promises nothing.
func TestRaylineARCReadinessRefusesAContextLimitAboveTheArtifactEncoder(t *testing.T) {
	t.Parallel()
	runtime, err := raylinearc.LoadRuntime(writeARCArtifact(t, "https://provider.example.test"))
	if err != nil {
		t.Fatalf("LoadRuntime() error = %v", err)
	}
	capacity := runtime.EncoderContract().MaxTokens
	if capacity != 262144 {
		t.Fatalf("fixture encoder max_tokens = %d, want 262144", capacity)
	}
	for limit, within := range map[int]bool{0: true, capacity - 1: true, capacity: true, capacity + 1: false} {
		arcConfig := &config.RaylineARCAlgorithmConfig{MaxContextTokens: limit}
		if got := raylineARCContextLimitWithinEncoder(runtime, arcConfig); got != within {
			t.Fatalf("limit %d within encoder %d = %t, want %t", limit, capacity, got, within)
		}
	}
}
