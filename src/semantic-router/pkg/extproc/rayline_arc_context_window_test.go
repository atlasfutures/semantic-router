package extproc

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The mask reads the same card and the same number the generic context
// filter reads, so a turn is over an arm's window here exactly when the
// filter would have removed that arm on a plain decision.
func TestRaylineARCOverContextArmsReadTheModelCards(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"arm-0": {ContextWindowSize: 1048576},
				"arm-1": {ContextWindowSize: 262144},
				"arm-2": {},
			},
		},
	}}
	refs := []config.ModelRef{{Model: "arm-0"}, {Model: "arm-1"}, {Model: "arm-2"}}
	if got := router.overContextArms(refs, 300000); !reflect.DeepEqual(got, []bool{false, true, false}) {
		t.Fatalf("overContextArms(300000) = %v, want only the 262144 arm marked; an undeclared window excludes nothing", got)
	}
	for _, tokens := range []int{0, 200000} {
		if got := router.overContextArms(refs, tokens); got != nil {
			t.Fatalf("overContextArms(%d) = %v, want nil: nothing is over its window", tokens, got)
		}
	}
	if got := router.overContextArms(refs, 1100000); !reflect.DeepEqual(got, []bool{true, true, false}) {
		t.Fatalf("overContextArms(1100000) = %v, want both declared arms marked", got)
	}
}

func contextSelectionContext(state *raylinearc.EpisodeState, overContextArms []bool) *selection.SelectionContext {
	selectionContext := validARCSelectionContext(state)
	selectionContext.RaylineARC.ContextTokens = 300000
	selectionContext.RaylineARC.OverContextArms = overContextArms
	return selectionContext
}

// A long turn excludes the arm it does not fit, the way an image turn
// excludes a text-only arm: the arm stays in the list at its ordinal and the
// policy scores around it.
func TestRaylineARCSelectorExcludesArmsTheTurnDoesNotFit(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())
	if _, err := selector.Select(context.Background(), contextSelectionContext(state, []bool{false, true})); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{false, true}) {
		t.Fatalf("scorer exclusion = %v, want the over-window arm excluded", scorer.excluded)
	}
}

// A turn no arm can hold fails closed with its own class, before the encoder
// is touched, and as a 503 the gateway replays rather than a 400 that ends
// the turn.
func TestRaylineARCSelectorFailsClosedWhenNoArmHoldsTheTurn(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	encoder := visionEncoder()
	selector := armedVisionSelector(visionScorer(), encoder)
	_, err = selector.Select(context.Background(), contextSelectionContext(state, []bool{true, true}))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != arcFailureNoContextArm {
		t.Fatalf("Select() error = %v, want class %s", err, arcFailureNoContextArm)
	}
	if encoder.calls != 0 {
		t.Fatalf("encoder calls = %d, want 0", encoder.calls)
	}
	if failure.contended() {
		t.Fatalf("%s must answer 503, not 429", arcFailureNoContextArm)
	}
}

// The masks compose: an arm that fits the prompt but is out of service stays
// excluded, and one that is in service but too small stays excluded, so the
// only eligible arm is the one that passes both.
func TestRaylineARCSelectorComposesTheContextMaskWithTheOthers(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())
	selectionContext := contextSelectionContext(state, []bool{false, true})
	selectionContext.RaylineARC.DisabledArms = []bool{false, false}
	if _, err := selector.Select(context.Background(), selectionContext); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{false, true}) {
		t.Fatalf("scorer exclusion = %v, want the over-window arm excluded and the enabled one kept", scorer.excluded)
	}
	selectionContext = contextSelectionContext(state, []bool{false, true})
	selectionContext.RaylineARC.DisabledArms = []bool{true, false}
	_, err = selector.Select(context.Background(), selectionContext)
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "no_enabled_arm" {
		t.Fatalf("Select() error = %v, want no_enabled_arm when the only arm that fits is disabled", err)
	}
}
