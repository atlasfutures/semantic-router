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
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestRaylineARCDisabledArmsReadTheModelCards(t *testing.T) {
	t.Parallel()
	yes := true
	no := false
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"arm-0": {},
				"arm-1": {Disabled: &yes},
				"arm-2": {Disabled: &no},
			},
		},
	}}
	refs := []config.ModelRef{
		{Model: "arm-0"},
		{Model: "arm-1"},
		{Model: "arm-2"},
	}
	want := []bool{false, true, false}
	if got := router.disabledArms(refs); !reflect.DeepEqual(got, want) {
		t.Fatalf("disabledArms() = %v, want %v", got, want)
	}
}

// An unmarked basket must behave exactly as it did before the flag existed,
// which is what makes the flag safe to add without touching any config.
func TestRaylineARCDisabledArmsAreNilWhenNothingIsMarked(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{"arm-0": {}},
		},
	}}
	if got := router.disabledArms([]config.ModelRef{{Model: "arm-0"}}); got != nil {
		t.Fatalf("disabledArms() = %v, want nil", got)
	}
}

func armConstraintContext(
	state *raylinearc.EpisodeState,
	imageBearing bool,
	nonVisionArms []bool,
	disabledArms []bool,
) *selection.SelectionContext {
	selectionContext := validARCSelectionContext(state)
	selectionContext.RaylineARC.ImageBearing = imageBearing
	selectionContext.RaylineARC.NonVisionArms = nonVisionArms
	selectionContext.RaylineARC.DisabledArms = disabledArms
	return selectionContext
}

// A disabled arm is out of service on every turn, not only on the turns a
// request-level constraint happens to cover.
func TestRaylineARCSelectorExcludesADisabledArmOnATextTurn(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())

	if _, err := selector.Select(
		context.Background(),
		armConstraintContext(state, false, nil, []bool{false, true}),
	); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{false, true}) {
		t.Fatalf(
			"scorer exclusion = %v, want the disabled arm excluded",
			scorer.excluded,
		)
	}
}

// A basket with nothing marked must reach the artifact unconstrained, or the
// flag would cost throughput for every deployment that never sets it.
func TestRaylineARCSelectorLeavesAnUnmarkedBasketUnconstrained(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())

	if _, err := selector.Select(
		context.Background(),
		armConstraintContext(state, false, nil, nil),
	); err != nil {
		t.Fatal(err)
	}
	if scorer.excluded != nil {
		t.Fatalf(
			"scorer exclusion = %v, want none when no arm is marked",
			scorer.excluded,
		)
	}
}

// An episode parked on an arm that has since been disabled must be pulled off
// it, not held there by the stay margin. The selector's part is to mark it.
func TestRaylineARCSelectorExcludesADisabledPreviousArm(t *testing.T) {
	previous := 1
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	state.PreviousArm = &previous
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())

	if _, err := selector.Select(
		context.Background(),
		armConstraintContext(state, false, nil, []bool{false, true}),
	); err != nil {
		t.Fatal(err)
	}
	if len(scorer.excluded) != 2 || !scorer.excluded[previous] {
		t.Fatalf(
			"scorer exclusion = %v, want the previous arm excluded",
			scorer.excluded,
		)
	}
}

// Disabling the whole basket leaves nothing to degrade to. The router says so
// with its own class rather than serving an arm an operator took out.
func TestRaylineARCSelectorFailsClosedWhenEveryArmIsDisabled(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	encoder := visionEncoder()
	selector := armedVisionSelector(visionScorer(), encoder)

	_, err = selector.Select(
		context.Background(),
		armConstraintContext(state, false, nil, []bool{true, true}),
	)
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "no_enabled_arm" {
		t.Fatalf("Select() error = %v, want class no_enabled_arm", err)
	}
	// The encoder is the expensive dependency. A request no arm can serve
	// must not occupy it.
	if encoder.calls != 0 {
		t.Fatalf("encoder calls = %d, want 0", encoder.calls)
	}
	if failure.contended() {
		t.Fatal("no_enabled_arm must answer 503, not 429")
	}
}

// Neither constraint empties the pool alone. Together they do, and the class
// has to name the constraint an operator can act on.
func TestRaylineARCSelectorFailsClosedWhenBothConstraintsEmptyThePool(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	selector := armedVisionSelector(visionScorer(), visionEncoder())

	_, err = selector.Select(
		context.Background(),
		armConstraintContext(state, true, []bool{true, false}, []bool{false, true}),
	)
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "no_enabled_arm" {
		t.Fatalf("Select() error = %v, want class no_enabled_arm", err)
	}
}

// When no arm in the basket takes an image, disabling one changes nothing
// about why the turn cannot be served, so the U17 class is still the answer.
func TestRaylineARCSelectorKeepsTheVisionClassWhenNoArmTakesImages(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	selector := armedVisionSelector(visionScorer(), visionEncoder())

	_, err = selector.Select(
		context.Background(),
		armConstraintContext(state, true, []bool{true, true}, []bool{false, true}),
	)
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "no_vision_arm" {
		t.Fatalf("Select() error = %v, want class no_vision_arm", err)
	}
}

// Both constraints reach the artifact as one mask, so an arm removed by
// either is removed, and the arm both leave standing is the one that serves.
func TestRaylineARCSelectorCombinesBothConstraints(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(3)
	if err != nil {
		t.Fatal(err)
	}
	scorer := &fakeARCScorer{
		workerIDs: []string{"worker-a", "worker-b", "worker-c"},
		decision: raylinearc.Decision{
			SelectedArm:     2,
			SelectedWorker:  "worker-c",
			RawScores:       []float32{0.9, 0.5, 0.1},
			AdjustedScores:  []float32{0.9, 0.5, 0.1},
			SwitchCostUSD:   []float64{0, 0, 0},
			CacheMissTokens: []int{0, 0, 0},
		},
	}
	selectionContext := armConstraintContext(
		state,
		true,
		[]bool{true, false, false},
		[]bool{false, true, false},
	)
	selectionContext.CandidateModels = append(
		selectionContext.CandidateModels,
		config.ModelRef{Model: "worker-c"},
	)
	selector := armedVisionSelector(scorer, visionEncoder())

	if _, err := selector.Select(
		context.Background(),
		selectionContext,
	); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{true, true, false}) {
		t.Fatalf(
			"scorer exclusion = %v, want the union of both constraints",
			scorer.excluded,
		)
	}
}

// A mask that does not describe this basket would silently disable the wrong
// arms, so the selector refuses it instead of guessing.
func TestRaylineARCSelectorRefusesADisabledMaskOfTheWrongLength(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	selector := armedVisionSelector(visionScorer(), visionEncoder())

	_, err = selector.Select(
		context.Background(),
		armConstraintContext(state, false, nil, []bool{false, true, false}),
	)
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "disabled_arm_mapping" {
		t.Fatalf("Select() error = %v, want class disabled_arm_mapping", err)
	}
}

// The exclusion has to reach the trace, or a switch away from a disabled arm
// reads in the logs as the scores changing their mind.
func TestRaylineARCSelectorRecordsADisabledArmInItsTrace(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	scorer.decision.ExcludedArms = []bool{false, true}
	selector := armedVisionSelector(scorer, visionEncoder())

	result, err := selector.Select(
		context.Background(),
		armConstraintContext(state, false, nil, []bool{false, true}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(
		result.RaylineARC.ExcludedArms,
		[]bool{false, true},
	) {
		t.Fatalf("trace exclusion = %v", result.RaylineARC.ExcludedArms)
	}
}
