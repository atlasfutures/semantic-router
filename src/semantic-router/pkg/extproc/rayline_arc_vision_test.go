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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestRaylineARCImageSignalSurvivesTheDroppedProjection(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name    string
		request *llmprotocol.Request
		want    bool
	}{
		{
			name: "text only",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{{
				Role: llmprotocol.RoleUser,
				Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentText, Text: "describe this"},
				},
			}}},
			want: false,
		},
		{
			name: "image block in a user message",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{{
				Role: llmprotocol.RoleUser,
				Content: []llmprotocol.Content{
					{Kind: llmprotocol.ContentText, Text: "describe this"},
					{
						Kind:      llmprotocol.ContentImage,
						MediaType: "image/png",
						URL:       "https://example.invalid/a.png",
					},
				},
			}}},
			want: true,
		},
		{
			// The provider receives the tool result verbatim, so an image
			// nested in one is image input just as much as a top-level block.
			name: "image nested in a tool result",
			request: &llmprotocol.Request{Messages: []llmprotocol.Message{{
				Role: llmprotocol.RoleTool,
				Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentToolResult,
					ToolResult: &llmprotocol.ToolResult{
						CallID: "call-1",
						Content: []llmprotocol.Content{{
							Kind:      llmprotocol.ContentImage,
							MediaType: "image/png",
							URL:       "https://example.invalid/b.png",
						}},
					},
				}},
			}}},
			want: true,
		},
		{
			name: "image block in the opening instructions",
			request: &llmprotocol.Request{
				Instructions: []llmprotocol.InstructionBlock{{
					Content: []llmprotocol.Content{{
						Kind:      llmprotocol.ContentImage,
						MediaType: "image/png",
						URL:       "https://example.invalid/c.png",
					}},
				}},
			},
			want: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := requestCarriesImageInput(test.request); got != test.want {
				t.Fatalf("requestCarriesImageInput() = %t, want %t", got, test.want)
			}
		})
	}
}

func TestRaylineARCNonVisionArmsReadTheModelCards(t *testing.T) {
	t.Parallel()
	no := false
	yes := true
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"arm-0": {},
				"arm-1": {Vision: &no},
				"arm-2": {Vision: &yes},
			},
		},
	}}
	refs := []config.ModelRef{
		{Model: "arm-0"},
		{Model: "arm-1"},
		{Model: "arm-2"},
	}
	want := []bool{false, true, false}
	if got := router.nonVisionArms(refs); !reflect.DeepEqual(got, want) {
		t.Fatalf("nonVisionArms() = %v, want %v", got, want)
	}
}

// An unmarked catalog must behave exactly as it did before the flag existed,
// which is what makes the flag safe to add without touching any config.
func TestRaylineARCNonVisionArmsAreNilWhenNothingIsMarked(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{"arm-0": {}},
		},
	}}
	if got := router.nonVisionArms([]config.ModelRef{{Model: "arm-0"}}); got != nil {
		t.Fatalf("nonVisionArms() = %v, want nil", got)
	}
}

func visionSelectionContext(
	state *raylinearc.EpisodeState,
	imageBearing bool,
	nonVisionArms []bool,
) *selection.SelectionContext {
	selectionContext := validARCSelectionContext(state)
	selectionContext.RaylineARC.ImageBearing = imageBearing
	selectionContext.RaylineARC.NonVisionArms = nonVisionArms
	return selectionContext
}

func armedVisionSelector(
	scorer *fakeARCScorer,
	encoder *fakeARCEncoder,
) *raylineARCSelector {
	return newRaylineARCSelector(scorer, encoder, nil, "artifact-revision")
}

func visionScorer() *fakeARCScorer {
	return &fakeARCScorer{
		workerIDs: []string{"worker-a", "worker-b"},
		decision: raylinearc.Decision{
			SelectedArm:     0,
			SelectedWorker:  "worker-a",
			RawScores:       []float32{0.9, 0.1},
			AdjustedScores:  []float32{0.9, 0.1},
			SwitchCostUSD:   []float64{0, 0},
			CacheMissTokens: []int{0, 0},
		},
	}
}

func visionEncoder() *fakeARCEncoder {
	return &fakeARCEncoder{result: &raylinearc.EncoderResult{
		Embedding:        make([]float32, 1024),
		SerializedTokens: 10,
		ModelRevision:    config.RaylineARCEncoderModelRevision,
	}}
}

func TestRaylineARCSelectorExcludesNonVisionArmsOnAnImageTurn(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())

	if _, err := selector.Select(
		context.Background(),
		visionSelectionContext(state, true, []bool{false, true}),
	); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{false, true}) {
		t.Fatalf(
			"scorer exclusion = %v, want the non-vision arm excluded",
			scorer.excluded,
		)
	}
}

// A text-only turn must route exactly as it did before the flag existed, even
// when an arm is marked, or the flag would cost throughput for nothing.
func TestRaylineARCSelectorLeavesATextTurnUnconstrained(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())

	if _, err := selector.Select(
		context.Background(),
		visionSelectionContext(state, false, []bool{false, true}),
	); err != nil {
		t.Fatal(err)
	}
	if scorer.excluded != nil {
		t.Fatalf(
			"scorer exclusion = %v, want none on a text-only turn",
			scorer.excluded,
		)
	}
}

// An episode parked on a non-vision arm must be pulled off it, not kept there
// by the stay margin. The selector's part is to mark the arm.
func TestRaylineARCSelectorExcludesANonVisionPreviousArm(t *testing.T) {
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
		visionSelectionContext(state, true, []bool{false, true}),
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

// No arm can serve the turn, so there is nothing to degrade to. The router
// fails closed with its own class rather than sending an image to a model
// that will answer 404.
func TestRaylineARCSelectorFailsClosedWhenNoArmTakesImages(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	encoder := visionEncoder()
	selector := armedVisionSelector(visionScorer(), encoder)

	_, err = selector.Select(
		context.Background(),
		visionSelectionContext(state, true, []bool{true, true}),
	)
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "no_vision_arm" {
		t.Fatalf("Select() error = %v, want class no_vision_arm", err)
	}
	// The encoder is the expensive dependency. A request no arm can serve
	// must not occupy it.
	if encoder.calls != 0 {
		t.Fatalf("encoder calls = %d, want 0", encoder.calls)
	}
	if failure.contended() {
		t.Fatal("no_vision_arm must answer 503, not 429")
	}
}
