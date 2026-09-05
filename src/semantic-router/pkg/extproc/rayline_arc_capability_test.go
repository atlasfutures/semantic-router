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
	"net/http"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Accept-by-default stops the cell refusing a request it cannot fully express.
// Two features would then be dropped silently and change what the model is
// shown: a tool the source API runs, and an image inside a tool result. The
// capability gate routes those instead of dropping them, on the same model-card
// mechanism the vision flag already uses.

func TestRaylineARCRoutingCapabilitiesReadTheNeutralRequest(t *testing.T) {
	t.Parallel()
	serverTool := llmprotocol.Request{Tools: []llmprotocol.Tool{
		{Name: "lookup", Type: "custom"},
		{Type: "web_search_20250305"},
	}}
	toolResultImage := llmprotocol.Request{Messages: []llmprotocol.Message{{
		Role: llmprotocol.RoleTool,
		Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult,
			ToolResult: &llmprotocol.ToolResult{
				CallID: "call-1",
				Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentImage, MediaType: "image/png", Data: "aGk=",
				}},
			},
		}},
	}}}
	textOnly := llmprotocol.Request{
		Tools: []llmprotocol.Tool{{Name: "lookup"}},
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleTool,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{
					CallID:  "call-1",
					Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "ok"}},
				},
			}},
		}},
	}
	for _, test := range []struct {
		name    string
		request llmprotocol.Request
		want    []string
	}{
		{name: "server tool", request: serverTool, want: []string{llmprotocol.RoutingCapabilityServerTools}},
		{name: "tool result image", request: toolResultImage, want: []string{llmprotocol.RoutingCapabilityToolResultImages}},
		{name: "text only", request: textOnly, want: nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got := llmprotocol.RequiredRoutingCapabilities(test.request)
			if !reflect.DeepEqual(got, test.want) {
				t.Fatalf("RequiredRoutingCapabilities() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestRaylineARCIncapableArmsReadTheModelCards(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"arm-0": {},
				"arm-1": {Capabilities: []string{"tools", llmprotocol.RoutingCapabilityServerTools}},
				"arm-2": {Capabilities: []string{llmprotocol.RoutingCapabilityToolResultImages}},
			},
		},
	}}
	refs := []config.ModelRef{{Model: "arm-0"}, {Model: "arm-1"}, {Model: "arm-2"}}
	required := []string{llmprotocol.RoutingCapabilityServerTools}
	want := []bool{true, false, true}
	if got := router.incapableArms(refs, required); !reflect.DeepEqual(got, want) {
		t.Fatalf("incapableArms() = %v, want %v", got, want)
	}
}

// A turn that requires nothing must route exactly as it did before the gate
// existed, or the gate would cost throughput on every ordinary turn.
func TestRaylineARCIncapableArmsAreNilWhenNothingIsRequired(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{"arm-0": {}},
		},
	}}
	if got := router.incapableArms([]config.ModelRef{{Model: "arm-0"}}, nil); got != nil {
		t.Fatalf("incapableArms() = %v, want nil", got)
	}
}

func capabilitySelectionContext(
	state *raylinearc.EpisodeState,
	incapableArms []bool,
) *selection.SelectionContext {
	selectionContext := validARCSelectionContext(state)
	selectionContext.RaylineARC.RequiredCapabilities = []string{
		llmprotocol.RoutingCapabilityServerTools,
	}
	selectionContext.RaylineARC.IncapableArms = incapableArms
	return selectionContext
}

func TestRaylineARCSelectorExcludesArmsWithoutTheCapability(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())

	if _, err := selector.Select(
		context.Background(),
		capabilitySelectionContext(state, []bool{false, true}),
	); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{false, true}) {
		t.Fatalf("scorer exclusion = %v, want the incapable arm excluded", scorer.excluded)
	}
}

// The whole point of the gate. No arm holds the capability, so there is no arm
// to route to and nothing honest to degrade to. The cell answers a status the
// gateway replays; a 400 would end the user's turn, because the gateway's
// BODY_LEVEL_STATUSES excludes 400, 413 and 422 and that rule stays.
func TestRaylineARCSelectorFailsReplayablyWhenNoArmIsCapable(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	encoder := visionEncoder()
	selector := armedVisionSelector(visionScorer(), encoder)

	_, err = selector.Select(
		context.Background(),
		capabilitySelectionContext(state, []bool{true, true}),
	)
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != arcFailureNoCapableArm {
		t.Fatalf("Select() error = %v, want class %s", err, arcFailureNoCapableArm)
	}
	// The encoder is the expensive dependency. A turn no arm can serve must
	// not occupy it.
	if encoder.calls != 0 {
		t.Fatalf("encoder calls = %d, want 0", encoder.calls)
	}
	if selectionFailureIsCallerError(failure.class) {
		t.Fatal("no_capable_arm is not the caller's omission and must not answer 400")
	}
	if selectionFailureIsContended(failure.class) {
		t.Fatal("no_capable_arm does not clear on retry and must not answer 429")
	}
}

// The status the cell actually returns, read off the admission contract rather
// than inferred from the class. 503 is what the cold-start and encoder paths
// already answer, and the gateway replays it.
func TestNoCapableArmAnswersAReplayableStatus(t *testing.T) {
	router := &OpenAIRouter{}
	failure := &modelSelectionFailure{
		algorithm: "rayline_arc",
		class:     arcFailureNoCapableArm,
	}
	response := router.authoritativeSelectionFailureResponse(failure, &RequestContext{})
	if response == nil {
		t.Fatal("no_capable_arm produced no admission response")
	}
	status := int(response.GetImmediateResponse().GetStatus().GetCode())
	if status != http.StatusServiceUnavailable {
		t.Fatalf("no_capable_arm status = %d, want %d", status, http.StatusServiceUnavailable)
	}
}
