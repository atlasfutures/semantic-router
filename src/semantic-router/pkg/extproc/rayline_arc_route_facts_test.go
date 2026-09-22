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
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

type fixedWorkerCatalog []raylinearc.WorkerManifest

func (catalog fixedWorkerCatalog) Worker(index int) (raylinearc.WorkerManifest, bool) {
	if index < 0 || index >= len(catalog) {
		return raylinearc.WorkerManifest{}, false
	}
	return catalog[index], true
}

func routeFactsCatalog() fixedWorkerCatalog {
	return fixedWorkerCatalog{
		{ID: "arm-0", Model: "deepseek/deepseek-v4-pro"},
		{ID: "arm-1", Model: "z-ai/glm-5.3-flash"},
		{ID: "arm-2", Model: "anthropic/claude-sonnet-5",
			EstimatedInputCostPerToken:  0.000003,
			EstimatedOutputCostPerToken: 0.000015,
		},
		{ID: "arm-3", Model: "openai/gpt-5-mini"},
	}
}

// The alternatives explain the choice, so they are ordered by how close each
// arm came and the chosen arm is not among them.
func TestRouteAlternativesRankWhatWasConsidered(t *testing.T) {
	t.Parallel()
	trace := &selection.RaylineARCTrace{
		SelectedArm:    0,
		AdjustedScores: []float32{0.9, 0.62, 0.41, 0.7},
		ExcludedArms:   []bool{false, false, false, false},
	}
	// Each entry names its arm, not only its model: two arms can serve one
	// model through different providers, and sorting by score has already
	// removed manifest order as an identity.
	want := []routerruntime.RouteAlternative{
		{Model: "openai/gpt-5-mini", Worker: "arm-3", Score: float64(float32(0.7))},
		{Model: "z-ai/glm-5.3-flash", Worker: "arm-1", Score: float64(float32(0.62))},
		{Model: "anthropic/claude-sonnet-5", Worker: "arm-2", Score: float64(float32(0.41))},
	}
	if got := routeAlternatives(trace, routeFactsCatalog()); !reflect.DeepEqual(got, want) {
		t.Fatalf("routeAlternatives() = %v, want %v", got, want)
	}
}

// An arm a hard constraint removed before scoring never competed. Publishing
// its adjusted score would invite a caller to read it as a near miss.
func TestRouteAlternativesLeaveOutExcludedArms(t *testing.T) {
	t.Parallel()
	trace := &selection.RaylineARCTrace{
		SelectedArm:    0,
		AdjustedScores: []float32{0.9, 0.62, 0.41, 0.7},
		ExcludedArms:   []bool{false, true, false, true},
	}
	got := routeAlternatives(trace, routeFactsCatalog())
	if len(got) != 1 || got[0].Model != "anthropic/claude-sonnet-5" {
		t.Fatalf("routeAlternatives() = %v, want only the one arm that competed", got)
	}
}

func TestRouteAlternativesAreEmptyWhenOnlyOneArmCompeted(t *testing.T) {
	t.Parallel()
	trace := &selection.RaylineARCTrace{
		SelectedArm:    0,
		AdjustedScores: []float32{0.9},
	}
	if got := routeAlternatives(trace, routeFactsCatalog()); len(got) != 0 {
		t.Fatalf("routeAlternatives() = %v, want none", got)
	}
}

// The baseline is a rate card per million tokens, because that is the unit
// every provider publishes and the caller is doing the arithmetic themselves.
func TestRouteBaselineResolvesTheReferenceWorkersRateCard(t *testing.T) {
	t.Parallel()
	got := routeBaseline(routeFactsCatalog(), "arm-2")
	want := routerruntime.RouteBaseline{
		Model:   "anthropic/claude-sonnet-5",
		Pricing: routerruntime.RoutePricing{InputPerMTok: 3, OutputPerMTok: 15},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("routeBaseline() = %+v, want %+v", got, want)
	}
}

// An artifact that declares no counterfactual gets none. A guessed baseline
// would be read as the number we measured against.
func TestRouteBaselineIsEmptyWithoutAReferenceWorker(t *testing.T) {
	t.Parallel()
	for name, reference := range map[string]string{
		"undeclared": "",
		"not an arm": "arm-missing",
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if got := routeBaseline(routeFactsCatalog(), reference); got.Model != "" {
				t.Fatalf("routeBaseline() = %+v, want empty", got)
			}
		})
	}
}
