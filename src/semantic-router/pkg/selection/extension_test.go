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

package selection

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// extInput and extTrace stand in for the typed structs that a selector
// registered outside the default set carries in and out of Select.
type extInput struct {
	EpisodeID string
}

type extTrace struct {
	SelectedArm int
}

const (
	extInputKey ExtensionKey = "example.com/router-ext.input"
	extTraceKey ExtensionKey = "example.com/router-ext.trace"
	unusedKey   ExtensionKey = "example.com/router-ext.unused"
)

func TestSelectionContextExtensionRoundTrip(t *testing.T) {
	selCtx := &SelectionContext{}
	if selCtx.Extensions != nil {
		t.Fatalf("expected extensions to be nil by default, got %#v", selCtx.Extensions)
	}

	selCtx.SetExtension(extInputKey, &extInput{EpisodeID: "episode-1"})

	input, ok := Extension[*extInput](selCtx.Extensions, extInputKey)
	if !ok {
		t.Fatal("expected the stored context extension to be readable")
	}
	if input.EpisodeID != "episode-1" {
		t.Fatalf("expected episode-1, got %q", input.EpisodeID)
	}
}

func TestSelectionResultExtensionRoundTrip(t *testing.T) {
	result := &SelectionResult{SelectedModel: "model-a"}
	if result.Extensions != nil {
		t.Fatalf("expected extensions to be nil by default, got %#v", result.Extensions)
	}

	result.SetExtension(extTraceKey, extTrace{SelectedArm: 2})

	trace, ok := Extension[extTrace](result.Extensions, extTraceKey)
	if !ok {
		t.Fatal("expected the stored result extension to be readable")
	}
	if trace.SelectedArm != 2 {
		t.Fatalf("expected arm 2, got %d", trace.SelectedArm)
	}
}

func TestExtensionMissingKeyReportsNotFound(t *testing.T) {
	selCtx := &SelectionContext{}
	selCtx.SetExtension(extInputKey, &extInput{EpisodeID: "episode-1"})

	if _, ok := Extension[*extInput](selCtx.Extensions, unusedKey); ok {
		t.Fatal("expected a missing key to report not found")
	}
}

func TestExtensionWrongTypeReportsNotFound(t *testing.T) {
	selCtx := &SelectionContext{}
	selCtx.SetExtension(extInputKey, &extInput{EpisodeID: "episode-1"})

	trace, ok := Extension[extTrace](selCtx.Extensions, extInputKey)
	if ok {
		t.Fatal("expected a type mismatch to report not found")
	}
	if trace != (extTrace{}) {
		t.Fatalf("expected the zero value on a type mismatch, got %#v", trace)
	}
}

func TestExtensionNilBagIsSafe(t *testing.T) {
	if _, ok := Extension[*extInput](nil, extInputKey); ok {
		t.Fatal("expected a nil extensions bag to report not found")
	}

	var nilCtx *SelectionContext
	nilCtx.SetExtension(extInputKey, &extInput{})

	var nilResult *SelectionResult
	nilResult.SetExtension(extTraceKey, extTrace{})
}

// The router copies a SelectionContext by value to build the router-learning
// candidate set, and rebuilds a SelectionResult when adaptation proposes a
// different model. Extensions must survive both.
func TestExtensionsSurviveShallowCopy(t *testing.T) {
	selCtx := &SelectionContext{}
	selCtx.SetExtension(extInputKey, &extInput{EpisodeID: "episode-1"})
	clonedCtx := *selCtx

	if _, ok := Extension[*extInput](clonedCtx.Extensions, extInputKey); !ok {
		t.Fatal("expected context extensions to survive a shallow copy")
	}

	result := &SelectionResult{SelectedModel: "model-a"}
	result.SetExtension(extTraceKey, extTrace{SelectedArm: 2})
	clonedResult := *result

	if _, ok := Extension[extTrace](clonedResult.Extensions, extTraceKey); !ok {
		t.Fatal("expected result extensions to survive a shallow copy")
	}
}

func TestSelectionContextExtensionsDoNotAffectValidation(t *testing.T) {
	selCtx := &SelectionContext{
		CandidateModels: []config.ModelRef{{Model: "model-a"}},
	}
	selCtx.SetExtension(extInputKey, &extInput{EpisodeID: "episode-1"})

	result := &SelectionResult{SelectedModel: "model-a"}
	result.SetExtension(extTraceKey, extTrace{SelectedArm: 2})

	if err := ValidateSelectionContext(selCtx); err != nil {
		t.Fatalf("expected extensions to leave context validation unchanged, got %v", err)
	}
	if err := ValidateSelectionResult(selCtx, result); err != nil {
		t.Fatalf("expected extensions to leave result validation unchanged, got %v", err)
	}
}
