package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const selectorFile = "pkg/extproc/req_filter_classification_selector.go"

// selectionFailure reaches one fallback site in selectorFile, each of which
// logs "using default candidate" and returns (candidates[0], method, nil).
type selectionFailure struct {
	name   string
	line   int
	router func() *OpenAIRouter
}

func selectionFailures() []selectionFailure {
	return []selectionFailure{
		{"no selector for the declared algorithm", 59, func() *OpenAIRouter {
			return &OpenAIRouter{ModelSelector: selection.NewRegistry()}
		}},
		{"selector returned an error", 99, func() *OpenAIRouter {
			registry := selection.NewRegistry()
			registry.Register(selection.MethodStatic, selectionResultSelector{err: errors.New("backend unreachable")})
			return &OpenAIRouter{ModelSelector: registry}
		}},
	}
}

func selectUnderFailure(failure selectionFailure, onError string) (string, error) {
	selected, _, err := failure.router().selectModelFromCandidates(
		&selection.SelectionContext{CandidateModels: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}}},
		&config.AlgorithmConfig{Type: config.DecisionAlgorithmStatic, OnError: onError},
		&RequestContext{},
	)
	if selected == nil {
		return "<nil>", err
	}
	return selected.Model, err
}

// TestSelectorFailureHonoursOnError is the runtime half of the ask.
//
// A non-prompt decision that declares on_error still answers with the first
// declared candidate and a nil error when the selector is missing or fails,
// because no fallback site consults the field. The repository gives a failure
// policy real effect elsewhere: classifier_on_error.go:23 is allow|block.
func TestSelectorFailureHonoursOnError(t *testing.T) {
	selection.InitializeMetrics()
	for _, failure := range selectionFailures() {
		t.Run(failure.name, func(t *testing.T) {
			model, err := selectUnderFailure(failure, "fail_closed")
			if err == nil {
				t.Errorf("%s:%d: on_error=fail_closed answered with candidate %q and no error", selectorFile, failure.line, model)
			}
		})
	}
}

// TestSelectorFailureDowngradesToFirstCandidateToday pins the present
// behaviour so a change to it shows up in the diff rather than only in the
// test above: on_error has no effect at selection time, whatever it says.
func TestSelectorFailureDowngradesToFirstCandidateToday(t *testing.T) {
	selection.InitializeMetrics()
	for _, onError := range []string{"", "fallback", "fail_closed"} {
		for _, failure := range selectionFailures() {
			model, err := selectUnderFailure(failure, onError)
			if err != nil || model != "model-a" {
				t.Errorf("%s:%d: on_error=%q gave (%q, %v), want (model-a, nil)", selectorFile, failure.line, onError, model, err)
			}
		}
	}
}
