package services

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

type evalModelSelectorStub struct {
	input EvalModelSelectionInput
}

func (s *evalModelSelectorStub) SelectModelForEval(
	input EvalModelSelectionInput,
) EvalModelSelection {
	s.input = input
	return EvalModelSelection{
		SelectedModel: "model-b",
		Status:        EvalSelectionSelected,
		Method:        "multi_factor",
		Reason:        "highest live score",
	}
}

func TestPopulateEvalModelSelectionReturnsConcreteRuntimeChoice(t *testing.T) {
	selector := &evalModelSelectorStub{}
	service := &ClassificationService{}
	service.SetEvalModelSelector(selector)
	response := &EvalResponse{Recipe: "balanced"}
	matchedDecision := &config.Decision{
		Name:      "balanced-route",
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
	}
	service.populateEvalModelSelection(
		response,
		intentSignalInput{
			currentUserText: "Explain the tradeoff.",
			requestFacts: classification.RequestFacts{
				ContextTokenFloor: 4096,
			},
		},
		&decision.DecisionResult{
			Decision:     matchedDecision,
			MatchedRules: []string{"domain:engineering"},
		},
		nil,
		nil,
	)

	if response.SelectedModel != "model-b" || response.SelectionStatus != EvalSelectionSelected {
		t.Fatalf("selection response = %+v", response)
	}
	if selector.input.Decision != matchedDecision || selector.input.Recipe != "balanced" {
		t.Fatalf("selector scope = %+v", selector.input)
	}
	if selector.input.Query != "Explain the tradeoff." || selector.input.Category != "engineering" {
		t.Fatalf("selector semantic input = %+v", selector.input)
	}
	if selector.input.ContextTokenCount != 4096 {
		t.Fatalf("selector context count = %d", selector.input.ContextTokenCount)
	}
}

func TestPopulateEvalModelSelectionDoesNotInventFirstRecommendedModel(t *testing.T) {
	response := &EvalResponse{Recipe: "accuracy"}
	service := &ClassificationService{}
	service.populateEvalModelSelection(
		response,
		intentSignalInput{},
		&decision.DecisionResult{Decision: &config.Decision{
			Name:      "fusion-route",
			ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		}},
		nil,
		nil,
	)

	if response.SelectedModel != "" || response.SelectionStatus != EvalSelectionUnavailable {
		t.Fatalf("unwired Eval invented a final model: %+v", response)
	}
}

// Eval selects under the context signal's count when it ran, as the routed
// path does, never below the request's floor.
func TestPopulateEvalModelSelectionUsesTheEvaluatedTokenCount(t *testing.T) {
	selector := &evalModelSelectorStub{}
	service := &ClassificationService{}
	service.SetEvalModelSelector(selector)
	matched := &decision.DecisionResult{Decision: &config.Decision{Name: "d", ModelRefs: []config.ModelRef{{Model: "m"}}}}
	input := intentSignalInput{requestFacts: classification.RequestFacts{ContextTokenFloor: 4096}}
	referencing := classification.NewClassifierWithContextSignalForTest(&config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
		Name: "long-turns", Rules: config.RuleNode{Type: config.SignalTypeContext, Name: "long"},
	}}}}, []config.ContextRule{{Name: "long", MinTokens: "1000"}})
	service.populateEvalModelSelection(&EvalResponse{}, input, matched, &classification.SignalResults{TokenCount: 300000}, referencing)
	if selector.input.ContextTokenCount != 300000 {
		t.Fatalf("selector context count = %d, want the signal's 300000", selector.input.ContextTokenCount)
	}
	service.populateEvalModelSelection(&EvalResponse{}, input, matched, &classification.SignalResults{TokenCount: 10}, referencing)
	if selector.input.ContextTokenCount != 4096 {
		t.Fatalf("selector context count = %d, want the floor 4096 over a smaller signal count", selector.input.ContextTokenCount)
	}
	// A context signal no decision rule references is forced to run by Eval
	// and skipped by routing; its count is set aside, as routing sets it.
	unreferenced := classification.NewClassifierWithContextSignalForTest(&config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
		Name: "keywords", Rules: config.RuleNode{Type: config.SignalTypeKeyword, Name: "math"},
	}}}}, []config.ContextRule{{Name: "long", MinTokens: "1000"}})
	service.populateEvalModelSelection(&EvalResponse{}, input, matched, &classification.SignalResults{TokenCount: 300000}, unreferenced)
	if selector.input.ContextTokenCount != 4096 {
		t.Fatalf("selector context count = %d, want the floor when routing would not evaluate the context signal", selector.input.ContextTokenCount)
	}
}
