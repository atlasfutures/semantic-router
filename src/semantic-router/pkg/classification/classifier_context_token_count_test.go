package classification

import (
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The count a caller gets here is the count the context signal would have
// stamped on a routed request: the recipe's own counter, floored, and false
// where the recipe declares no context signal.
func TestContextTokenCountUsesTheRecipesCounter(t *testing.T) {
	t.Parallel()
	rules := []config.ContextRule{{Name: "long", MinTokens: "1000"}}
	referencing := &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
		Name: "long-turns", Rules: config.RuleNode{Type: config.SignalTypeContext, Name: "long"},
	}}}}
	classifier := &Classifier{Config: referencing, contextClassifier: NewContextClassifier(&mockTokenCounter{count: 300000}, rules)}
	if count, ok := classifier.ContextTokenCount("ignored by the mock", 100); !ok || count != 300000 {
		t.Fatalf("ContextTokenCount() = %d, %t, want the counter's 300000", count, ok)
	}
	if count, ok := classifier.ContextTokenCount("ignored by the mock", 400000); !ok || count != 400000 {
		t.Fatalf("ContextTokenCount() = %d, %t, want the floor 400000 to win", count, ok)
	}
	var none *Classifier
	if _, ok := none.ContextTokenCount("x", 1); ok {
		t.Fatal("a nil classifier reported a count")
	}
	if _, ok := (&Classifier{Config: referencing}).ContextTokenCount("x", 1); ok {
		t.Fatal("a classifier with no context signal reported a count")
	}
	// A context signal no decision rule references is skipped on a routed
	// request and the heuristic counts it; the same answer here.
	unreferenced := &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
		Name: "keywords", Rules: config.RuleNode{Type: config.SignalTypeKeyword, Name: "math"},
	}}}}
	idle := &Classifier{Config: unreferenced, contextClassifier: NewContextClassifier(&mockTokenCounter{count: 300000}, rules)}
	if _, ok := idle.ContextTokenCount("x", 1); ok {
		t.Fatal("a context signal no rule references reported a count")
	}
}

// The number ContextTokenCount reports is the number the context signal
// stamps on a routed request for the same text and floor: one counter, one
// floor rule, so the route lookup and the routed turn mask the same arms.
func TestContextTokenCountMatchesTheContextSignal(t *testing.T) {
	t.Parallel()
	rules := []config.ContextRule{{Name: "long", MinTokens: "1000"}}
	cfg := &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
		Name: "long-turns", Rules: config.RuleNode{Type: config.SignalTypeContext, Name: "long"},
	}}}}
	for _, counter := range []int{300000, 50} {
		classifier := &Classifier{Config: cfg, contextClassifier: NewContextClassifier(&mockTokenCounter{count: counter}, rules)}
		var mu sync.Mutex
		results := newSignalResultsForTest()
		classifier.evaluateContextSignal(results, &mu, "the same text", 20000)
		count, ok := classifier.ContextTokenCount("the same text", 20000)
		if !ok || count != results.TokenCount {
			t.Fatalf("ContextTokenCount() = %d, %t; the context signal stamped %d", count, ok, results.TokenCount)
		}
	}
}
