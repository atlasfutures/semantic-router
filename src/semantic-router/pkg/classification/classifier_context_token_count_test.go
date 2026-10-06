package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The count a caller gets here is the count the context signal would have
// stamped on a routed request: the recipe's own counter, floored, and false
// where the recipe declares no context signal.
func TestContextTokenCountUsesTheRecipesCounter(t *testing.T) {
	t.Parallel()
	rules := []config.ContextRule{{Name: "long", MinTokens: "1000"}}
	classifier := &Classifier{contextClassifier: NewContextClassifier(&mockTokenCounter{count: 300000}, rules)}
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
	if _, ok := (&Classifier{}).ContextTokenCount("x", 1); ok {
		t.Fatal("a classifier with no context signal reported a count")
	}
}
