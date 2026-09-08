package config

import "testing"

// TestDecisionOnErrorIsValidatedForEveryAlgorithm is the ask.
//
// AlgorithmConfig declares on_error (decision_config.go:126), so the key
// decodes for every algorithm type. Only validator_prompt.go:52 reads it, and
// only for type: prompt, where "fallback" is the sole accepted value. Every
// other type accepts any string that no runtime path acts on.
func TestDecisionOnErrorIsValidatedForEveryAlgorithm(t *testing.T) {
	modelRefs := []ModelRef{{Model: "model-a"}, {Model: "model-b"}}
	blockless := []string{DecisionAlgorithmStatic, DecisionAlgorithmKNN, DecisionAlgorithmKMeans, DecisionAlgorithmSVM, DecisionAlgorithmMLP}
	for _, algorithmType := range blockless {
		algorithm := &AlgorithmConfig{Type: algorithmType, OnError: "fail_closed"}
		if err := validateDecisionAlgorithmConfig("on-error-decision", modelRefs, algorithm); err == nil {
			t.Errorf("algorithm.type=%s accepted on_error=fail_closed, which nothing reads", algorithmType)
		}
	}
}
