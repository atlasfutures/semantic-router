package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// A declared context window must not change what the ARC selector is handed.
// The arm list is positional and the selector refuses a shortened one, so
// the generic context filter would turn a long turn into a candidate_count
// failure the moment a card declared a window. A plain decision keeps the
// filter.
func TestDeclaredContextWindowsLeaveTheARCArmListWhole(t *testing.T) {
	t.Parallel()
	cards := map[string]config.ModelParams{
		"glm":  {ContextWindowSize: 1048576},
		"qwen": {ContextWindowSize: 262144},
		"mimo": {ContextWindowSize: 1048576},
	}
	refs := []config.ModelRef{{Model: "glm"}, {Model: "qwen"}, {Model: "mimo"}}
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: cards}}}
	arc := &config.Decision{
		Name:      "rayline-arc",
		ModelRefs: refs,
		Algorithm: &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType, OnError: "fail_closed", RaylineARC: &config.RaylineARCAlgorithmConfig{}},
	}
	plain := &config.Decision{Name: "plain", ModelRefs: refs}
	const longTurn = 300000

	ctx := &RequestContext{Headers: map[string]string{}}
	eligible, err := router.decisionCandidateModelRefs(arc, arc.Name, longTurn, ctx)
	if err != nil {
		t.Fatalf("decisionCandidateModelRefs(arc) error = %v", err)
	}
	if len(eligible) != len(refs) || len(ctx.VSREligibleModelRefs) != len(refs) {
		t.Fatalf("ARC candidates = %d (context %d), want all %d arms: a shortened list fails the turn with candidate_count", len(eligible), len(ctx.VSREligibleModelRefs), len(refs))
	}
	for index := range refs {
		if eligible[index].Model != refs[index].Model {
			t.Fatalf("ARC candidates[%d] = %q, want %q: order is the contract", index, eligible[index].Model, refs[index].Model)
		}
	}

	eligible, err = router.decisionCandidateModelRefs(plain, plain.Name, longTurn, &RequestContext{Headers: map[string]string{}})
	if err != nil {
		t.Fatalf("decisionCandidateModelRefs(plain) error = %v", err)
	}
	if len(eligible) != 2 {
		t.Fatalf("plain candidates = %v, want the window filter to drop qwen", eligible)
	}
	// Short turns are untouched either way.
	eligible, _ = router.decisionCandidateModelRefs(plain, plain.Name, 1000, &RequestContext{Headers: map[string]string{}})
	if len(eligible) != len(refs) {
		t.Fatalf("plain candidates at 1000 tokens = %v, want all three", eligible)
	}
}

// Eval answers the same way the routed path would: an ARC turn above an
// arm's window is execution_required, because the selector would receive
// every arm, not unavailable, which is what the context filter alone said.
func TestEvalKeepsTheARCArmListWholeToo(t *testing.T) {
	t.Parallel()
	cards := map[string]config.ModelParams{
		"glm":  {ContextWindowSize: 1048576},
		"qwen": {ContextWindowSize: 262144},
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: cards}}}
	arc := &config.Decision{
		Name:      "rayline-arc",
		ModelRefs: []config.ModelRef{{Model: "glm"}, {Model: "qwen"}},
		Algorithm: &config.AlgorithmConfig{Type: config.RaylineARCAlgorithmType, OnError: "fail_closed", RaylineARC: &config.RaylineARCAlgorithmConfig{}},
	}
	result := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: arc, ContextTokenCount: 300000})
	if result.Status != services.EvalSelectionExecutionRequired {
		t.Fatalf("eval status = %q (%s), want %s: the routed path keeps every arm", result.Status, result.Reason, services.EvalSelectionExecutionRequired)
	}
	plain := &config.Decision{Name: "plain", ModelRefs: []config.ModelRef{{Model: "qwen"}}}
	result = router.SelectModelForEval(services.EvalModelSelectionInput{Decision: plain, ContextTokenCount: 300000})
	if result.Status != services.EvalSelectionUnavailable {
		t.Fatalf("plain eval status = %q, want %s when every candidate is over its window", result.Status, services.EvalSelectionUnavailable)
	}
}
