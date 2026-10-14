package publicmodels

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The candidates cell with the decision's own context limit declared.
func contextLimitListing(t *testing.T, limit string) map[string]interface{} {
	t.Helper()
	t.Setenv("PUBLICMODELS_CANDIDATES_TEST_KEY", "public-candidates-test-key")
	source := strings.Replace(candidatesConfig,
		"        rayline_arc:\n          routes_api:",
		"        rayline_arc:\n          max_context_tokens: "+limit+"\n          routes_api:", 1)
	if source == candidatesConfig {
		t.Fatal("the candidates config no longer has the rayline_arc block the test rewrites")
	}
	cfg, err := config.ParseYAMLBytes([]byte(source))
	if err != nil {
		t.Fatalf("the config did not load: %v", err)
	}
	return marshalListing(t, NewOpenAIModelList(cfg, 123))
}

// The alias publishes the decision's declared limit beside its candidates,
// as the number a gateway may advertise for the alias: the cell refuses a
// longer turn before any arm is consulted, whatever the arms' windows. A
// cell that declares none publishes no number, not a zero.
func TestAliasPublishesTheDeclaredContextLimit(t *testing.T) {
	routing := modelEntry(t, contextLimitListing(t, "1000000"), "rayline/arc-dev")["routing"].(map[string]interface{})
	if got, ok := routing["max_context_tokens"]; !ok || got != float64(1_000_000) {
		t.Fatalf("routing.max_context_tokens = %v (present %t), want 1000000", got, ok)
	}
	if candidates, _ := routing["candidates"].([]interface{}); len(candidates) != 3 {
		t.Fatalf("candidates = %v, want the three arms still listed beside the limit", candidates)
	}
	routing = modelEntry(t, candidatesListing(t), "rayline/arc-dev")["routing"].(map[string]interface{})
	if got, ok := routing["max_context_tokens"]; ok {
		t.Fatalf("routing.max_context_tokens = %v, want absent on a cell that declares none", got)
	}
}

// A recipe has one ARC limit: its ARC decisions share one selector and
// startup refuses differing selection configs, so the listing reads the
// first ARC decision's, skipping decisions of other algorithms, and nil when
// it declares none.
func TestAliasReadsTheRecipesOneLimit(t *testing.T) {
	declared := func(limit int) config.Decision {
		decision := arcDecision("arc", config.ModelRef{Model: "arm"})
		decision.Algorithm.RaylineARC.MaxContextTokens = limit
		return decision
	}
	plain := config.Decision{Name: "plain", ModelRefs: []config.ModelRef{{Model: "arm"}}}
	recipe := &config.RoutingRecipe{Profile: config.RoutingProfile{Decisions: []config.Decision{
		plain, declared(1_000_000), declared(1_000_000),
	}}}
	if got := routingContextLimitOf(recipe); got == nil || *got != 1_000_000 {
		t.Fatalf("routingContextLimitOf() = %v, want the ARC decision's 1000000 past the plain decision", got)
	}
	recipe.Profile.Decisions = []config.Decision{plain, declared(0)}
	if got := routingContextLimitOf(recipe); got != nil {
		t.Fatalf("routingContextLimitOf() = %d, want nil when the ARC decision declares none", *got)
	}
	if got := routingContextLimitOf(nil); got != nil {
		t.Fatalf("routingContextLimitOf(nil) = %d, want nil", *got)
	}
}
