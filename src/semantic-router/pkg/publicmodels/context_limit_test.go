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

// Two ARC decisions in one recipe publish the smaller limit, since the alias
// cannot choose the decision and only the smaller is guaranteed; a decision
// that declares none does not lower it to nothing.
func TestAliasPublishesTheSmallerLimitAcrossDecisions(t *testing.T) {
	declared := func(limit int) config.Decision {
		decision := arcDecision("arc", config.ModelRef{Model: "arm"})
		decision.Algorithm.RaylineARC.MaxContextTokens = limit
		return decision
	}
	recipe := &config.RoutingRecipe{Profile: config.RoutingProfile{Decisions: []config.Decision{
		declared(1_000_000), declared(0), declared(262_144),
	}}}
	if got := routingContextLimitOf(recipe); got == nil || *got != 262_144 {
		t.Fatalf("routingContextLimitOf() = %v, want the smaller declared limit 262144", got)
	}
	recipe.Profile.Decisions = []config.Decision{declared(0)}
	if got := routingContextLimitOf(recipe); got != nil {
		t.Fatalf("routingContextLimitOf() = %d, want nil when no decision declares a limit", *got)
	}
	if got := routingContextLimitOf(nil); got != nil {
		t.Fatalf("routingContextLimitOf(nil) = %d, want nil", *got)
	}
}
