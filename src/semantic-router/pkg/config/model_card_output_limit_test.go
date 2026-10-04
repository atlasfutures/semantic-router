package config

import (
	"strings"
	"testing"
)

// A card's max_output_tokens reaches the runtime params, compiles into the
// catalog's limits, and exports back as the operator's value.
func TestModelCardMaxOutputTokensRoundTrips(t *testing.T) {
	cfg, err := ParseYAMLBytesWithoutEnvExpansion([]byte(`
version: v0.3
providers:
  models:
    - name: slow-arm
      api_format: openai
      backend_refs:
        - name: primary
          provider: vllm
          base_url: http://127.0.0.1:8000/v1
routing:
  modelCards:
    - name: slow-arm
      max_output_tokens: 16000
`))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.ModelConfig["slow-arm"].MaxOutputTokens; got != 16000 {
		t.Fatalf("runtime max_output_tokens = %d, want 16000", got)
	}
	effective, ok := cfg.EffectiveModelRegistry.Model("slow-arm")
	if !ok || effective.Card.Card.Limits.MaxOutputTokens != 16000 {
		t.Fatalf("catalog limits.max_output_tokens = %+v", effective.Card.Card.Limits)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	if len(exported.Routing.ModelCards) != 1 {
		t.Fatalf("exported cards = %+v", exported.Routing.ModelCards)
	}
	if got := exported.Routing.ModelCards[0].MaxOutputTokens; got == nil || *got != 16000 {
		t.Fatalf("exported max_output_tokens = %v, want 16000", got)
	}
}

// The catalog's built-in maximum is the model's ceiling, not a limit the
// operator chose for this deployment, so it is never the dispatched value.
func TestCatalogBuiltInMaxOutputIsNotACardLimit(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: production
      catalog: openai/gpt-5.4
      api_format: responses
      backend_refs:
        - name: primary
          provider: openai
routing: {}
`))
	if err != nil {
		t.Fatal(err)
	}
	effective, ok := cfg.EffectiveModelRegistry.Model("production")
	if !ok || effective.Card.Card.Limits.MaxOutputTokens <= 0 {
		t.Fatalf("the built-in card carries no output limit to test against: %+v", effective.Card.Card.Limits)
	}
	if got := cfg.ModelConfig["production"].MaxOutputTokens; got != 0 {
		t.Fatalf("runtime max_output_tokens = %d, want 0 (undeclared)", got)
	}
}

func TestModelCardMaxOutputTokensOnARoutingOnlyCard(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: private-model
routing:
  modelCards:
    - name: private-model
      max_output_tokens: 8000
    - name: silent
`))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.ModelConfig["private-model"].MaxOutputTokens; got != 8000 {
		t.Fatalf("max_output_tokens = %d, want 8000", got)
	}
	if got := cfg.ModelConfig["silent"].MaxOutputTokens; got != 0 {
		t.Fatalf("an undeclared card has max_output_tokens %d", got)
	}
}

func TestNonPositiveModelCardMaxOutputTokensIsRefused(t *testing.T) {
	for _, value := range []string{"0", "-1"} {
		_, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: private-model
routing:
  modelCards:
    - name: private-model
      max_output_tokens: ` + value + `
`))
		if err == nil || !strings.Contains(err.Error(), "routing.modelCards[private-model].max_output_tokens must be positive") {
			t.Fatalf("max_output_tokens: %s returned %v", value, err)
		}
		_, err = ParseRoutingYAMLBytes([]byte(`
routing:
  modelCards:
    - name: private-model
      max_output_tokens: ` + value + `
`))
		if err == nil || !strings.Contains(err.Error(), "max_output_tokens must be positive") {
			t.Fatalf("a routing fragment with max_output_tokens: %s returned %v", value, err)
		}
	}
}

// A LoRA adapter without its own entry takes its base model's card limit; an
// entry of its own wins.
func TestGetModelMaxOutputTokensFallsBackToTheLoRABase(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelConfig = map[string]ModelParams{
		"base":  {MaxOutputTokens: 12000, LoRAs: []LoRAAdapter{{Name: "adapter"}, {Name: "own"}}},
		"own":   {MaxOutputTokens: 900},
		"plain": {},
	}
	for model, want := range map[string]int{"base": 12000, "adapter": 12000, "own": 900, "plain": 0, "unknown": 0} {
		if got := cfg.GetModelMaxOutputTokens(model); got != want {
			t.Fatalf("%s: max_output_tokens = %d, want %d", model, got, want)
		}
	}
}
