package config

import (
	"strings"
	"testing"
)

func pricingSnapshotConfig(prompt float64) *RouterConfig {
	write := 3.75
	return &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
		"a": {Pricing: ModelPricing{Currency: "USD", PromptPer1M: prompt, CompletionPer1M: 15, CacheWritePer1M: &write}},
		"b": {Pricing: ModelPricing{PromptPer1M: 1}, ExternalModelIDs: map[string]string{"x": "b-ext", "y": "b-two"}},
		"c": {LoRAs: []LoRAAdapter{{Name: "c-lora-2"}, {Name: "c-lora-1"}}},
	}}}
}

func TestPricingSnapshotIDIsStableAndNamesTheRates(t *testing.T) {
	first := pricingSnapshotConfig(3).PricingSnapshotID()
	if !strings.HasPrefix(first, PricingSnapshotPrefix) || len(first) != len(PricingSnapshotPrefix)+64 {
		t.Fatalf("snapshot id = %q", first)
	}
	// Map iteration order differs between calls; the id must not.
	for range 20 {
		if again := pricingSnapshotConfig(3).PricingSnapshotID(); again != first {
			t.Fatalf("snapshot id changed between equal tables: %s vs %s", first, again)
		}
	}
	if changed := pricingSnapshotConfig(3.01).PricingSnapshotID(); changed == first {
		t.Fatal("a changed rate kept the same snapshot id")
	}
	unaliased := pricingSnapshotConfig(3)
	params := unaliased.ModelConfig["b"]
	params.ExternalModelIDs = map[string]string{"x": "b-ext"}
	unaliased.ModelConfig["b"] = params
	if unaliased.PricingSnapshotID() == first {
		t.Fatal("a changed alias, which changes what a lookup prices, kept the same snapshot id")
	}
}

func TestPricingSnapshotIDExistsWithoutATable(t *testing.T) {
	var missing *RouterConfig
	if missing.PricingSnapshotID() != (&RouterConfig{}).PricingSnapshotID() {
		t.Fatal("a nil config and an empty one price nothing alike and must share a snapshot")
	}
}
