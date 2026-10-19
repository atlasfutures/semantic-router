package config

import (
	"reflect"
	"strings"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

// An arm's service tier is a provider access binding beside its pin: it
// loads into the merged model params and is a known canonical key.
func TestProviderServiceTierLoadsIntoModelParams(t *testing.T) {
	source := providerPreferencesConfigYAML("      service_tier: default\n")
	parsed, err := ParseYAMLBytes([]byte(source))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if got := parsed.ServiceTierForModel("model-a"); got != "default" {
		t.Fatalf("service tier = %q, want default", got)
	}
	var raw map[string]interface{}
	if err := yamlv3.Unmarshal([]byte(source), &raw); err != nil {
		t.Fatalf("unmarshal raw canonical config: %v", err)
	}
	if warnings := collectUnknownFields(raw, reflect.TypeOf(CanonicalConfig{})); len(warnings) != 0 {
		t.Fatalf("service_tier is not a known canonical key: %v", warnings)
	}
}

func TestProviderServiceTierIsAbsentWhenUnset(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML("")))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if got := parsed.ServiceTierForModel("model-a"); got != "" {
		t.Fatalf("an arm without the key has tier %q", got)
	}
}

// The loader takes only the tiers OpenRouter takes.
func TestProviderServiceTierValidatesTheEnum(t *testing.T) {
	for _, tier := range []string{"default", "flex", "priority", "fast", "ultrafast"} {
		if _, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML("      service_tier: " + tier + "\n"))); err != nil {
			t.Fatalf("tier %q was refused: %v", tier, err)
		}
	}
	for _, tier := range []string{"auto", "scale", "Default", "standard"} {
		_, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML("      service_tier: " + tier + "\n")))
		if err == nil || !strings.Contains(err.Error(), "providers.models[model-a].service_tier") {
			t.Fatalf("tier %q was accepted or refused for another reason: %v", tier, err)
		}
	}
}
