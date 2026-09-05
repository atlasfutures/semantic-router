package config

import (
	"reflect"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

func providerPreferencesConfigYAML(preferences string) string {
	return `
version: v0.3
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
` + preferences + `      backend_refs:
        - base_url: https://openrouter.ai/api/v1
          provider: openai
routing:
  modelCards:
    - name: model-a
  decisions:
    - name: default
      rules:
        operator: AND
      modelRefs:
        - model: model-a
`
}

// A pinned arm names the providers OpenRouter may serve it from. The key is a
// provider access binding, so it sits beside the arm's backend_refs rather
// than on the routing model card.
func TestProviderPreferencesRoundTripCanonicalConfig(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML(
		`      provider_preferences:
        order:
          - deepinfra
          - together
        allow_fallbacks: false
        only:
          - deepinfra
          - together
        ignore:
          - novita
        require_parameters: true
        data_collection: deny
`)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}

	preferences := parsed.ModelConfig["model-a"].ProviderPreferences
	if preferences == nil {
		t.Fatal("provider_preferences did not reach the merged model params")
	}
	if !reflect.DeepEqual(preferences.Order, []string{"deepinfra", "together"}) {
		t.Fatalf("order did not normalize: %#v", preferences.Order)
	}
	if preferences.AllowFallbacks == nil || *preferences.AllowFallbacks {
		t.Fatalf("allow_fallbacks did not normalize: %#v", preferences.AllowFallbacks)
	}
	if !reflect.DeepEqual(preferences.Only, []string{"deepinfra", "together"}) {
		t.Fatalf("only did not normalize: %#v", preferences.Only)
	}
	if !reflect.DeepEqual(preferences.Ignore, []string{"novita"}) {
		t.Fatalf("ignore did not normalize: %#v", preferences.Ignore)
	}
	if preferences.RequireParameters == nil || !*preferences.RequireParameters {
		t.Fatalf("require_parameters did not normalize: %#v", preferences.RequireParameters)
	}
	if preferences.DataCollection != ProviderDataCollectionDeny {
		t.Fatalf("data_collection did not normalize: %q", preferences.DataCollection)
	}
}

// An arm that says nothing about providers keeps the bytes it had before the
// key existed, so the absent case must stay absent rather than become empty.
func TestProviderPreferencesAreAbsentWhenUnset(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML("")))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if preferences := parsed.ModelConfig["model-a"].ProviderPreferences; preferences != nil {
		t.Fatalf("an unpinned arm carries preferences: %#v", preferences)
	}
}

// The unknown-key validator rejects any YAML key without a struct field, so
// the key has to be part of the declared provider-model schema.
func TestProviderPreferencesIsAKnownCanonicalKey(t *testing.T) {
	var raw map[string]interface{}
	if err := yamlv3.Unmarshal([]byte(providerPreferencesConfigYAML(
		`      provider_preferences:
        order:
          - deepinfra
        allow_fallbacks: false
`)), &raw); err != nil {
		t.Fatalf("unmarshal raw canonical config: %v", err)
	}
	if warnings := collectUnknownFields(raw, reflect.TypeOf(CanonicalConfig{})); len(warnings) != 0 {
		t.Fatalf("provider_preferences is not a known canonical key: %v", warnings)
	}
}

func TestProviderPreferencesRejectsUnusableValues(t *testing.T) {
	falseValue := false
	tests := []struct {
		name        string
		preferences OpenRouterProviderPreferences
	}{
		{
			// A blank slug pins nothing and OpenRouter would refuse the body.
			name:        "a blank slug in order",
			preferences: OpenRouterProviderPreferences{Order: []string{"deepinfra", "  "}},
		},
		{
			name:        "a blank slug in only",
			preferences: OpenRouterProviderPreferences{Only: []string{""}},
		},
		{
			name:        "a blank slug in ignore",
			preferences: OpenRouterProviderPreferences{Order: []string{"deepinfra"}, Ignore: []string{""}},
		},
		{
			// The key is present, so it must say which providers may serve
			// the arm. allow_fallbacks alone would refuse every provider.
			name:        "neither order nor only",
			preferences: OpenRouterProviderPreferences{AllowFallbacks: &falseValue},
		},
		{
			name:        "a data_collection OpenRouter does not define",
			preferences: OpenRouterProviderPreferences{Order: []string{"deepinfra"}, DataCollection: "maybe"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := validateProviderPreferences("model-a", &test.preferences); err == nil {
				t.Fatalf("unusable provider preferences were accepted: %#v", test.preferences)
			}
		})
	}
}

func TestProviderPreferencesAcceptsBothDataCollectionValues(t *testing.T) {
	for _, value := range []string{ProviderDataCollectionAllow, ProviderDataCollectionDeny} {
		preferences := OpenRouterProviderPreferences{Order: []string{"deepinfra"}, DataCollection: value}
		if err := validateProviderPreferences("model-a", &preferences); err != nil {
			t.Fatalf("data_collection %q was rejected: %v", value, err)
		}
	}
}

// A config that pins an arm to a provider that cannot serve it is an operator
// error the loader has to name, not one the upstream discovers at dispatch.
func TestProviderPreferencesRejectedByCanonicalLoad(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML(
		`      provider_preferences:
        allow_fallbacks: false
`)))
	if err == nil {
		t.Fatal("canonical load accepted provider_preferences that pins nothing")
	}
}
