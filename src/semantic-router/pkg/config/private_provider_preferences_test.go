package config

import (
	"reflect"
	"strings"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

const privateProviderPinYAML = `      provider_preferences:
        order:
          - openai
        only:
          - openai
        max_price:
          completion: 2.55
      private_provider_preferences:
        order:
          - phala
          - novita
        only:
          - phala
          - novita
        allow_fallbacks: true
        max_price:
          completion: 3.0
`

// A private pin loads beside the standard one, each with its own values, and
// a private turn is the only turn that reads it.
func TestPrivateProviderPreferencesLoadBesideTheStandardPin(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML(privateProviderPinYAML)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	private := parsed.PrivateProviderPreferencesForModel("model-a")
	if private == nil || !reflect.DeepEqual(private.Order, []string{"phala", "novita"}) ||
		private.MaxPrice["completion"] != 3.0 || private.AllowFallbacks == nil || !*private.AllowFallbacks {
		t.Fatalf("private_provider_preferences did not reach the model params: %#v", private)
	}
	standard := parsed.ProviderPreferencesForModel("model-a")
	if standard == nil || !reflect.DeepEqual(standard.Order, []string{"openai"}) || standard.MaxPrice["completion"] != 2.55 {
		t.Fatalf("the standard pin changed: %#v", standard)
	}
	if got := parsed.ProviderPreferencesForTurn("model-a", false); got != standard {
		t.Fatalf("a standard turn read %#v, want the standard pin", got)
	}
	if got := parsed.ProviderPreferencesForTurn("model-a", true); got != private {
		t.Fatalf("a private turn read %#v, want the private pin", got)
	}
}

// An arm with no private pin sends its standard pin on private turns too.
func TestPrivateTurnFallsBackToTheStandardPin(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML(`      provider_preferences:
        order:
          - openai
`)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if parsed.PrivateProviderPreferencesForModel("model-a") != nil {
		t.Fatal("an arm without the key has a private pin")
	}
	if got := parsed.ProviderPreferencesForTurn("model-a", true); got != parsed.ProviderPreferencesForModel("model-a") {
		t.Fatalf("a private turn on an arm without a private pin read %#v", got)
	}
}

func TestPrivateProviderPreferencesIsAKnownCanonicalKey(t *testing.T) {
	var raw map[string]interface{}
	if err := yamlv3.Unmarshal([]byte(providerPreferencesConfigYAML(privateProviderPinYAML)), &raw); err != nil {
		t.Fatalf("unmarshal raw canonical config: %v", err)
	}
	if warnings := collectUnknownFields(raw, reflect.TypeOf(CanonicalConfig{})); len(warnings) != 0 {
		t.Fatalf("private_provider_preferences is not a known canonical key: %v", warnings)
	}
}

// The private pin is held to the provider_preferences rules, and its errors
// name its own key.
func TestPrivateProviderPreferencesRejectsMalformedValues(t *testing.T) {
	tests := map[string]struct {
		yaml string
		want string
	}{
		"pins nothing": {`      private_provider_preferences:
        allow_fallbacks: false
`, "private_provider_preferences must set order or only"},
		"a blank slug": {`      private_provider_preferences:
        order:
          - " "
`, "private_provider_preferences.order[0] must be a non-empty provider slug"},
		"a max_price kind OpenRouter does not define": {`      private_provider_preferences:
        order:
          - phala
        max_price:
          tokens: 1
`, "private_provider_preferences.max_price.tokens"},
		"a negative max_price": {`      private_provider_preferences:
        order:
          - phala
        max_price:
          completion: -1
`, "private_provider_preferences.max_price.completion"},
		"a data_collection OpenRouter does not define": {`      private_provider_preferences:
        order:
          - phala
        data_collection: maybe
`, "private_provider_preferences.data_collection"},
		"an unknown key": {`      private_provider_preferences:
        order:
          - phala
        sort: price
`, `unsupported field "sort"`},
		"not a mapping": {`      private_provider_preferences: phala
`, ""},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML(test.yaml)))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("malformed private_provider_preferences: err = %v, want it to contain %q", err, test.want)
			}
		})
	}
}
