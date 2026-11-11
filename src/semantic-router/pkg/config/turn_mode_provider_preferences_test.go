package config

import (
	"reflect"
	"strings"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

const regularPinYAML = `      provider_preferences:
        order:
          - openai
        only:
          - openai
        max_price:
          completion: 2.55
`

const privacyPinYAML = `      privacy_provider_preferences:
        order:
          - deepinfra
        only:
          - deepinfra
`

const zdrPinYAML = `      zdr_provider_preferences:
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

// Each pin loads with its own values, and each turn mode reads its own: a
// standard turn the regular pin, a privacy turn the privacy override, a zdr
// turn the zdr pin.
func TestTurnModePinsLoadBesideTheRegularPin(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML(regularPinYAML + privacyPinYAML + zdrPinYAML)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	zdr := parsed.ZDRProviderPreferencesForModel("model-a")
	if zdr == nil || !reflect.DeepEqual(zdr.Order, []string{"phala", "novita"}) ||
		zdr.MaxPrice["completion"] != 3.0 || zdr.AllowFallbacks == nil || !*zdr.AllowFallbacks {
		t.Fatalf("zdr_provider_preferences did not reach the model params: %#v", zdr)
	}
	privacy := parsed.PrivacyProviderPreferencesForModel("model-a")
	if privacy == nil || !reflect.DeepEqual(privacy.Only, []string{"deepinfra"}) {
		t.Fatalf("privacy_provider_preferences did not reach the model params: %#v", privacy)
	}
	standard := parsed.ProviderPreferencesForModel("model-a")
	if standard == nil || !reflect.DeepEqual(standard.Order, []string{"openai"}) || standard.MaxPrice["completion"] != 2.55 {
		t.Fatalf("the regular pin changed: %#v", standard)
	}
	for mode, want := range map[string]*OpenRouterProviderPreferences{
		"":                      standard,
		ProviderTurnModePrivacy: privacy,
		ProviderTurnModeZDR:     zdr,
	} {
		if got := parsed.ProviderPreferencesForTurn("model-a", mode); got != want {
			t.Fatalf("a %q turn read %#v, want %#v", mode, got, want)
		}
	}
}

// A privacy (Standard) turn on an arm with only a zdr pin sends the regular
// pin, never the zdr one: the zdr pin is what narrowed Standard to ZDR hosts
// (memex-desktop#8229).
func TestPrivacyTurnNeverReadsTheZDRPin(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML(regularPinYAML + zdrPinYAML)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if parsed.PrivacyProviderPreferencesForModel("model-a") != nil {
		t.Fatal("an arm without privacy_provider_preferences has a privacy override")
	}
	standard := parsed.ProviderPreferencesForModel("model-a")
	if got := parsed.ProviderPreferencesForTurn("model-a", ProviderTurnModePrivacy); got != standard {
		t.Fatalf("a privacy turn read %#v, want the regular pin %#v", got, standard)
	}
	if got := parsed.ProviderPreferencesForTurn("model-a", ProviderTurnModeZDR); got != parsed.ZDRProviderPreferencesForModel("model-a") {
		t.Fatalf("a zdr turn read %#v, want the zdr pin", got)
	}
}

// A zdr turn on an arm with only a privacy override sends the regular pin,
// never the privacy one: the override admits hosts a zdr turn may not use.
func TestZDRTurnNeverReadsThePrivacyPin(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML(regularPinYAML + privacyPinYAML)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	standard := parsed.ProviderPreferencesForModel("model-a")
	if got := parsed.ProviderPreferencesForTurn("model-a", ProviderTurnModeZDR); got != standard {
		t.Fatalf("a zdr turn read %#v, want the regular pin %#v", got, standard)
	}
	if got := parsed.ProviderPreferencesForTurn("model-a", ProviderTurnModePrivacy); got != parsed.PrivacyProviderPreferencesForModel("model-a") {
		t.Fatalf("a privacy turn read %#v, want the privacy override", got)
	}
}

// A mode the cell does not know is served as zdr, the most restrictive.
func TestUnknownTurnModeIsServedAsZDR(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML(regularPinYAML + privacyPinYAML + zdrPinYAML)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if got := parsed.ProviderPreferencesForTurn("model-a", "strict"); got != parsed.ZDRProviderPreferencesForModel("model-a") {
		t.Fatalf("an unknown-mode turn read %#v, want the zdr pin", got)
	}
}

// An arm with neither override sends its regular pin on every turn.
func TestPrivateTurnsFallBackToTheRegularPin(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML(`      provider_preferences:
        order:
          - openai
`)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if parsed.ZDRProviderPreferencesForModel("model-a") != nil || parsed.PrivacyProviderPreferencesForModel("model-a") != nil {
		t.Fatal("an arm without the keys has an override pin")
	}
	for _, mode := range []string{ProviderTurnModePrivacy, ProviderTurnModeZDR} {
		if got := parsed.ProviderPreferencesForTurn("model-a", mode); got != parsed.ProviderPreferencesForModel("model-a") {
			t.Fatalf("a %s turn on an arm without an override read %#v", mode, got)
		}
	}
}

func TestTurnModePinsAreKnownCanonicalKeys(t *testing.T) {
	var raw map[string]interface{}
	if err := yamlv3.Unmarshal([]byte(providerPreferencesConfigYAML(privacyPinYAML+zdrPinYAML)), &raw); err != nil {
		t.Fatalf("unmarshal raw canonical config: %v", err)
	}
	if warnings := collectUnknownFields(raw, reflect.TypeOf(CanonicalConfig{})); len(warnings) != 0 {
		t.Fatalf("privacy/zdr_provider_preferences are not known canonical keys: %v", warnings)
	}
}

// The retired private_provider_preferences key is refused at load, by name,
// and the error points at both keys that replace it. Warning and dropping it
// would silently move zdr turns onto the regular pin.
func TestLegacyPrivateProviderPreferencesIsRefused(t *testing.T) {
	// A null, empty or malformed value is refused as a valid one is: by the
	// key's presence, with the same message.
	for name, value := range map[string]string{
		"a mapping":    "\n        only:\n          - phala\n",
		"null":         " null\n",
		"empty":        "\n",
		"not a pin":    " phala\n",
		"an empty map": " {}\n",
	} {
		for parser, parse := range map[string]func([]byte) (*RouterConfig, error){
			"ParseYAMLBytes": ParseYAMLBytes, "ParseYAMLBytesWithoutEnvExpansion": ParseYAMLBytesWithoutEnvExpansion,
		} {
			t.Run(name+"/"+parser, func(t *testing.T) {
				_, err := parse([]byte(providerPreferencesConfigYAML(regularPinYAML + "      private_provider_preferences:" + value)))
				if err == nil {
					t.Fatal("a config with private_provider_preferences loaded")
				}
				for _, want := range []string{
					"providers.models[model-a].private_provider_preferences is no longer accepted",
					"zdr_provider_preferences", "privacy_provider_preferences",
				} {
					if !strings.Contains(err.Error(), want) {
						t.Fatalf("err = %v, want it to contain %q", err, want)
					}
				}
			})
		}
	}
}

// Each override is held to the provider_preferences rules, and its errors
// name its own key.
func TestTurnModePinsRejectMalformedValues(t *testing.T) {
	cases := map[string]struct {
		body string
		want string
	}{
		"pins nothing": {`
        allow_fallbacks: false
`, "KEY must set order or only"},
		"a blank slug": {`
        order:
          - " "
`, "KEY.order[0] must be a non-empty provider slug"},
		"a max_price kind OpenRouter does not define": {`
        order:
          - phala
        max_price:
          tokens: 1
`, "KEY.max_price.tokens"},
		"a negative max_price": {`
        order:
          - phala
        max_price:
          completion: -1
`, "KEY.max_price.completion"},
		"a data_collection OpenRouter does not define": {`
        order:
          - phala
        data_collection: maybe
`, "KEY.data_collection"},
		"an unknown key": {`
        order:
          - phala
        sort: price
`, `unsupported field "sort"`},
		"not a mapping": {" phala\n", ""},
	}
	for _, key := range []string{"privacy_provider_preferences", "zdr_provider_preferences"} {
		for name, test := range cases {
			t.Run(key+"/"+name, func(t *testing.T) {
				want := strings.ReplaceAll(test.want, "KEY", key)
				_, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML("      " + key + ":" + test.body)))
				if err == nil || !strings.Contains(err.Error(), want) {
					t.Fatalf("malformed %s: err = %v, want it to contain %q", key, err, want)
				}
			})
		}
	}
}
