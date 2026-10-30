package config

import (
	"strings"
	"testing"
)

// standardTwinConfigYAML is a flex arm and its standard twin: the same
// provider model, the flex arm pinned to openai/flex, the twin ignoring it.
// flexExtra and twinPin replace the parts a case varies.
func standardTwinConfigYAML(flexExtra, twinPin string) string {
	return `
version: v0.3
providers:
  defaults:
    model: luna-flex
  models:
    - name: luna-flex
      provider_model_id: openai/gpt-luna
      provider_preferences:
        only:
          - openai/flex
        allow_fallbacks: false
` + flexExtra + `      backend_refs:
        - base_url: https://openrouter.ai/api/v1
          provider: openai
    - name: luna
      provider_model_id: openai/gpt-luna
` + twinPin + `      backend_refs:
        - base_url: https://openrouter.ai/api/v1
          provider: openai
routing:
  modelCards:
    - name: luna-flex
    - name: luna
  decisions:
    - name: default
      rules:
        operator: AND
      modelRefs:
        - model: luna-flex
`
}

const standardTwinIgnorePin = `      provider_preferences:
        order:
          - openai
        ignore:
          - openai/flex
`

// A flex arm that names its twin carries it to the merged params; every other
// arm, the twin included, declares none, so nothing else is re-sent.
func TestStandardTwinReachesTheFlexArmOnly(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(standardTwinConfigYAML(
		"      standard_twin: luna\n", standardTwinIgnorePin,
	)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if twin := parsed.StandardTwinForModel("luna-flex"); twin != "luna" {
		t.Fatalf("the flex arm's twin = %q, want luna", twin)
	}
	if twin := parsed.StandardTwinForModel("luna"); twin != "" {
		t.Fatalf("the twin itself declared a twin: %q", twin)
	}
}

// Default off: a config without the key re-sends nothing.
func TestStandardTwinIsAbsentByDefault(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(standardTwinConfigYAML("", standardTwinIgnorePin)))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if twin := parsed.StandardTwinForModel("luna-flex"); twin != "" {
		t.Fatalf("an arm that declares no twin resolved one: %q", twin)
	}
}

// Each pairing that could re-send to another model, back onto the discount
// endpoint, or along a chain is refused at load.
func TestStandardTwinRefusesUnsafePairings(t *testing.T) {
	cases := map[string]struct {
		flexExtra, twinPin, want string
	}{
		"twin is unknown": {
			"      standard_twin: nobody\n", standardTwinIgnorePin, "is not a providers.models entry",
		},
		"twin is itself": {
			"      standard_twin: luna-flex\n", standardTwinIgnorePin, "must name another",
		},
		"twin admits the flex endpoint": {
			"      standard_twin: luna\n",
			"      provider_preferences:\n        order:\n          - openai\n",
			`must exclude its endpoint "openai/flex"`,
		},
		"twin's private pin admits the flex endpoint": {
			"      standard_twin: luna\n",
			standardTwinIgnorePin + "      private_provider_preferences:\n        only:\n          - openai/flex\n",
			`private-turn pin must exclude the private endpoint "openai/flex"`,
		},
		"twin's private pin admits the discount's private endpoint": {
			"      standard_twin: luna\n      private_provider_preferences:\n        only:\n          - azure\n",
			standardTwinIgnorePin + "      private_provider_preferences:\n        only:\n          - azure\n",
			`private-turn pin must exclude the private endpoint "azure"`,
		},
		"twin pins nothing": {
			"      standard_twin: luna\n", "", "must set provider_preferences",
		},
		"twin lists the flex endpoint in only": {
			"      standard_twin: luna\n",
			"      provider_preferences:\n        only:\n          - openai\n          - openai/flex\n",
			`must exclude its endpoint "openai/flex"`,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(standardTwinConfigYAML(tc.flexExtra, tc.twinPin)))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want it to contain %q", err, tc.want)
			}
		})
	}
}

// A twin serves the same model: a pairing across provider models is refused.
func TestStandardTwinRefusesAnotherModel(t *testing.T) {
	yaml := strings.Replace(
		standardTwinConfigYAML("      standard_twin: luna\n", standardTwinIgnorePin),
		"    - name: luna\n      provider_model_id: openai/gpt-luna\n",
		"    - name: luna\n      provider_model_id: openai/gpt-sol\n", 1,
	)
	_, err := ParseYAMLBytes([]byte(yaml))
	if err == nil || !strings.Contains(err.Error(), "a twin is the same model") {
		t.Fatalf("err = %v, want the same-model refusal", err)
	}
}

// A twin that declares a twin of its own would chain re-sends.
func TestStandardTwinRefusesAChain(t *testing.T) {
	yaml := standardTwinConfigYAML("      standard_twin: luna\n", standardTwinIgnorePin+"      standard_twin: luna-flex\n")
	_, err := ParseYAMLBytes([]byte(yaml))
	if err == nil || !strings.Contains(err.Error(), "declares a twin of its own") {
		t.Fatalf("err = %v, want the chain refusal", err)
	}
}

// The discount arm must state the endpoint it pins, or there is nothing for
// the twin to exclude.
func TestStandardTwinNeedsTheDiscountPin(t *testing.T) {
	yaml := strings.Replace(
		standardTwinConfigYAML("      standard_twin: luna\n", standardTwinIgnorePin),
		"        only:\n          - openai/flex\n", "        order:\n          - openai\n", 1,
	)
	_, err := ParseYAMLBytes([]byte(yaml))
	if err == nil || !strings.Contains(err.Error(), "needs provider_preferences.only") {
		t.Fatalf("err = %v, want the discount-pin refusal", err)
	}
}
