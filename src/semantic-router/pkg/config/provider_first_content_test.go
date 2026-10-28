package config

import (
	"reflect"
	"strings"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

// An arm's first-content limit is a provider binding beside its pin: it
// replaces the global response_stream.first_content_sec for that arm only.
func TestArmFirstContentSecOverridesTheGlobalLimit(t *testing.T) {
	source := providerPreferencesConfigYAML("      first_content_sec: 30\n") + `global:
  router:
    response_stream:
      first_content_sec: 90
      host_retry: true
`
	parsed, err := ParseYAMLBytes([]byte(source))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if got := parsed.FirstContentSecForModel("model-a"); got != 30 {
		t.Fatalf("arm limit = %d, want 30", got)
	}
	if got := parsed.FirstContentSecForModel("another-model"); got != 90 {
		t.Fatalf("an arm without its own limit has %d, want the global 90", got)
	}
	if !parsed.ResponseStreamHostRetry {
		t.Fatal("response_stream.host_retry did not load")
	}
	var raw map[string]interface{}
	if err := yamlv3.Unmarshal([]byte(source), &raw); err != nil {
		t.Fatalf("unmarshal raw canonical config: %v", err)
	}
	if warnings := collectUnknownFields(raw, reflect.TypeOf(CanonicalConfig{})); len(warnings) != 0 {
		t.Fatalf("the keys are not known canonical keys: %v", warnings)
	}
}

func TestHostRetryIsOffAndArmLimitAbsentByDefault(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML("")))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if parsed.ResponseStreamHostRetry || parsed.FirstContentSecForModel("model-a") != 0 {
		t.Fatalf("host_retry %v, arm limit %d", parsed.ResponseStreamHostRetry, parsed.FirstContentSecForModel("model-a"))
	}
}

func TestNegativeArmFirstContentSecIsRefused(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(providerPreferencesConfigYAML("      first_content_sec: -1\n")))
	if err == nil || !strings.Contains(err.Error(), "providers.models[model-a].first_content_sec") {
		t.Fatalf("err = %v", err)
	}
}
