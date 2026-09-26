package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func validThinkingLeverConfig() *RaylineARCThinkingLeverConfig {
	return &RaylineARCThinkingLeverConfig{
		Enabled: true,
		Source:  RaylineARCThinkingSourceRule,
		Level:   "down",
		Workers: map[string]RaylineARCThinkingBindingConfig{
			"z-ai/glm-5.3-flash@default": {
				Admission:  RaylineARCThinkingAdmissionCertified,
				Lever:      "prompt_steering_suffix",
				Emit:       "on_change",
				Placements: []string{"append_tail_user_text", "insert_user_after_tool_run"},
				Levels: []RaylineARCThinkingLevelConfig{
					{Level: "none", Rank: 0},
					{Level: "down", Rank: -1, Suffix: "Until the next steering instruction, use minimal deliberation."},
					{Level: "up", Rank: 1, Suffix: "Until the next steering instruction, reason more thoroughly."},
				},
			},
		},
	}
}

func validateThinkingLeverDecision(t *testing.T, lever *RaylineARCThinkingLeverConfig) error {
	t.Helper()
	decision := validRaylineARCDecision()
	decision.Algorithm.RaylineARC.ThinkingLever = lever
	return validateDecisionAlgorithmConfig(decision.Name, decision.ModelRefs, decision.Algorithm)
}

func TestRaylineARCThinkingLeverIsOptional(t *testing.T) {
	if err := validateThinkingLeverDecision(t, nil); err != nil {
		t.Fatalf("absent block: %v", err)
	}
	// A disabled block is not validated, so an operator can stage bindings
	// before switching the lever on.
	if err := validateThinkingLeverDecision(t, &RaylineARCThinkingLeverConfig{Source: "anything"}); err != nil {
		t.Fatalf("disabled block: %v", err)
	}
	if err := validateThinkingLeverDecision(t, validThinkingLeverConfig()); err != nil {
		t.Fatalf("valid block: %v", err)
	}
}

func TestRaylineARCThinkingLeverRejectsWhatThePlannerWould(t *testing.T) {
	const worker = "z-ai/glm-5.3-flash@default"
	cases := []struct {
		name    string
		mutate  func(*RaylineARCThinkingLeverConfig)
		wantErr string
	}{
		{"unserved source", func(c *RaylineARCThinkingLeverConfig) { c.Source = "artifact" }, "is not served"},
		{"no level", func(c *RaylineARCThinkingLeverConfig) { c.Level = "" }, "level is required"},
		{"no workers", func(c *RaylineARCThinkingLeverConfig) { c.Workers = nil }, "worker binding"},
		{"unknown admission", func(c *RaylineARCThinkingLeverConfig) { c.Admission = "maybe" }, "admission"},
		{"level missing from a binding", func(c *RaylineARCThinkingLeverConfig) { c.Level = "max" }, "not in the binding"},
		{"experimental binding under certified admission", func(c *RaylineARCThinkingLeverConfig) {
			binding := c.Workers[worker]
			binding.Admission = RaylineARCThinkingAdmissionExperimental
			c.Workers[worker] = binding
		}, "experimental"},
		{"placement of the other lever", func(c *RaylineARCThinkingLeverConfig) {
			binding := c.Workers[worker]
			binding.Placements = []string{"system_before_governed_turn"}
			c.Workers[worker] = binding
		}, "does not fit"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			lever := validThinkingLeverConfig()
			test.mutate(lever)
			err := validateThinkingLeverDecision(t, lever)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestRaylineARCThinkingLeverAdmitsExperimentalWhenAsked(t *testing.T) {
	lever := validThinkingLeverConfig()
	lever.Admission = RaylineARCThinkingAdmissionExperimental
	binding := lever.Workers["z-ai/glm-5.3-flash@default"]
	binding.Admission = RaylineARCThinkingAdmissionExperimental
	lever.Workers["z-ai/glm-5.3-flash@default"] = binding
	if err := validateThinkingLeverDecision(t, lever); err != nil {
		t.Fatalf("experimental admission: %v", err)
	}
}

func TestRaylineARCThinkingLeverRoundTripsStrictly(t *testing.T) {
	algorithm := validRaylineARCDecision().Algorithm
	algorithm.RaylineARC.ThinkingLever = validThinkingLeverConfig()
	encoded, err := yaml.Marshal(algorithm)
	if err != nil {
		t.Fatal(err)
	}
	var decoded AlgorithmConfig
	if err := yaml.UnmarshalStrict(encoded, &decoded); err != nil {
		t.Fatalf("strict decode: %v", err)
	}
	if !reflect.DeepEqual(algorithm.RaylineARC.ThinkingLever, decoded.RaylineARC.ThinkingLever) {
		t.Fatalf("thinking_lever did not round-trip:\n%s", encoded)
	}
}
