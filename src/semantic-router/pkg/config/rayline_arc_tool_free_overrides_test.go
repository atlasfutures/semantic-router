package config

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

// toolFreeStageOne is a stage_one part with presets, bounds, two offered
// models and this tool_free_overrides value ("" omits it).
func toolFreeStageOne(overrides string) string {
	part := `{"offered": [{"model": "glm-5.3-flash", "effort": null}, {"model": "kimi-k3", "effort": null}],` +
		` "operating_bounds": {"cost_weight": [0.2, 10.0]}, "presets": {"swift": 10.0, "sage": 0.25}`
	if overrides != "" {
		part += `, "tool_free_overrides": ` + overrides
	}
	return part + `}`
}

// A declared override is read per tier: the tier's preset, the cost weight a
// tool-free request decides at (preset / v_multiplier), the excluded models
// and the latency weight. Absent overrides leave every tier without one.
func TestParseStageOneToolFreeOverrides(t *testing.T) {
	got, err := parseStageOneToolFreeOverrides(json.RawMessage(toolFreeStageOne(
		`{"sage": {"v_multiplier": 0.5, "excluded_models": ["kimi-k3"], "latency_weight": 0.05}, "swift": {"v_multiplier": 2}}`)))
	if err != nil {
		t.Fatal(err)
	}
	sage, swift := got["sage"], got["swift"]
	if sage.PresetCostWeight != 0.25 || sage.CostWeight != 0.5 || !slices.Equal(sage.ExcludedModels, []string{"kimi-k3"}) ||
		sage.LatencyWeight != 0.05 || sage.VMultiplier != 0.5 {
		t.Fatalf("sage %+v", sage)
	}
	if swift.CostWeight != 5 || len(swift.ExcludedModels) != 0 || swift.LatencyWeight != 0 {
		t.Fatalf("swift %+v", swift)
	}
	for _, absent := range []string{toolFreeStageOne(""), toolFreeStageOne("null"), ""} {
		raw := json.RawMessage(absent)
		if absent == "" {
			raw = nil
		}
		if got, err := parseStageOneToolFreeOverrides(raw); err != nil || got != nil {
			t.Fatalf("absent %q: %v, %v", absent, got, err)
		}
	}
}

// A malformed override is refused, as pathfinder's contract refuses it.
func TestParseStageOneToolFreeOverridesRefusesMalformed(t *testing.T) {
	for overrides, want := range map[string]string{
		`[]`:                             "not an object",
		`{}`:                             "names no tier",
		`{"gold": {"v_multiplier": 1}}`:  "presets does not",
		`{"sage": {"v_multiplier": 0}}`:  "v_multiplier",
		`{"sage": {"v_multiplier": -1}}`: "v_multiplier",
		`{"sage": {}}`:                   "v_multiplier",
		`{"sage": {"v_multiplier": 1, "latency_weight": -0.1}}`:                          "latency_weight",
		`{"sage": {"v_multiplier": 1, "excluded_models": ["gpt-x"]}}`:                    "does not offer",
		`{"sage": {"v_multiplier": 1, "excluded_models": ["kimi-k3", "kimi-k3"]}}`:       "unique",
		`{"sage": {"v_multiplier": 1, "excluded_models": [" "]}}`:                        "unique",
		`{"sage": {"v_multiplier": 1, "excluded_models": ["kimi-k3", "glm-5.3-flash"]}}`: "every offered",
		`{"sage": {"v_multiplier": 1, "speed": 2}}`:                                      "not an override object",
		`{"sage": {"v_multiplier": 1, "latency_weight": null}}`:                          "latency_weight",
		`{"sage": {"v_multiplier": 1, "excluded_models": null}}`:                         "excluded_models",
		`{"sage": null}`: "not an override object",
	} {
		_, err := parseStageOneToolFreeOverrides(json.RawMessage(toolFreeStageOne(overrides)))
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: err = %v, want %q", overrides, err, want)
		}
	}
}

// The package reads its overrides at load: a malformed one refuses the
// package at startup, and a valid one is served per tier.
func TestRaylineARCPolicyPackageV5ToolFreeOverridesAtStartup(t *testing.T) {
	fixture := readPolicyV5Fixture(t)
	withPart := func(part string) []byte {
		return bytes.Replace(fixture, []byte("{"), []byte(`{"stage_one": `+part+`,`), 1)
	}
	cfg, decision := policyV5Decision(t, withPart(toolFreeStageOne(`{"sage": {"v_multiplier": 0.5, "excluded_models": ["kimi-k3"]}}`)))
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("valid overrides refused: %v", err)
	}
	policy := decision.Algorithm.RaylineARC.PolicyService
	if override, ok := policy.PackageToolFreeOverride("sage"); !ok || override.CostWeight != 0.5 {
		t.Fatalf("sage override %+v, %v", override, ok)
	}
	if _, ok := policy.PackageToolFreeOverride("swift"); ok {
		t.Fatal("swift declares no override")
	}
	cfg, decision = policyV5Decision(t, withPart(toolFreeStageOne(`{"gold": {"v_multiplier": 1}}`)))
	if err := validatePolicyDispatch(cfg, decision); err == nil || !strings.Contains(err.Error(), "tool_free_overrides") {
		t.Fatalf("malformed overrides: err = %v", err)
	}
	_, decision = policyV5Decision(t, withPart(toolFreeStageOne("")))
	if _, ok := decision.Algorithm.RaylineARC.PolicyService.PackageToolFreeOverride("sage"); ok {
		t.Fatal("a package without overrides serves one")
	}
}
