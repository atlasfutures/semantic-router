package config

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// pathfinder's v6 fixture (da1fbb6608): the v5 canonical_v1 fixture with the
// canonical_v2 image encoding profile (ADR 0122), so the same actions.
const policyV6Fixture = "../selection/raylinearc/testdata/policy_service/package_manifest.v6.canonical_v2.json"

func readPolicyV6Fixture(t *testing.T) []byte {
	t.Helper()
	raw, err := os.ReadFile(policyV6Fixture)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"schema_version": "rayline.arc-policy-package.v6"`)) {
		t.Fatal("the v6 fixture is no longer a v6 package")
	}
	return raw
}

// A v6 package readies on v5's path with v5-shaped bindings ({action_id,
// worker}), its actions resolve as v5's do, and dispatch reads it as a v5
// package.
func TestRaylineARCPolicyPackageV6Loads(t *testing.T) {
	cfg, decision := policyV6Decision(t)
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("the v6 fixture refused: %v", err)
	}
	policy := decision.Algorithm.RaylineARC.PolicyService
	if !policy.IsPackageV5() {
		t.Fatal("a v6 package does not take v5's dispatch path")
	}
	action, ok := policy.PackageV5Action(policyV5GLMUp)
	if !ok || action.Model != "glm-5.3-flash" || action.Control.Instruction == nil || action.Control.Instruction.Level != "up" {
		t.Fatalf("PackageV5Action = %+v, %v", action, ok)
	}
}

// v6 readiness refuses what v5's refuses: v4 binding fields, v4-only
// settings, and a schedule the package does not declare.
func TestRaylineARCPolicyPackageV6Refusals(t *testing.T) {
	fixture := readPolicyV6Fixture(t)
	for name, test := range map[string]struct {
		edit func(decision *Decision)
		want string
	}{
		"a binding that declares a level": {
			edit: func(d *Decision) { d.Algorithm.RaylineARC.PolicyService.Bindings[2].Level = "up" },
			want: "only action_id and worker",
		},
		"a binding that declares a model": {
			edit: func(d *Decision) { d.Algorithm.RaylineARC.PolicyService.Bindings[0].Model = "anthropic/claude-opus-5" },
			want: "only action_id and worker",
		},
		"a binding that declares an effort": {
			edit: func(d *Decision) {
				effort := "high"
				d.Algorithm.RaylineARC.PolicyService.Bindings[0].Effort = &effort
			},
			want: "only action_id and worker",
		},
		"a binding that declares reasoning_max_tokens": {
			edit: func(d *Decision) {
				budget := int64(1024)
				d.Algorithm.RaylineARC.PolicyService.Bindings[0].ReasoningMaxTokens = &budget
			},
			want: "only action_id and worker",
		},
		"dispatch_effort": {
			edit: func(d *Decision) {
				d.Algorithm.RaylineARC.PolicyService.DispatchEffort = RaylineARCPolicyDispatchEffortProviderDefault
			},
			want: "dispatch_effort serves v4",
		},
		"a model_schedule the package does not declare": {
			edit: func(d *Decision) { d.Algorithm.RaylineARC.PolicyService.ModelSchedule = "" },
			want: "the package declares",
		},
		"no trained_models": {
			edit: func(d *Decision) { d.Algorithm.RaylineARC.PolicyService.TrainedModels = nil },
			want: "trained_models",
		},
	} {
		t.Run(name, func(t *testing.T) {
			cfg, decision := policyV5Decision(t, fixture)
			test.edit(&decision)
			if err := validatePolicyDispatch(cfg, decision); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err = %v, want %q", err, test.want)
			}
		})
	}
}

// A v6 package's harness_shell loads through startup validation when absent
// (exclude), include or include_v2 (pathfinder#3653), and any other value is
// refused at startup.
func TestRaylineARCPolicyPackageV6HarnessShellAtStartup(t *testing.T) {
	const anchor = `"tool_definitions": "include_recorded",`
	fixture := readPolicyV6Fixture(t)
	if !bytes.Contains(fixture, []byte(anchor)) {
		t.Fatalf("the v6 fixture no longer holds %q", anchor)
	}
	for member, accepted := range map[string]bool{
		``:                               true,
		`"harness_shell": "include",`:    true,
		`"harness_shell": "include_v2",`: true,
		`"harness_shell": "include_v3",`: false,
	} {
		manifest := bytes.Replace(fixture, []byte(anchor), []byte(anchor+" "+member), 1)
		cfg, decision := policyV5Decision(t, manifest)
		setPolicyVision(cfg, "arm-opus", true)
		setPolicyVision(cfg, "arm-glm", false)
		err := validatePolicyDispatch(cfg, decision)
		if accepted && err != nil {
			t.Errorf("%q: refused: %v", member, err)
		}
		if !accepted && (err == nil || !strings.Contains(err.Error(), "encoding_profile.harness_shell")) {
			t.Errorf("%q: err = %v, want a harness_shell refusal", member, err)
		}
	}
}

// A package_manifest is v5 or v6; any other schema, and a v6 manifest with a
// text profile, is refused at startup.
func TestRaylineARCPolicyPackageManifestSchemas(t *testing.T) {
	fixture := readPolicyV6Fixture(t)
	for name, edit := range map[string]func(manifest map[string]any){
		"schema v7": func(m map[string]any) { m["schema_version"] = "rayline.arc-policy-package.v7" },
		"schema v4": func(m map[string]any) { m["schema_version"] = "rayline.arc-policy-package.v4" },
		"a text profile": func(m map[string]any) {
			profile := m["encoding_profile"].(map[string]any)
			profile["conversation"] = "canonical_v1"
		},
	} {
		t.Run(name, func(t *testing.T) {
			var manifest map[string]any
			if err := json.Unmarshal(fixture, &manifest); err != nil {
				t.Fatal(err)
			}
			edit(manifest)
			raw, err := json.Marshal(manifest)
			if err != nil {
				t.Fatal(err)
			}
			cfg, decision := policyV5Decision(t, raw)
			if err := validatePolicyDispatch(cfg, decision); err == nil {
				t.Fatal("loaded")
			}
		})
	}
}

// Every v5 fixture still readies unchanged beside v6.
func TestRaylineARCPolicyPackageV5FixturesStillLoadBesideV6(t *testing.T) {
	for _, path := range []string{policyV5Fixture, policyV5Canonical, policyV5Yarn} {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		cfg, decision := policyV5Decision(t, raw)
		if err := validatePolicyDispatch(cfg, decision); err != nil {
			t.Errorf("%s refused: %v", path, err)
		}
	}
}

// A package manifest names its own fallback, so derived_hold_model beside
// one is refused.
func TestRaylineARCDerivedHoldModelIsRefusedWithAManifest(t *testing.T) {
	cfg, decision := policyV5Decision(t, readPolicyV6Fixture(t))
	decision.Algorithm.RaylineARC.PolicyService.DerivedHoldModel = "claude-opus-5"
	if err := validatePolicyDispatch(cfg, decision); err == nil || !strings.Contains(err.Error(), "names its own fallback") {
		t.Fatalf("err = %v", err)
	}
}

// policyV6Decision is the v6 fixture's decision with every bound arm's card
// stating vision, as a v6 package requires: Opus reads images, GLM does not.
func policyV6Decision(t *testing.T) (*RouterConfig, Decision) {
	t.Helper()
	cfg, decision := policyV5Decision(t, readPolicyV6Fixture(t))
	setPolicyVision(cfg, "arm-opus", true)
	setPolicyVision(cfg, "arm-glm", false)
	return cfg, decision
}

func setPolicyVision(cfg *RouterConfig, arm string, vision bool) {
	params := cfg.ModelConfig[arm]
	params.Vision = &vision
	cfg.ModelConfig[arm] = params
}

// A v6 package's encoder reads images, and an image turn leaves the offer
// only on arms whose card says vision: false; an unmarked card counts as
// vision-capable (#215). So every bound arm of a v6 package must state
// vision, true or false, and startup names the arms that do not.
func TestRaylineARCPolicyPackageV6RequiresExplicitVision(t *testing.T) {
	cfg, decision := policyV5Decision(t, readPolicyV6Fixture(t))
	err := validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "arm-glm") || !strings.Contains(err.Error(), "arm-opus") ||
		!strings.Contains(err.Error(), "vision") {
		t.Fatalf("both arms unmarked: err = %v, want both named", err)
	}
	setPolicyVision(cfg, "arm-opus", true)
	err = validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "arm-glm") || strings.Contains(err.Error(), "arm-opus") {
		t.Fatalf("one arm unmarked: err = %v, want only arm-glm named", err)
	}
	for _, glm := range []bool{false, true} {
		setPolicyVision(cfg, "arm-glm", glm)
		if err := validatePolicyDispatch(cfg, decision); err != nil {
			t.Fatalf("every arm marked (glm vision %v): %v", glm, err)
		}
	}
}

// v5 keeps the default: an unmarked card is vision-capable, and loads.
func TestRaylineARCPolicyPackageV5LoadsWithUnmarkedVision(t *testing.T) {
	for _, path := range []string{policyV5Fixture, policyV5Canonical} {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		cfg, decision := policyV5Decision(t, raw)
		for arm, params := range cfg.ModelConfig {
			if params.Vision != nil {
				t.Fatalf("%s: arm %s is marked", path, arm)
			}
		}
		if err := validatePolicyDispatch(cfg, decision); err != nil {
			t.Errorf("%s refused: %v", path, err)
		}
	}
}
