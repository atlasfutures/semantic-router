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
	cfg, decision := policyV5Decision(t, readPolicyV6Fixture(t))
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
