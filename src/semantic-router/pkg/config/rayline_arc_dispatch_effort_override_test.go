package config

import (
	"strings"
	"testing"
)

// dispatch_effort_overrides loads on a reasoning worker whose bound actions
// include the native default base (every fixture action is at default).
func TestRaylineARCDispatchEffortOverrideLoads(t *testing.T) {
	cfg, decision := policyV5Decision(t, readPolicyV5Fixture(t))
	decision.Algorithm.RaylineARC.PolicyService.DispatchEffortOverrides = map[string]string{"arm-glm": "max"}
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("the override refused: %v", err)
	}
	if effort, ok := decision.Algorithm.RaylineARC.PolicyService.DispatchEffortOverride("arm-glm"); !ok || effort != "max" {
		t.Fatalf("DispatchEffortOverride = %q, %v", effort, ok)
	}
	if _, ok := decision.Algorithm.RaylineARC.PolicyService.DispatchEffortOverride("arm-opus"); ok {
		t.Fatal("an unlisted worker has an override")
	}
}

func TestRaylineARCDispatchEffortOverrideRefusals(t *testing.T) {
	fixture := readPolicyV5Fixture(t)
	for name, test := range map[string]struct {
		edit func(cfg *RouterConfig, decision *Decision)
		want string
	}{
		"an unbound worker": {
			edit: func(_ *RouterConfig, d *Decision) {
				d.Algorithm.RaylineARC.PolicyService.DispatchEffortOverrides = map[string]string{"arm-nobody": "max"}
			},
			want: "which no binding uses",
		},
		"an unknown effort": {
			edit: func(_ *RouterConfig, d *Decision) {
				d.Algorithm.RaylineARC.PolicyService.DispatchEffortOverrides = map[string]string{"arm-glm": "extreme"}
			},
			want: "is not one of",
		},
		"beside dispatch_effort": {
			edit: func(_ *RouterConfig, d *Decision) {
				policy := d.Algorithm.RaylineARC.PolicyService
				policy.DispatchEffortOverrides = map[string]string{"arm-glm": "max"}
				policy.DispatchEffort = RaylineARCPolicyDispatchEffortProviderDefault
			},
			want: "dispatch_effort serves v4",
		},
	} {
		t.Run(name, func(t *testing.T) {
			cfg, decision := policyV5Decision(t, fixture)
			test.edit(cfg, &decision)
			err := validatePolicyDispatch(cfg, decision)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

// Without a v5 manifest there is no control base to override.
func TestRaylineARCDispatchEffortOverrideNeedsV5(t *testing.T) {
	_, decision := policyV5Decision(t, readPolicyV5Fixture(t))
	policy := decision.Algorithm.RaylineARC.PolicyService
	policy.DispatchEffortOverrides = map[string]string{"arm-glm": "max"}
	policy.PackageManifest = ""
	err := validateRaylineARCPolicyServiceConfig(policy)
	if err == nil || !strings.Contains(err.Error(), "needs a v5 package") {
		t.Fatalf("error = %v, want the v5 refusal", err)
	}
}
