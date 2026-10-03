package config

import (
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

// fallback parses as the CLI schema writes it, defaults to off with a 30 s
// route exclusion, and refuses a time outside [0, 3600].
func TestRaylineARCPolicyFallbackContract(t *testing.T) {
	var policy RaylineARCPolicyServiceConfig
	if err := yaml.Unmarshal([]byte("fallback:\n  enabled: true\n  cell_exclusion_seconds: 90\n"), &policy); err != nil {
		t.Fatal(err)
	}
	if !policy.FallbackEnabled() || policy.CellExclusionTTL() != 90*time.Second {
		t.Fatalf("parsed fallback = %+v", policy.Fallback)
	}
	var unset RaylineARCPolicyServiceConfig
	if unset.FallbackEnabled() || unset.CellExclusionTTL() != DefaultRaylineARCCellExclusionSeconds*time.Second {
		t.Fatal("an unset fallback is not off with the default exclusion")
	}
	for seconds, valid := range map[int]bool{-1: false, 0: true, 3600: true, 3601: false} {
		_, decision := policyDispatchFixture()
		decision.Algorithm.RaylineARC.PolicyService.Fallback = &RaylineARCPolicyFallbackConfig{Enabled: true, CellExclusionSeconds: seconds}
		if err := validateRaylineARCPolicyBindings(decision); (err == nil) != valid {
			t.Fatalf("cell_exclusion_seconds %d: %v", seconds, err)
		}
	}
}
