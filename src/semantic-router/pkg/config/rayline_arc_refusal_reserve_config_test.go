package config

import (
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

// reserve parses, is refused on an eval cell or without the fallback, and
// keeps its bounds; family_exclusion keeps its own.
func TestRaylineARCRefusalReserveContract(t *testing.T) {
	var policy RaylineARCPolicyServiceConfig
	source := "fallback:\n  enabled: true\n  family_exclusion: true\n  reserve:\n    enabled: true\n    max_reserves: 1\n    hold_families: [claude, gpt]\n"
	if err := yaml.Unmarshal([]byte(source), &policy); err != nil {
		t.Fatal(err)
	}
	maxReserves, budget, hold, families := policy.RefusalReserveLimits()
	if !policy.RefusalReserveEnabled() || !policy.FamilyExclusionEnabled() || maxReserves != 1 ||
		budget != 45*time.Second || hold != 60*time.Second || len(families) != 2 || policy.FamilyExclusionTTL() != 24*time.Hour {
		t.Fatalf("parsed reserve = %+v", policy.Fallback.Reserve)
	}
	for name, tc := range map[string]struct {
		mutate func(*RaylineARCPolicyServiceConfig)
		valid  bool
	}{
		"on":                   {func(*RaylineARCPolicyServiceConfig) {}, true},
		"on an eval cell":      {func(p *RaylineARCPolicyServiceConfig) { p.EvalCell = true }, false},
		"without the fallback": {func(p *RaylineARCPolicyServiceConfig) { p.Fallback.Enabled = false }, false},
		"three re-serves":      {func(p *RaylineARCPolicyServiceConfig) { p.Fallback.Reserve.MaxReserves = 3 }, false},
		"a glm hold":           {func(p *RaylineARCPolicyServiceConfig) { p.Fallback.Reserve.HoldFamilies = []string{"glm"} }, false},
		"budget past 120 s":    {func(p *RaylineARCPolicyServiceConfig) { p.Fallback.Reserve.BudgetSeconds = 121 }, false},
		"a week of exclusion":  {func(p *RaylineARCPolicyServiceConfig) { p.Fallback.FamilyExclusionHours = 169 }, false},
	} {
		_, decision := policyDispatchFixture()
		policy := decision.Algorithm.RaylineARC.PolicyService
		policy.Fallback = &RaylineARCPolicyFallbackConfig{Enabled: true, Reserve: &RaylineARCRefusalReserveConfig{Enabled: true}}
		tc.mutate(policy)
		if err := validateRaylineARCPolicyBindings(decision); (err == nil) != tc.valid {
			t.Fatalf("%s: %v", name, err)
		}
	}
}
