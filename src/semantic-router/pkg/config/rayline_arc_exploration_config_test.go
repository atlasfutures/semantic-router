package config

import (
	"math"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestRaylineARCExplorationConfigParsesAndValidates(t *testing.T) {
	sha := strings.Repeat("a", 64)
	var policy RaylineARCPolicyServiceConfig
	if err := yaml.Unmarshal([]byte(`
eval_cell: true
exploration:
  tau: 0.02
  tau_max: 0.05
  basis: {r: 0.05, package_sha256: `+sha+`, readout_pin: "qwen3.8-27b-arc-encoder@depth8"}
  salt: vsr-explore-2026-10
`), &policy); err != nil {
		t.Fatal(err)
	}
	explore := policy.Exploration
	if explore == nil || *explore.Tau != 0.02 || *explore.TauMax != 0.05 || *explore.Basis.R != 0.05 ||
		explore.Basis.PackageSHA256 != sha || explore.Basis.ReadoutPin == "" || explore.Salt != "vsr-explore-2026-10" ||
		!policy.EvalCell {
		t.Fatalf("parsed %+v", policy)
	}

	cfg, decision := policyDispatchFixture()
	policyService := decision.Algorithm.RaylineARC.PolicyService
	if policyService.Exploration != nil || policyService.EvalCell {
		t.Fatal("exploration is on by default")
	}
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("a cell without exploration refused: %v", err)
	}
	policyService.Exploration = explore
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("valid exploration refused: %v", err)
	}

	value := func(v float64) *float64 { return &v }
	with := func(change func(*RaylineARCExplorationConfig)) RaylineARCExplorationConfig {
		copied := *explore
		change(&copied)
		return copied
	}
	for name, tc := range map[string]struct {
		cfg  RaylineARCExplorationConfig
		want string
	}{
		"no tau":           {with(func(c *RaylineARCExplorationConfig) { c.Tau = nil }), "tau must"},
		"negative tau":     {with(func(c *RaylineARCExplorationConfig) { c.Tau = value(-0.1) }), "tau must"},
		"NaN tau":          {with(func(c *RaylineARCExplorationConfig) { c.Tau = value(math.NaN()) }), "tau must"},
		"tau above max":    {with(func(c *RaylineARCExplorationConfig) { c.Tau = value(0.06) }), "tau_max"},
		"no tau_max":       {with(func(c *RaylineARCExplorationConfig) { c.TauMax = nil }), "tau_max"},
		"infinite tau_max": {with(func(c *RaylineARCExplorationConfig) { c.TauMax = value(math.Inf(1)) }), "tau_max"},
		"no r":             {with(func(c *RaylineARCExplorationConfig) { c.Basis.R = nil }), "basis.r"},
		"r above 1":        {with(func(c *RaylineARCExplorationConfig) { c.Basis.R = value(1.5) }), "basis.r"},
		"bad sha":          {with(func(c *RaylineARCExplorationConfig) { c.Basis.PackageSHA256 = "ABC" }), "basis.package_sha256"},
		"another package": {with(func(c *RaylineARCExplorationConfig) {
			c.Basis.PackageSHA256 = strings.Repeat("b", 64)
		}), "cell's package_sha256"},
		"no readout pin":   {with(func(c *RaylineARCExplorationConfig) { c.Basis.ReadoutPin = "" }), "readout_pin"},
		"long readout pin": {with(func(c *RaylineARCExplorationConfig) { c.Basis.ReadoutPin = strings.Repeat("p", 129) }), "readout_pin"},
		"no salt":          {with(func(c *RaylineARCExplorationConfig) { c.Salt = "" }), "salt"},
		"long salt":        {with(func(c *RaylineARCExplorationConfig) { c.Salt = strings.Repeat("s", 129) }), "salt"},
	} {
		policyService.Exploration = &tc.cfg
		err := validatePolicyDispatch(cfg, decision)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want one naming %q", name, err, tc.want)
		}
	}
	// tau 0 (greedy) and tau at the ceiling are both valid.
	for _, tau := range []float64{0, 0.05} {
		policyService.Exploration = &RaylineARCExplorationConfig{}
		*policyService.Exploration = with(func(c *RaylineARCExplorationConfig) { c.Tau = value(tau) })
		if err := validatePolicyDispatch(cfg, decision); err != nil {
			t.Errorf("tau %g refused: %v", tau, err)
		}
	}
}
