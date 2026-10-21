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

func TestRaylineARCExplorationTemperatureConfigParsesAndValidates(t *testing.T) {
	sha := strings.Repeat("a", 64)
	var policy RaylineARCPolicyServiceConfig
	if err := yaml.Unmarshal([]byte(`
exploration_temperature:
  temperature: 0.4
  temperature_max: 1.5
  caller_temperature: true
  salt: vsr-explore-2026-10
  calibration: {r: 0.05, package_sha256: `+sha+`, readout_pin: "qwen3.8-27b-arc-encoder@depth8"}
`), &policy); err != nil {
		t.Fatal(err)
	}
	explore := policy.ExplorationTemperature
	if explore == nil || *explore.Temperature != 0.4 || *explore.TemperatureMax != 1.5 || !explore.CallerTemperature ||
		explore.Salt != "vsr-explore-2026-10" || explore.Calibration == nil || *explore.Calibration.R != 0.05 {
		t.Fatalf("parsed %+v", explore)
	}

	cfg, decision := policyDispatchFixture()
	policyService := decision.Algorithm.RaylineARC.PolicyService
	if policyService.ExplorationTemperature != nil {
		t.Fatal("exploration_temperature is on by default")
	}
	policyService.ExplorationTemperature = explore
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("valid exploration_temperature refused: %v", err)
	}
	// A calibration is provenance and optional; it names any package.
	uncalibrated := *explore
	uncalibrated.Calibration = nil
	policyService.ExplorationTemperature = &uncalibrated
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("exploration_temperature without a calibration refused: %v", err)
	}

	value := func(v float64) *float64 { return &v }
	with := func(change func(*RaylineARCExplorationTemperatureConfig)) RaylineARCExplorationTemperatureConfig {
		copied := *explore
		calibration := *explore.Calibration
		copied.Calibration = &calibration
		change(&copied)
		return copied
	}
	type tcfg = RaylineARCExplorationTemperatureConfig
	for name, tc := range map[string]struct {
		cfg  tcfg
		want string
	}{
		"no temperature":       {with(func(c *tcfg) { c.Temperature = nil }), "temperature must"},
		"negative temperature": {with(func(c *tcfg) { c.Temperature = value(-0.1) }), "temperature must"},
		"NaN temperature":      {with(func(c *tcfg) { c.Temperature = value(math.NaN()) }), "temperature must"},
		"default above cap":    {with(func(c *tcfg) { c.Temperature = value(1.6) }), "temperature_max"},
		"no cap":               {with(func(c *tcfg) { c.TemperatureMax = nil }), "temperature_max"},
		"infinite cap":         {with(func(c *tcfg) { c.TemperatureMax = value(math.Inf(1)) }), "temperature_max"},
		"no salt":              {with(func(c *tcfg) { c.Salt = "" }), "salt"},
		"long salt":            {with(func(c *tcfg) { c.Salt = strings.Repeat("s", 129) }), "salt"},
		"r above 1":            {with(func(c *tcfg) { c.Calibration.R = value(1.5) }), "calibration.r"},
		"bad sha":              {with(func(c *tcfg) { c.Calibration.PackageSHA256 = "ABC" }), "calibration.package_sha256"},
		"no readout pin":       {with(func(c *tcfg) { c.Calibration.ReadoutPin = "" }), "readout_pin"},
	} {
		policyService.ExplorationTemperature = &tc.cfg
		err := validatePolicyDispatch(cfg, decision)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want one naming %q", name, err, tc.want)
		}
	}

	// The service refuses a request carrying both shapes, so a cell
	// configures one.
	policyService.ExplorationTemperature = explore
	policyService.Exploration = &RaylineARCExplorationConfig{
		Tau: value(0.02), TauMax: value(0.05), Salt: "s",
		Basis: RaylineARCExplorationBasis{R: value(0.05), PackageSHA256: policyService.PackageSHA256, ReadoutPin: "p"},
	}
	if err := validatePolicyDispatch(cfg, decision); err == nil || !strings.Contains(err.Error(), "exclusive") {
		t.Fatalf("both shapes configured: err = %v", err)
	}
}
