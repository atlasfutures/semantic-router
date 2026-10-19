package config

import (
	"math"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestRaylineARCDecisionsV2ConfigParsesAndValidates(t *testing.T) {
	var policy RaylineARCPolicyServiceConfig
	if err := yaml.Unmarshal([]byte(`
decisions_v2:
  cost_weight: 1.0
  tier_points:
    - {name: swift, cost_weight: 2.0}
    - {name: sage, cost_weight: 0}
  response_fields: [price_per_mtok, thinking]
`), &policy); err != nil {
		t.Fatal(err)
	}
	v2 := policy.DecisionsV2
	if v2 == nil || v2.CostWeight == nil || *v2.CostWeight != 1 || len(v2.TierPoints) != 2 ||
		v2.TierPoints[1].Name != "sage" || v2.TierPoints[1].CostWeight == nil || *v2.TierPoints[1].CostWeight != 0 ||
		len(v2.ResponseFields) != 2 {
		t.Fatalf("parsed %+v", v2)
	}

	// Absent is off, and valid; so is the parsed block.
	cfg, decision := policyDispatchFixture()
	if decision.Algorithm.RaylineARC.PolicyService.DecisionsV2 != nil {
		t.Fatal("decisions_v2 is on by default")
	}
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("a cell without decisions_v2 refused: %v", err)
	}
	decision.Algorithm.RaylineARC.PolicyService.DecisionsV2 = v2
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("valid decisions_v2 refused: %v", err)
	}

	weight := func(value float64) *float64 { return &value }
	points := func(count int) []RaylineARCTierPoint {
		out := make([]RaylineARCTierPoint, count)
		for index := range out {
			out[index] = RaylineARCTierPoint{Name: strings.Repeat("t", index+1), CostWeight: weight(1)}
		}
		return out
	}
	for name, tc := range map[string]struct {
		cfg  RaylineARCDecisionsV2Config
		want string
	}{
		"negative V":           {RaylineARCDecisionsV2Config{CostWeight: weight(-0.5)}, "cost_weight"},
		"NaN V":                {RaylineARCDecisionsV2Config{CostWeight: weight(math.NaN())}, "cost_weight"},
		"infinite V":           {RaylineARCDecisionsV2Config{CostWeight: weight(math.Inf(1))}, "cost_weight"},
		"nine tier points":     {RaylineARCDecisionsV2Config{TierPoints: points(9)}, "at most 8"},
		"unnamed tier point":   {RaylineARCDecisionsV2Config{TierPoints: []RaylineARCTierPoint{{CostWeight: weight(1)}}}, "name"},
		"long tier name":       {RaylineARCDecisionsV2Config{TierPoints: []RaylineARCTierPoint{{Name: strings.Repeat("x", 65), CostWeight: weight(1)}}}, "name"},
		"repeated tier name":   {RaylineARCDecisionsV2Config{TierPoints: []RaylineARCTierPoint{{Name: "a", CostWeight: weight(1)}, {Name: "a", CostWeight: weight(2)}}}, "repeats"},
		"tier point without V": {RaylineARCDecisionsV2Config{TierPoints: []RaylineARCTierPoint{{Name: "a"}}}, "cost_weight"},
		"negative tier V":      {RaylineARCDecisionsV2Config{TierPoints: []RaylineARCTierPoint{{Name: "a", CostWeight: weight(-1)}}}, "cost_weight"},
		"unknown field":        {RaylineARCDecisionsV2Config{ResponseFields: []string{"predicted_usage"}}, "response_fields"},
		"repeated field":       {RaylineARCDecisionsV2Config{ResponseFields: []string{"thinking", "thinking"}}, "repeats"},
	} {
		decision.Algorithm.RaylineARC.PolicyService.DecisionsV2 = &tc.cfg
		err := validatePolicyDispatch(cfg, decision)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want one naming %q", name, err, tc.want)
		}
	}
	decision.Algorithm.RaylineARC.PolicyService.DecisionsV2 = &RaylineARCDecisionsV2Config{TierPoints: points(8)}
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("eight tier points refused: %v", err)
	}
}
