package config

import (
	"math"
	"strings"
	"testing"
)

// A cell's caller-V maximum defaults to 5 and may be raised to the public
// contract's ceiling of 30 (pathfinder#6105), never past it and never to or
// below the minimum.
func TestRaylineARCPolicyServiceValueUSDMax(t *testing.T) {
	for value, valid := range map[float64]bool{
		0: true, 5: true, 4.5: true, 30: true,
		RaylineARCValueUSDMin: false, 0.05: false, -1: false, 30.01: false, math.Inf(1): false, math.NaN(): false,
	} {
		cfg := &RaylineARCPolicyServiceConfig{
			BaseURL: "https://policy.example", TotalTimeoutSeconds: 5, PackageAlias: "a",
			PackageSHA256: strings.Repeat("a", 64), ValueUSDMax: value,
			Bindings: []RaylineARCPolicyBinding{{ActionID: strings.Repeat("b", 64), Worker: "w"}},
		}
		err := validateRaylineARCPolicyServiceConfig(cfg)
		refused := err != nil && strings.Contains(err.Error(), "value_usd_max")
		if refused == valid {
			t.Errorf("value_usd_max %g: err = %v", value, err)
		}
	}
	var unset *RaylineARCPolicyServiceConfig
	if got := unset.EffectiveValueUSDMax(); got != DefaultRaylineARCValueUSDMax {
		t.Fatalf("a cell without a policy service accepts V up to %g", got)
	}
	if got := (&RaylineARCPolicyServiceConfig{}).EffectiveValueUSDMax(); got != 5 {
		t.Fatalf("unset value_usd_max is %g, want 5", got)
	}
	if got := (&RaylineARCPolicyServiceConfig{ValueUSDMax: 30}).EffectiveValueUSDMax(); got != 30 {
		t.Fatalf("value_usd_max 30 is %g", got)
	}
}
