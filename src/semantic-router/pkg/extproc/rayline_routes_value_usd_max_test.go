package extproc

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A cell that raises policy_service.value_usd_max accepts caller V up to it
// on /v1/routes and on generation routing, and refuses past it in the
// contract's words; readiness then asks the package for cost_weight 1/max,
// so a package bounded narrower keeps the cell unready instead of refusing
// V at decide time. The default cell is unchanged: V 5.01 is refused and
// cost_weight 0.2 is required.
func TestRaylineARCValueUSDMaxIsPerCell(t *testing.T) {
	body := func(value string) []byte {
		return []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"x"}],"routing":{"value_usd":` + value + `}}`)
	}
	for _, tc := range []struct {
		max     float64
		value   string
		refused bool
		message string
	}{
		{config.DefaultRaylineARCValueUSDMax, "5", false, ""},
		{config.DefaultRaylineARCValueUSDMax, "5.01", true, "routing.value_usd must lie in [0.1, 5]; it is never clamped."},
		{30, "30", false, ""},
		{30, "12.5", false, ""},
		{30, "30.01", true, "routing.value_usd must lie in [0.1, 30]; it is never clamped."},
		{30, "0.09", true, "routing.value_usd must lie in [0.1, 30]; it is never clamped."},
	} {
		routing, _, refusal := readRaylineRoutesRouting(body(tc.value), nil, tc.max)
		if tc.refused {
			if refusal == nil || refusal.code != "value_out_of_bounds" || refusal.message != tc.message {
				t.Errorf("max %g, V %s: refusal %+v", tc.max, tc.value, refusal)
			}
			continue
		}
		if refusal != nil || routing == nil || routing.valueUSD == nil {
			t.Errorf("max %g, V %s: refused %+v", tc.max, tc.value, refusal)
		}
	}

	policy := generationPolicy(true, nil)
	if _, _, refusal := readRaylineGenerationRouting(body("12.5"), policy); refusal == nil || refusal.code != "value_out_of_bounds" {
		t.Fatalf("the default generation cell took V 12.5: %+v", refusal)
	}
	policy.ValueUSDMax = 30
	if _, _, refusal := readRaylineGenerationRouting(body("12.5"), policy); refusal != nil {
		t.Fatalf("a value_usd_max 30 generation cell refused V 12.5: %+v", refusal)
	}

	var routes raylinearc.PolicyServiceRequirements
	routesRequirements(&routes, &config.RaylineARCRoutesRoutingConfig{}, 30)
	var generation raylinearc.PolicyServiceRequirements
	generationRoutingRequirements(&generation, policy)
	for name, required := range map[string]raylinearc.PolicyServiceRequirements{"routes": routes, "generation": generation} {
		if !slices.Contains(required.CostWeights, 1.0/30) || slices.Contains(required.CostWeights, 0.2) {
			t.Errorf("%s readiness at value_usd_max 30 requires cost weights %v", name, required.CostWeights)
		}
	}
}
