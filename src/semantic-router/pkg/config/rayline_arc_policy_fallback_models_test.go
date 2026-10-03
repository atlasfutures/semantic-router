package config

import (
	"fmt"
	"strings"
	"testing"
)

// A fallback-enabled package may serve no more distinct models than an
// episode can exclude. Control: the same bindings with the fallback off pass
// this check.
func TestRaylineARCPolicyFallbackBoundsTheModels(t *testing.T) {
	for _, models := range []int{RaylineARCMaxFallbackModels, RaylineARCMaxFallbackModels + 1} {
		for _, enabled := range []bool{true, false} {
			_, decision := policyDispatchFixture()
			policy := decision.Algorithm.RaylineARC.PolicyService
			policy.Bindings = nil
			for index := 0; index < models; index++ {
				policy.Bindings = append(policy.Bindings, RaylineARCPolicyBinding{
					ActionID: fmt.Sprintf("a%d", index), Worker: "arm-off", Model: fmt.Sprintf("vendor/m%d", index),
				})
			}
			policy.Fallback = &RaylineARCPolicyFallbackConfig{Enabled: enabled}
			err := validateRaylineARCPolicyBindings(decision)
			refused := err != nil && strings.Contains(err.Error(), "distinct models")
			if refused != (enabled && models > RaylineARCMaxFallbackModels) {
				t.Fatalf("models=%d enabled=%v: %v", models, enabled, err)
			}
		}
	}
}
