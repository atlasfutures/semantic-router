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

// A fallback-enabled package's model names fit an exclusion. Control: the
// same name with the fallback off passes this check.
func TestRaylineARCPolicyFallbackBoundsModelNames(t *testing.T) {
	for _, size := range []int{RaylineARCMaxFallbackModelBytes, RaylineARCMaxFallbackModelBytes + 1} {
		for _, enabled := range []bool{true, false} {
			_, decision := policyDispatchFixture()
			policy := decision.Algorithm.RaylineARC.PolicyService
			policy.Bindings = []RaylineARCPolicyBinding{{ActionID: "a", Worker: "arm-off", Model: strings.Repeat("m", size)}}
			policy.Fallback = &RaylineARCPolicyFallbackConfig{Enabled: enabled}
			err := validateRaylineARCPolicyBindings(decision)
			refused := err != nil && strings.Contains(err.Error(), "bytes")
			if refused != (enabled && size > RaylineARCMaxFallbackModelBytes) {
				t.Fatalf("size=%d enabled=%v: %v", size, enabled, err)
			}
		}
	}
}

// With the fallback on, each worker serves one model, so a held worker names
// the model it holds. Control: the fallback off, and one model on the worker.
func TestRaylineARCPolicyFallbackNeedsOneModelPerWorker(t *testing.T) {
	for _, second := range []string{"vendor/a", "vendor/b"} {
		for _, enabled := range []bool{true, false} {
			_, decision := policyDispatchFixture()
			policy := decision.Algorithm.RaylineARC.PolicyService
			policy.Bindings = []RaylineARCPolicyBinding{
				{ActionID: "a", Worker: "arm-off", Model: "vendor/a"},
				{ActionID: "b", Worker: "arm-off", Model: second},
			}
			policy.Fallback = &RaylineARCPolicyFallbackConfig{Enabled: enabled}
			err := validateRaylineARCPolicyBindings(decision)
			refused := err != nil && strings.Contains(err.Error(), "one model per worker")
			if refused != (enabled && second != "vendor/a") {
				t.Fatalf("second=%s enabled=%v: %v", second, enabled, err)
			}
		}
	}
}
