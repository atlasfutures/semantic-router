package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// consistency is a per-cell choice: strict (or omitted) serializes turns
// behind the lease, relaxed commits optimistically. Anything else is refused.
func TestRaylineARCEpisodeConsistencyValues(t *testing.T) {
	for _, value := range []string{"", RaylineARCConsistencyStrict, RaylineARCConsistencyRelaxed} {
		decision := validRaylineARCDecision()
		decision.Algorithm.RaylineARC.Episode.Consistency = value
		if err := validateRaylineARCAlgorithmConfig(decision.Algorithm.RaylineARC); err != nil {
			t.Fatalf("consistency %q refused: %v", value, err)
		}
	}
	decision := validRaylineARCDecision()
	decision.Algorithm.RaylineARC.Episode.Consistency = "eventual"
	if err := validateRaylineARCAlgorithmConfig(decision.Algorithm.RaylineARC); err == nil ||
		!strings.Contains(err.Error(), "consistency must be") {
		t.Fatalf("unknown consistency error = %v", err)
	}
}

// A close fans out under the turn's lease, which a relaxed turn does not hold.
func TestRaylineARCRelaxedRefusesCloseHeader(t *testing.T) {
	decision := validRaylineARCDecision()
	decision.Algorithm.RaylineARC.Episode.Consistency = RaylineARCConsistencyRelaxed
	decision.Algorithm.RaylineARC.Episode.CloseHeader = "x-rayline-episode-close"
	if err := validateRaylineARCAlgorithmConfig(decision.Algorithm.RaylineARC); err == nil ||
		!strings.Contains(err.Error(), "close_header is not served with consistency=relaxed") {
		t.Fatalf("relaxed with close_header error = %v", err)
	}
}

// The policy service serves relaxed decide calls (episode_mode: relaxed), so
// a relaxed policy-service cell loads; close_header stays refused there.
func TestRaylineARCRelaxedServedInPolicyServiceMode(t *testing.T) {
	_, decision := policyDispatchFixture()
	decision.Algorithm.RaylineARC.Episode.Consistency = RaylineARCConsistencyRelaxed
	if err := validateRaylineARCAlgorithmConfig(decision.Algorithm.RaylineARC); err != nil {
		t.Fatalf("relaxed policy-service cell refused: %v", err)
	}
}

// A retained encoder session serializes same-episode encodes, which relaxed
// turns must not wait on.
func TestRaylineARCRelaxedRefusesRetainedEncoderSessions(t *testing.T) {
	decision := validRaylineARCDecision()
	arc := decision.Algorithm.RaylineARC
	arc.Encoder.ServingRung = RaylineARCServingRungB
	arc.Encoder.RequiredCapabilities = []string{RaylineARCCapabilityChunkedMean, RaylineARCCapabilityResumableMean}
	if err := validateRaylineARCAlgorithmConfig(arc); err != nil {
		t.Fatalf("strict retained-session fixture refused: %v", err)
	}
	arc.Episode.Consistency = RaylineARCConsistencyRelaxed
	if err := validateRaylineARCAlgorithmConfig(arc); err == nil ||
		!strings.Contains(err.Error(), "not served with the resumable_causal_mean encoder capability") {
		t.Fatalf("relaxed with retained sessions error = %v", err)
	}
}

// cache_context is off unless a cell sets it, and reads as a YAML bool.
func TestRaylineARCPolicyServiceCacheContextSwitch(t *testing.T) {
	var policy RaylineARCPolicyServiceConfig
	if err := yaml.Unmarshal([]byte("package_alias: a\n"), &policy); err != nil || policy.CacheContext {
		t.Fatalf("absent cache_context = %v, %v; want off", policy.CacheContext, err)
	}
	if err := yaml.Unmarshal([]byte("cache_context: true\n"), &policy); err != nil || !policy.CacheContext {
		t.Fatalf("cache_context: true = %v, %v", policy.CacheContext, err)
	}
	if err := yaml.Unmarshal([]byte("cache_context: sometimes\n"), &policy); err == nil {
		t.Fatal("a non-bool cache_context was accepted")
	}
	_, decision := policyDispatchFixture()
	decision.Algorithm.RaylineARC.PolicyService.CacheContext = true
	if err := validateRaylineARCAlgorithmConfig(decision.Algorithm.RaylineARC); err != nil {
		t.Fatalf("a cell with cache_context refused: %v", err)
	}
}
