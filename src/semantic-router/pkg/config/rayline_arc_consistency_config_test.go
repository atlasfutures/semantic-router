package config

import (
	"strings"
	"testing"
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

// The policy service refuses a second in-flight call for one session, so a
// relaxed policy-service cell is refused until its lock-free call exists.
func TestRaylineARCRelaxedNotYetServedInPolicyServiceMode(t *testing.T) {
	_, decision := policyDispatchFixture()
	if err := validateRaylineARCAlgorithmConfig(decision.Algorithm.RaylineARC); err != nil {
		t.Fatalf("strict policy fixture refused: %v", err)
	}
	decision.Algorithm.RaylineARC.Episode.Consistency = RaylineARCConsistencyRelaxed
	if err := validateRaylineARCAlgorithmConfig(decision.Algorithm.RaylineARC); err == nil ||
		!strings.Contains(err.Error(), "consistency=relaxed is not yet served in the policy-service mode") {
		t.Fatalf("relaxed policy-service error = %v", err)
	}
}
