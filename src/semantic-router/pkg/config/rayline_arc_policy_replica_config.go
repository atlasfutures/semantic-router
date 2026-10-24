package config

import (
	"fmt"
	"slices"
	"strings"
)

// RaylineARCPolicyFailoverV1 is the policy-service replica failover contract.
const RaylineARCPolicyFailoverV1 = "rayline.arc.policy-failover.v1"

// raylineARCPolicyFailoverStatuses are the statuses a policy replica may be
// declared unavailable by: each is answered before the service touches the
// episode (404 package not loaded, 429 at capacity, 503 unavailable). A 409
// session_busy, a 5xx gateway error or a timeout never moves an episode.
var raylineARCPolicyFailoverStatuses = []int{404, 429, 503}

// validateRaylineARCPolicyServiceEndpoints accepts exactly one of base_url
// (one service, today's shape) or replicas (an episode-sticky pool). A cell
// without replicas is unchanged: failover is refused without them.
func validateRaylineARCPolicyServiceEndpoints(cfg *RaylineARCPolicyServiceConfig) error {
	hasBaseURL := strings.TrimSpace(cfg.BaseURL) != ""
	hasReplicas := len(cfg.Replicas) != 0
	if hasBaseURL && hasReplicas {
		return fmt.Errorf("exactly one of base_url or replicas must be configured")
	}
	if !hasReplicas {
		// Today's single-service cell, validated exactly as before.
		if !emptyRaylineARCFailoverConfig(cfg.Failover) {
			return fmt.Errorf("failover requires replicas")
		}
		return validateRaylineARCBaseURL(cfg.BaseURL)
	}
	// One replica is allowed, so a pool can scale down to a single service
	// without changing the config shape; it has no failover target.
	if len(cfg.Replicas) > maxRaylineARCEncoderReplicas {
		return fmt.Errorf("replicas must contain between 1 and %d entries", maxRaylineARCEncoderReplicas)
	}
	active, err := validateRaylineARCReplicaMembers(cfg.Replicas)
	if err != nil {
		return err
	}
	if active == 0 {
		return fmt.Errorf("replicas require at least one active member")
	}
	return validateRaylineARCPolicyFailover(cfg.Failover)
}

func validateRaylineARCPolicyFailover(cfg RaylineARCEncoderFailoverConfig) error {
	if cfg.SchemaVersion != RaylineARCPolicyFailoverV1 {
		return fmt.Errorf("failover.schema_version must be %q", RaylineARCPolicyFailoverV1)
	}
	if cfg.UnavailableCooldownSeconds <= 0 || cfg.UnavailableCooldownSeconds > maxRaylineARCCooldownSeconds {
		return fmt.Errorf("failover.unavailable_cooldown_seconds must be between 1 and %d", maxRaylineARCCooldownSeconds)
	}
	if cfg.MaxRemaps != 1 {
		return fmt.Errorf("failover.max_remaps must be 1")
	}
	if len(cfg.UnavailableStatusCodes) == 0 {
		return fmt.Errorf("failover.unavailable_status_codes is required")
	}
	seen := make(map[int]struct{}, len(cfg.UnavailableStatusCodes))
	for _, status := range cfg.UnavailableStatusCodes {
		if !slices.Contains(raylineARCPolicyFailoverStatuses, status) {
			return fmt.Errorf("failover unavailable status must be one of %v", raylineARCPolicyFailoverStatuses)
		}
		if _, exists := seen[status]; exists {
			return fmt.Errorf("failover unavailable status is duplicated")
		}
		seen[status] = struct{}{}
	}
	return nil
}
