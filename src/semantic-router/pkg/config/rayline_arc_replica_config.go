package config

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

const (
	RaylineARCEncoderFailoverV1      = "rayline.arc.encoder-failover.v1"
	RaylineARCEncoderMembershipV1    = "rayline.arc.encoder-membership.v1"
	RaylineARCEncoderMembershipRedis = "redis"
	RaylineARCEncoderActive          = "active"
	RaylineARCEncoderDraining        = "draining"

	maxRaylineARCEncoderReplicas          = 8
	maxRaylineARCReplicaIDLength          = 64
	maxRaylineARCCooldownSeconds          = 3600
	maxRaylineARCMembershipRefreshSeconds = 300
)

var raylineARCReplicaIDPattern = regexp.MustCompile(
	`^[A-Za-z0-9][A-Za-z0-9._-]*$`,
)

// RaylineARCEncoderReplicaConfig is a versioned retained-encoder member. A
// draining replica continues serving persisted owners but receives no new
// episode assignments.
type RaylineARCEncoderReplicaConfig struct {
	ID      string `yaml:"id"`
	BaseURL string `yaml:"base_url"`
	State   string `yaml:"state"`
}

// RaylineARCEncoderMembershipConfig opts a retained-session deployment into
// the reviewed dynamic membership source. The source holds the member URLs and
// active/draining state; this static config holds only the versioned transport
// and refresh contract.
type RaylineARCEncoderMembershipConfig struct {
	SchemaVersion  string `yaml:"schema_version"`
	Source         string `yaml:"source"`
	RefreshSeconds int    `yaml:"refresh_seconds"`
}

// RaylineARCEncoderFailoverConfig opts a replicated retained-session service
// into one exact semantic contract. Ambiguous transport failures are always
// fail-closed and therefore are not configurable here.
type RaylineARCEncoderFailoverConfig struct {
	SchemaVersion              string `yaml:"schema_version"`
	UnavailableStatusCodes     []int  `yaml:"unavailable_status_codes"`
	UnavailableCooldownSeconds int    `yaml:"unavailable_cooldown_seconds"`
	MaxRemaps                  int    `yaml:"max_remaps"`
}

func validateRaylineARCEncoderMembership(cfg RaylineARCEncoderConfig) error {
	hasBaseURL := strings.TrimSpace(cfg.BaseURL) != ""
	hasReplicas := len(cfg.Replicas) != 0
	hasDynamicMembership := cfg.usesDynamicMembership()
	modeCount := 0
	for _, enabled := range []bool{hasBaseURL, hasReplicas, hasDynamicMembership} {
		if enabled {
			modeCount++
		}
	}
	if modeCount != 1 {
		if !hasDynamicMembership {
			return fmt.Errorf("exactly one of base_url or replicas must be configured")
		}
		return fmt.Errorf("exactly one of base_url, replicas, or membership must be configured")
	}
	if hasBaseURL {
		if err := validateRaylineARCBaseURL(cfg.BaseURL); err != nil {
			return err
		}
		if !emptyRaylineARCFailoverConfig(cfg.Failover) {
			return fmt.Errorf("failover requires replicas")
		}
		return nil
	}
	if hasDynamicMembership {
		return validateRaylineARCDynamicEncoderMembership(cfg)
	}
	return validateRaylineARCEncoderReplicas(cfg)
}

func (cfg RaylineARCEncoderConfig) usesDynamicMembership() bool {
	return cfg.Membership.SchemaVersion != "" ||
		cfg.Membership.Source != "" ||
		cfg.Membership.RefreshSeconds != 0
}

func (cfg RaylineARCEncoderConfig) usesReplicaMembership() bool {
	return len(cfg.Replicas) != 0 || cfg.usesDynamicMembership()
}

func emptyRaylineARCFailoverConfig(cfg RaylineARCEncoderFailoverConfig) bool {
	return cfg.SchemaVersion == "" &&
		len(cfg.UnavailableStatusCodes) == 0 &&
		cfg.UnavailableCooldownSeconds == 0 &&
		cfg.MaxRemaps == 0
}

func validateRaylineARCEncoderReplicas(cfg RaylineARCEncoderConfig) error {
	if len(cfg.Replicas) < 2 || len(cfg.Replicas) > maxRaylineARCEncoderReplicas {
		return fmt.Errorf(
			"replicas must contain between 2 and %d entries",
			maxRaylineARCEncoderReplicas,
		)
	}
	if err := validateRaylineARCReplicaServingMode(cfg); err != nil {
		return err
	}
	active, err := validateRaylineARCReplicaMembers(cfg.Replicas)
	if err != nil {
		return err
	}
	if active == 0 {
		return fmt.Errorf("replicas require at least one active member")
	}
	return validateRaylineARCEncoderFailover(cfg.Failover)
}

func validateRaylineARCDynamicEncoderMembership(
	cfg RaylineARCEncoderConfig,
) error {
	if err := validateRaylineARCReplicaServingMode(cfg); err != nil {
		return err
	}
	if err := validateRaylineARCEncoderDynamicMembershipConfig(
		cfg.Membership,
	); err != nil {
		return err
	}
	return validateRaylineARCEncoderFailover(cfg.Failover)
}

func validateRaylineARCReplicaServingMode(cfg RaylineARCEncoderConfig) error {
	if cfg.ServingRung != RaylineARCServingRungB ||
		!slices.Contains(
			cfg.RequiredCapabilities,
			RaylineARCCapabilityResumableMean,
		) {
		return fmt.Errorf(
			"replicas require retained serving rung B with resumable capability",
		)
	}
	if cfg.MaxRetries != 0 {
		return fmt.Errorf("replicas require max_retries=0")
	}
	return nil
}

func validateRaylineARCEncoderDynamicMembershipConfig(
	cfg RaylineARCEncoderMembershipConfig,
) error {
	if cfg.SchemaVersion != RaylineARCEncoderMembershipV1 {
		return fmt.Errorf(
			"membership.schema_version must be %q",
			RaylineARCEncoderMembershipV1,
		)
	}
	if cfg.Source != RaylineARCEncoderMembershipRedis {
		return fmt.Errorf(
			"membership.source must be %q",
			RaylineARCEncoderMembershipRedis,
		)
	}
	if cfg.RefreshSeconds <= 0 ||
		cfg.RefreshSeconds > maxRaylineARCMembershipRefreshSeconds {
		return fmt.Errorf(
			"membership.refresh_seconds must be between 1 and %d",
			maxRaylineARCMembershipRefreshSeconds,
		)
	}
	return nil
}

func validateRaylineARCReplicaMembers(
	replicas []RaylineARCEncoderReplicaConfig,
) (int, error) {
	seenIDs := make(map[string]struct{}, len(replicas))
	seenURLs := make(map[string]struct{}, len(replicas))
	active := 0
	for _, replica := range replicas {
		isActive, err := validateRaylineARCReplica(
			replica,
			seenIDs,
			seenURLs,
		)
		if err != nil {
			return 0, err
		}
		if isActive {
			active++
		}
	}
	return active, nil
}

func validateRaylineARCReplica(
	replica RaylineARCEncoderReplicaConfig,
	seenIDs map[string]struct{},
	seenURLs map[string]struct{},
) (bool, error) {
	if len(replica.ID) == 0 ||
		len(replica.ID) > maxRaylineARCReplicaIDLength ||
		!raylineARCReplicaIDPattern.MatchString(replica.ID) {
		return false, fmt.Errorf("replica id must be a bounded stable identifier")
	}
	if _, exists := seenIDs[replica.ID]; exists {
		return false, fmt.Errorf("replica id %q is duplicated", replica.ID)
	}
	seenIDs[replica.ID] = struct{}{}
	if err := validateRaylineARCBaseURL(replica.BaseURL); err != nil {
		return false, fmt.Errorf("replica %q: %w", replica.ID, err)
	}
	urlKey := strings.TrimRight(strings.TrimSpace(replica.BaseURL), "/")
	if _, exists := seenURLs[urlKey]; exists {
		return false, fmt.Errorf("replica base_url is duplicated")
	}
	seenURLs[urlKey] = struct{}{}
	switch replica.State {
	case RaylineARCEncoderActive:
		return true, nil
	case RaylineARCEncoderDraining:
		return false, nil
	default:
		return false, fmt.Errorf(
			"replica state must be %q or %q",
			RaylineARCEncoderActive,
			RaylineARCEncoderDraining,
		)
	}
}

func validateRaylineARCEncoderFailover(
	cfg RaylineARCEncoderFailoverConfig,
) error {
	if cfg.SchemaVersion != RaylineARCEncoderFailoverV1 {
		return fmt.Errorf(
			"failover.schema_version must be %q",
			RaylineARCEncoderFailoverV1,
		)
	}
	if cfg.UnavailableCooldownSeconds <= 0 ||
		cfg.UnavailableCooldownSeconds > maxRaylineARCCooldownSeconds {
		return fmt.Errorf(
			"failover.unavailable_cooldown_seconds must be between 1 and %d",
			maxRaylineARCCooldownSeconds,
		)
	}
	if cfg.MaxRemaps != 1 {
		return fmt.Errorf("failover.max_remaps must be 1")
	}
	if len(cfg.UnavailableStatusCodes) == 0 ||
		len(cfg.UnavailableStatusCodes) > 16 {
		return fmt.Errorf(
			"failover.unavailable_status_codes must contain between 1 and 16 entries",
		)
	}
	seen := make(map[int]struct{}, len(cfg.UnavailableStatusCodes))
	for _, statusCode := range cfg.UnavailableStatusCodes {
		if statusCode < 400 || statusCode > 599 {
			return fmt.Errorf(
				"failover unavailable status must be between 400 and 599",
			)
		}
		if _, exists := seen[statusCode]; exists {
			return fmt.Errorf("failover unavailable status is duplicated")
		}
		seen[statusCode] = struct{}{}
	}
	return nil
}
