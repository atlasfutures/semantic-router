package config

import (
	"fmt"
	"strings"
)

const (
	defaultRaylineARCPolicySessionIdleSeconds  = 900
	defaultRaylineARCPolicySessionStartSeconds = 10
	maxRaylineARCPolicySessionIdleSeconds      = 24 * 60 * 60
	maxRaylineARCPolicySessionStartSeconds     = 60
)

// RaylineARCPolicySessionsConfig selects a policy service deployed as a Modal
// Server with sticky sessions (@modal.sessioned): base_url is the Server's
// URL, and each episode holds one Modal session, persisted with its episode
// state, so its turns reach the one container that keeps its retained
// encoder session while the Server autoscales. Absent or disabled, the cell
// calls base_url (or its replicas) as before.
type RaylineARCPolicySessionsConfig struct {
	Enabled bool `yaml:"enabled,omitempty"`
	// IdleTimeoutSeconds is how long Modal keeps a session without a request
	// in flight; zero selects 900, the episode store's shipped idle TTL.
	IdleTimeoutSeconds int `yaml:"idle_timeout_seconds,omitempty"`
	// StartTimeoutSeconds bounds a session start; zero selects 10. Modal
	// queues a start for capacity for minutes, so a turn whose start lapses
	// fails closed instead of waiting.
	StartTimeoutSeconds int `yaml:"start_timeout_seconds,omitempty"`
	// FallbackBaseURL is a non-sessioned deployment of the same package,
	// called with the same proxy key. Set, a turn whose session start fails
	// or lapses is decided there, without a session, instead of failing
	// closed; llm_usage labels it policy_session_fallback. Empty (the
	// default) keeps the turn failing closed.
	FallbackBaseURL string `yaml:"fallback_base_url,omitempty"`
}

// EffectiveIdleTimeoutSeconds is the session idle timeout in force.
func (cfg RaylineARCPolicySessionsConfig) EffectiveIdleTimeoutSeconds() int {
	if cfg.IdleTimeoutSeconds <= 0 {
		return defaultRaylineARCPolicySessionIdleSeconds
	}
	return cfg.IdleTimeoutSeconds
}

// EffectiveStartTimeoutSeconds is the session start bound in force.
func (cfg RaylineARCPolicySessionsConfig) EffectiveStartTimeoutSeconds() int {
	if cfg.StartTimeoutSeconds <= 0 {
		return defaultRaylineARCPolicySessionStartSeconds
	}
	return cfg.StartTimeoutSeconds
}

// validateRaylineARCPolicySessions accepts sessions only on a single
// base_url with the Modal proxy key (a session start is refused without
// it). Sessions and replicas are exclusive: a sessioned Server autoscales
// itself, so it is one URL. Settings without enabled are refused rather
// than ignored.
func validateRaylineARCPolicySessions(cfg *RaylineARCPolicyServiceConfig) error {
	sessions := cfg.Sessions
	if !sessions.Enabled {
		if sessions.IdleTimeoutSeconds != 0 || sessions.StartTimeoutSeconds != 0 || sessions.FallbackBaseURL != "" {
			return fmt.Errorf("sessions settings require sessions.enabled")
		}
		return nil
	}
	if len(cfg.Replicas) != 0 || strings.TrimSpace(cfg.BaseURL) == "" {
		return fmt.Errorf("sessions require base_url and no replicas")
	}
	if cfg.ModalKeyEnv == "" || cfg.ModalSecretEnv == "" {
		return fmt.Errorf("sessions require modal_key_env and modal_secret_env")
	}
	if sessions.IdleTimeoutSeconds < 0 || sessions.IdleTimeoutSeconds > maxRaylineARCPolicySessionIdleSeconds {
		return fmt.Errorf("sessions.idle_timeout_seconds must be between 1 and %d", maxRaylineARCPolicySessionIdleSeconds)
	}
	if sessions.StartTimeoutSeconds < 0 || sessions.StartTimeoutSeconds > maxRaylineARCPolicySessionStartSeconds {
		return fmt.Errorf("sessions.start_timeout_seconds must be between 1 and %d", maxRaylineARCPolicySessionStartSeconds)
	}
	if fallback := sessions.FallbackBaseURL; fallback != "" &&
		(strings.TrimSpace(fallback) != fallback || strings.TrimRight(fallback, "/") == strings.TrimRight(cfg.BaseURL, "/")) {
		return fmt.Errorf("sessions.fallback_base_url must be a URL other than base_url")
	}
	return nil
}
