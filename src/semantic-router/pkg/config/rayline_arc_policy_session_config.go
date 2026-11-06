package config

import (
	"fmt"
	"net/url"
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
	if sessions.FallbackBaseURL != "" {
		return validateRaylineARCPolicySessionFallback(sessions.FallbackBaseURL, cfg.BaseURL)
	}
	return nil
}

// validateRaylineARCPolicySessionFallback accepts an absolute https URL (the
// fallback is called with the proxy key) other than base_url. It names a
// deployment of the cell's own package: nothing here can read which package
// it serves, so every answer is checked against the cell's package alias and
// SHA-256 per turn, and one for another package fails the turn closed
// (policy_package_mismatch).
func validateRaylineARCPolicySessionFallback(fallback string, baseURL string) error {
	parsed, err := url.Parse(fallback)
	if err != nil || strings.TrimSpace(fallback) != fallback || len(fallback) > maxRaylineARCConfigStringLength ||
		parsed.Scheme != "https" || parsed.Host == "" ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return fmt.Errorf("sessions.fallback_base_url must be an absolute https URL without credentials, a query, or a fragment")
	}
	if strings.TrimRight(fallback, "/") == strings.TrimRight(strings.TrimSpace(baseURL), "/") {
		return fmt.Errorf("sessions.fallback_base_url must be a URL other than base_url")
	}
	return nil
}
