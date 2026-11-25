package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestRaylineARCPolicySessionsValidation(t *testing.T) {
	keyed := func(sessions RaylineARCPolicySessionsConfig) RaylineARCPolicyServiceConfig {
		return RaylineARCPolicyServiceConfig{
			BaseURL: "https://policy.example.modal.direct", ModalKeyEnv: "KEY", ModalSecretEnv: "SECRET", Sessions: sessions,
		}
	}
	for _, tc := range []struct {
		name string
		cfg  RaylineARCPolicyServiceConfig
		want string
	}{
		{"absent, as before", RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"}, ""},
		{"enabled", keyed(RaylineARCPolicySessionsConfig{Enabled: true}), ""},
		{"enabled with settings", keyed(RaylineARCPolicySessionsConfig{Enabled: true, IdleTimeoutSeconds: 600, StartTimeoutSeconds: 5}), ""},
		{"settings without enabled", keyed(RaylineARCPolicySessionsConfig{StartTimeoutSeconds: 5}), "require sessions.enabled"},
		{"without the proxy key", RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example", Sessions: RaylineARCPolicySessionsConfig{Enabled: true}}, "modal_key_env and modal_secret_env"},
		{"with replicas", RaylineARCPolicyServiceConfig{
			Replicas: []RaylineARCEncoderReplicaConfig{{ID: "s1", BaseURL: "https://s1.example", State: RaylineARCEncoderActive}},
			Failover: policyReplicaFailover(), ModalKeyEnv: "KEY", ModalSecretEnv: "SECRET",
			Sessions: RaylineARCPolicySessionsConfig{Enabled: true},
		}, "base_url and no replicas"},
		{"start timeout too long", keyed(RaylineARCPolicySessionsConfig{Enabled: true, StartTimeoutSeconds: 300}), "start_timeout_seconds"},
		{"caller budget", keyed(RaylineARCPolicySessionsConfig{Enabled: true, MaxNewSessionsPerCaller: 20}), ""},
		{"caller budget without enabled", keyed(RaylineARCPolicySessionsConfig{MaxNewSessionsPerCaller: 20}), "require sessions.enabled"},
		{"negative caller budget", keyed(RaylineARCPolicySessionsConfig{Enabled: true, MaxNewSessionsPerCaller: -1}), "max_new_sessions_per_caller"},
		{"negative idle", keyed(RaylineARCPolicySessionsConfig{Enabled: true, IdleTimeoutSeconds: -1}), "idle_timeout_seconds"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tc.cfg
			err := validateRaylineARCPolicySessions(&cfg)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("err = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestRaylineARCPolicySessionsDefaultsAndYAML(t *testing.T) {
	var cfg RaylineARCPolicyServiceConfig
	if err := yaml.Unmarshal([]byte(`
base_url: https://policy.example.modal.direct
modal_key_env: KEY
modal_secret_env: SECRET
sessions: {enabled: true, max_new_sessions_per_caller: 20}
`), &cfg); err != nil {
		t.Fatal(err)
	}
	if err := validateRaylineARCPolicySessions(&cfg); err != nil || !cfg.Sessions.Enabled || cfg.Sessions.MaxNewSessionsPerCaller != 20 {
		t.Fatalf("decoded %+v, err %v", cfg.Sessions, err)
	}
	if cfg.Sessions.EffectiveIdleTimeoutSeconds() != 900 || cfg.Sessions.EffectiveStartTimeoutSeconds() != 10 {
		t.Fatalf("defaults %d / %d", cfg.Sessions.EffectiveIdleTimeoutSeconds(), cfg.Sessions.EffectiveStartTimeoutSeconds())
	}
	var empty RaylineARCPolicyServiceConfig
	if err := yaml.Unmarshal([]byte("base_url: https://policy.example\nsessions: {}\n"), &empty); err != nil ||
		validateRaylineARCPolicySessions(&empty) != nil || empty.Sessions.Enabled {
		t.Fatalf("an empty sessions block is not today's cell: %+v, %v", empty.Sessions, err)
	}
}
