package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func policyReplicaFailover() RaylineARCEncoderFailoverConfig {
	return RaylineARCEncoderFailoverConfig{
		SchemaVersion:              RaylineARCPolicyFailoverV1,
		UnavailableStatusCodes:     []int{404, 429, 503},
		UnavailableCooldownSeconds: 30,
		MaxRemaps:                  1,
	}
}

func TestRaylineARCPolicyServiceEndpoints(t *testing.T) {
	replicas := []RaylineARCEncoderReplicaConfig{
		{ID: "s1", BaseURL: "https://s1.example", State: RaylineARCEncoderActive},
		{ID: "s2", BaseURL: "https://s2.example", State: RaylineARCEncoderDraining},
	}
	for _, tc := range []struct {
		name string
		cfg  RaylineARCPolicyServiceConfig
		want string
	}{
		{"base_url alone", RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example"}, ""},
		{"neither, as before", RaylineARCPolicyServiceConfig{}, "base_url must be an absolute HTTP(S) URL"},
		{"failover without replicas", RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example", Failover: policyReplicaFailover()}, "failover requires replicas"},
		{"replicas", RaylineARCPolicyServiceConfig{Replicas: replicas, Failover: policyReplicaFailover()}, ""},
		{"both", RaylineARCPolicyServiceConfig{BaseURL: "https://policy.example", Replicas: replicas, Failover: policyReplicaFailover()}, "exactly one of base_url or replicas"},
		{"replicas without failover", RaylineARCPolicyServiceConfig{Replicas: replicas}, "failover.schema_version"},
		{"encoder schema", RaylineARCPolicyServiceConfig{Replicas: replicas, Failover: func() RaylineARCEncoderFailoverConfig {
			f := policyReplicaFailover()
			f.SchemaVersion = RaylineARCEncoderFailoverV1
			return f
		}()}, "failover.schema_version"},
		{"session_busy status", RaylineARCPolicyServiceConfig{Replicas: replicas, Failover: func() RaylineARCEncoderFailoverConfig {
			f := policyReplicaFailover()
			f.UnavailableStatusCodes = []int{409}
			return f
		}()}, "failover unavailable status must be one of"},
		{"no active replica", RaylineARCPolicyServiceConfig{Replicas: replicas[1:], Failover: policyReplicaFailover()}, "at least one active member"},
		{"duplicated id", RaylineARCPolicyServiceConfig{Replicas: []RaylineARCEncoderReplicaConfig{replicas[0], {ID: "s1", BaseURL: "https://s3.example", State: RaylineARCEncoderActive}}, Failover: policyReplicaFailover()}, "duplicated"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tc.cfg
			err := validateRaylineARCPolicyServiceEndpoints(&cfg)
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

func TestRaylineARCPolicyServiceReplicasDecodeFromYAML(t *testing.T) {
	var cfg RaylineARCPolicyServiceConfig
	err := yaml.Unmarshal([]byte(`
replicas:
  - {id: s1, base_url: "https://s1.example", state: active}
  - {id: s2, base_url: "https://s2.example", state: active}
failover:
  schema_version: rayline.arc.policy-failover.v1
  unavailable_status_codes: [503]
  unavailable_cooldown_seconds: 30
  max_remaps: 1
`), &cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateRaylineARCPolicyServiceEndpoints(&cfg); err != nil || len(cfg.Replicas) != 2 || cfg.BaseURL != "" {
		t.Fatalf("decoded %+v, err %v", cfg, err)
	}
}
