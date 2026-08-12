package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestRaylineRemoteConfigCanonicalRoundTripKeepsOnlySecretReferences(
	t *testing.T,
) {
	original := validRaylineRemoteDecision().Algorithm
	encoded, err := yaml.Marshal(original)
	if err != nil {
		t.Fatalf("yaml.Marshal() error = %v", err)
	}
	content := string(encoded)
	for _, reference := range []string{
		"api_key_env: RAYLINE_API_KEY",
		"episode_hmac_key_env: RAYLINE_EPISODE_HMAC_KEY",
	} {
		if !strings.Contains(content, reference) {
			t.Fatalf(
				"serialized remote config omitted %q:\n%s",
				reference,
				content,
			)
		}
	}
	if strings.Contains(content, "api_key:") ||
		strings.Contains(content, "hmac_key:") {
		t.Fatalf("serialized remote config contains secret material:\n%s", content)
	}

	var decoded AlgorithmConfig
	if err := yaml.UnmarshalStrict(encoded, &decoded); err != nil {
		t.Fatalf("yaml.UnmarshalStrict() error = %v", err)
	}
	if !reflect.DeepEqual(*original, decoded) {
		t.Fatalf(
			"remote algorithm did not round-trip:\noriginal=%#v\ndecoded=%#v",
			*original,
			decoded,
		)
	}
}

func TestValidateRaylineRemoteAlgorithmConfig(t *testing.T) {
	decision := validRaylineRemoteDecision()
	if err := validateDecisionAlgorithmConfig(
		decision.Name,
		decision.ModelRefs,
		decision.Algorithm,
	); err != nil {
		t.Fatalf("valid remote config rejected: %v", err)
	}
	if err := validateRaylineRemoteDecisionContract(
		&RouterConfig{},
		decision,
	); err != nil {
		t.Fatalf("valid remote decision contract rejected: %v", err)
	}
}

type remoteConfigInvalidCase struct {
	name    string
	mutate  func(*Decision)
	wantErr string
}

var remoteConfigInvalidCases = []remoteConfigInvalidCase{
	{
		name: "missing block",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote = nil
		},
		wantErr: "configuration is required",
	},
	{
		name: "fallback mode",
		mutate: func(decision *Decision) {
			decision.Algorithm.OnError = "skip"
		},
		wantErr: "requires algorithm.on_error=fail_closed",
	},
	{
		name: "mutable bundle",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.BundleVersion = "latest"
		},
		wantErr: "cannot use mutable value",
	},
	{
		name: "inline URL credentials",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.BaseURL = "https://user:secret@router.example"
		},
		wantErr: "cannot contain credentials",
	},
	{
		name: "plaintext base_url without opt-in",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.BaseURL = "http://rayline-router:8000"
		},
		wantErr: "allow_insecure_transport",
	},
	{
		name: "invalid secret reference",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.APIKeyEnv = "literal-secret!"
		},
		wantErr: "api_key_env",
	},
	{
		name: "reused secret reference",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.EpisodeHMACKeyEnv = decision.Algorithm.RaylineRemote.APIKeyEnv
		},
		wantErr: "distinct secrets",
	},
	{
		name: "uppercase header",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.EpisodeIDHeader = "X-Rayline-Episode"
		},
		wantErr: "lowercase HTTP field name",
	},
	{
		name: "connect exceeds request timeout",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.ConnectTimeoutMS = 1001
		},
		wantErr: "cannot exceed request_timeout_ms",
	},
	{
		name: "lease does not cover request budget",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.RequestTimeoutMS = 30000
			decision.Algorithm.RaylineRemote.LeaseTTLSeconds = 30
		},
		wantErr: "cover at least two",
	},
	{
		name: "too many retries",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.MaxRetries = 3
		},
		wantErr: "max_retries",
	},
	{
		name: "incomplete worker map",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.Workers = decision.Algorithm.RaylineRemote.Workers[:1]
		},
		wantErr: "workers must contain between",
	},
	{
		name: "unknown mapped model",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.Workers[1].Model = "other"
		},
		wantErr: "must reference a decision modelRef",
	},
	{
		name: "duplicate worker id",
		mutate: func(decision *Decision) {
			decision.Algorithm.RaylineRemote.Workers[1].ID = decision.Algorithm.RaylineRemote.Workers[0].ID
		},
		wantErr: "worker ids must be unique",
	},
}

func TestValidateRaylineRemoteAlgorithmConfigRejectsInvalidContracts(
	t *testing.T,
) {
	for _, test := range remoteConfigInvalidCases {
		t.Run(test.name, func(t *testing.T) {
			decision := validRaylineRemoteDecision()
			test.mutate(&decision)
			err := validateDecisionAlgorithmConfig(
				decision.Name,
				decision.ModelRefs,
				decision.Algorithm,
			)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

// Prompt bodies and tool schemas cross this hop verbatim, so a plaintext hop
// has to be a deliberate, written-down operator choice rather than a default.
func TestValidateRaylineRemoteAlgorithmConfigAllowsOptedInPlaintextTransport(
	t *testing.T,
) {
	decision := validRaylineRemoteDecision()
	decision.Algorithm.RaylineRemote.BaseURL = "http://rayline-router:8000"
	decision.Algorithm.RaylineRemote.AllowInsecureTransport = true
	if err := validateDecisionAlgorithmConfig(
		decision.Name,
		decision.ModelRefs,
		decision.Algorithm,
	); err != nil {
		t.Fatalf("opted-in plaintext remote config rejected: %v", err)
	}
}

func TestValidateRaylineRemoteAlgorithmConfigIgnoresOptInForTLSTransport(
	t *testing.T,
) {
	decision := validRaylineRemoteDecision()
	decision.Algorithm.RaylineRemote.AllowInsecureTransport = true
	if err := validateDecisionAlgorithmConfig(
		decision.Name,
		decision.ModelRefs,
		decision.Algorithm,
	); err != nil {
		t.Fatalf("https remote config with opt-in rejected: %v", err)
	}
}

func TestValidateRaylineRemoteDecisionRejectsLearningReplayAndAutoAlias(
	t *testing.T,
) {
	tests := []struct {
		name    string
		mutate  func(*RouterConfig, *Decision)
		wantErr string
	}{
		{
			name: "learning applies",
			mutate: func(_ *RouterConfig, decision *Decision) {
				decision.Adaptations.Mode = ""
			},
			wantErr: "requires adaptations.mode=bypass",
		},
		{
			name: "router replay enabled",
			mutate: func(cfg *RouterConfig, _ *Decision) {
				cfg.RouterReplay.Enabled = true
			},
			wantErr: "requires router_replay disabled",
		},
		{
			name: "auto alias",
			mutate: func(_ *RouterConfig, decision *Decision) {
				decision.ModelRefs[1].Model = "auto"
				decision.Algorithm.RaylineRemote.Workers[1].Model = "auto"
			},
			wantErr: "collides with an auto-routing alias",
		},
		{
			name: "LoRA alias",
			mutate: func(_ *RouterConfig, decision *Decision) {
				decision.ModelRefs[1].LoRAName = "adapter"
			},
			wantErr: "does not support LoRA",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := validRaylineRemoteDecision()
			cfg := &RouterConfig{}
			test.mutate(cfg, &decision)
			err := validateRaylineRemoteDecisionContract(cfg, decision)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func validRaylineRemoteDecision() Decision {
	useReasoning := false
	return Decision{
		Name: "remote-route",
		ModelRefs: []ModelRef{
			{Model: "model-a", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: &useReasoning,
			}},
			{Model: "model-b", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: &useReasoning,
			}},
		},
		Adaptations: DecisionAdaptationsConfig{
			Mode: DecisionAdaptationModeBypass,
		},
		Algorithm: &AlgorithmConfig{
			Type:    RaylineRemoteAlgorithmType,
			OnError: "fail_closed",
			RaylineRemote: &RaylineRemoteAlgorithmConfig{
				BaseURL:           "https://rayline-router:8443",
				BundleVersion:     "bundle.test.v1",
				APIKeyEnv:         "RAYLINE_API_KEY",
				EpisodeIDHeader:   "x-rayline-episode-id",
				EpisodeHMACKeyEnv: "RAYLINE_EPISODE_HMAC_KEY",
				DecisionIDHeader:  "x-rayline-route-id",
				ConnectTimeoutMS:  250,
				RequestTimeoutMS:  1000,
				LeaseTTLSeconds:   30,
				MaxRetries:        1,
				Workers: []RaylineRemoteWorkerConfig{
					{ID: "worker-a", Model: "model-a"},
					{ID: "worker-b", Model: "model-b"},
				},
			},
		},
	}
}
