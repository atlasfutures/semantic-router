package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestValidateRaylineARCDecisionContract(t *testing.T) {
	decision := validRaylineARCDecision()
	if err := validateDecisionAlgorithmConfig(decision.Name, decision.ModelRefs, decision.Algorithm); err != nil {
		t.Fatalf("validateDecisionAlgorithmConfig() error = %v", err)
	}
	if err := validateRaylineARCDecisionContract(decision); err != nil {
		t.Fatalf("validateRaylineARCDecisionContract() error = %v", err)
	}
}

func TestRaylineARCConfigCanonicalRoundTripKeepsOnlyCredentialReference(t *testing.T) {
	original := validRaylineARCDecision().Algorithm
	encoded, err := yaml.Marshal(original)
	if err != nil {
		t.Fatalf("yaml.Marshal() error = %v", err)
	}
	content := string(encoded)
	if strings.Contains(content, "\n      password:") || strings.Contains(content, "\n    password:") {
		t.Fatalf("serialized ARC config contains a password value:\n%s", content)
	}
	if !strings.Contains(content, "password_env: RAYLINE_ARC_REDIS_PASSWORD") {
		t.Fatalf("serialized ARC config omitted password_env reference:\n%s", content)
	}

	var decoded AlgorithmConfig
	if err := yaml.UnmarshalStrict(encoded, &decoded); err != nil {
		t.Fatalf("yaml.UnmarshalStrict() error = %v", err)
	}
	if !reflect.DeepEqual(*original, decoded) {
		t.Fatalf("ARC algorithm did not round-trip:\noriginal=%#v\ndecoded=%#v", *original, decoded)
	}
}

func TestValidateRaylineARCAlgorithmConfigRejectsInvalidContracts(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Decision)
		wantErr string
	}{
		{
			name: "missing block",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC = nil
			},
			wantErr: "configuration is required",
		},
		{
			name: "fallback error mode",
			mutate: func(decision *Decision) {
				decision.Algorithm.OnError = "skip"
			},
			wantErr: "requires algorithm.on_error=fail_closed",
		},
		{
			name: "mutable artifact revision",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.ArtifactRevision = "latest"
			},
			wantErr: "cannot use mutable value",
		},
		{
			name: "unpinned model revision",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Encoder.ModelRevision = "main"
			},
			wantErr: "model_revision must be",
		},
		{
			name: "unknown capability",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Encoder.RequiredCapabilities = []string{"unknown"}
			},
			wantErr: "unsupported capability",
		},
		{
			name: "connect exceeds total timeout",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Encoder.ConnectTimeoutSeconds = 181
			},
			wantErr: "cannot exceed total_timeout_seconds",
		},
		{
			name: "memory outside development",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Episode.Backend = RaylineARCBackendMemory
				decision.Algorithm.RaylineARC.Episode.DevelopmentMode = false
			},
			wantErr: "requires development_mode=true",
		},
		{
			name: "missing redis address",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Episode.Redis.Address = ""
			},
			wantErr: "redis.address",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := validRaylineARCDecision()
			test.mutate(&decision)
			err := validateDecisionAlgorithmConfig(decision.Name, decision.ModelRefs, decision.Algorithm)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestValidateRaylineARCDecisionRejectsLearningAndCandidateDrift(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Decision)
		wantErr string
	}{
		{
			name: "learning defaults to apply",
			mutate: func(decision *Decision) {
				decision.Adaptations.Mode = ""
			},
			wantErr: "requires adaptations.mode=bypass",
		},
		{
			name: "single candidate",
			mutate: func(decision *Decision) {
				decision.ModelRefs = decision.ModelRefs[:1]
			},
			wantErr: "requires at least two modelRefs",
		},
		{
			name: "duplicate candidate",
			mutate: func(decision *Decision) {
				decision.ModelRefs[1].Model = decision.ModelRefs[0].Model
			},
			wantErr: "requires unique modelRefs",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := validRaylineARCDecision()
			test.mutate(&decision)
			err := validateRaylineARCDecisionContract(decision)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func validRaylineARCDecision() Decision {
	return Decision{
		Name: "arc-route",
		ModelRefs: []ModelRef{
			{Model: "public-arm-a"},
			{Model: "public-arm-b"},
		},
		Adaptations: DecisionAdaptationsConfig{Mode: DecisionAdaptationModeBypass},
		Algorithm: &AlgorithmConfig{
			Type:    RaylineARCAlgorithmType,
			OnError: "fail_closed",
			RaylineARC: &RaylineARCAlgorithmConfig{
				ArtifactDir:      "/var/lib/vllm-sr/rayline-arc",
				ArtifactRevision: "public-synthetic-v1",
				Encoder: RaylineARCEncoderConfig{
					BaseURL:               "http://rayline-arc-encoder:8000",
					Model:                 RaylineARCEncoderModel,
					ModelRevision:         RaylineARCEncoderModelRevision,
					ExpectedBuildID:       "vllm@public-synthetic-build",
					ExpectedPluginVersion: "rayline-arc-io@0.1.0",
					SerializerVersion:     RaylineARCSerializerVersion,
					RequiredCapabilities:  []string{RaylineARCCapabilityPluginMean},
					ConnectTimeoutSeconds: 5,
					TotalTimeoutSeconds:   180,
					MaxRetries:            1,
				},
				Episode: RaylineARCEpisodeConfig{
					IDHeader:              "x-rayline-episode-id",
					Backend:               RaylineARCBackendRedis,
					KeyPrefix:             "vsr:rayline-arc:",
					AcquireTimeoutSeconds: 30,
					LeaseTTLSeconds:       60,
					IdleTTLSeconds:        900,
					MaxInMemoryEpisodes:   1024,
					Redis: RaylineARCRedisConfig{
						Address:     "redis:6379",
						DB:          0,
						PasswordEnv: "RAYLINE_ARC_REDIS_PASSWORD",
						UseTLS:      false,
						PoolSize:    16,
					},
				},
			},
		},
	}
}
