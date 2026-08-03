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
	if err := validateRaylineARCDecisionContract(&RouterConfig{}, decision); err != nil {
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
	if !strings.Contains(content, "modal_key_env: RAYLINE_ARC_MODAL_KEY") ||
		!strings.Contains(content, "modal_secret_env: RAYLINE_ARC_MODAL_SECRET") {
		t.Fatalf("serialized ARC config omitted Modal credential references:\n%s", content)
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
			name: "unknown serving rung",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Encoder.ServingRung = "C"
			},
			wantErr: "serving_rung must be",
		},
		{
			name: "serving rung capability mismatch",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Encoder.ServingRung = RaylineARCServingRungB
			},
			wantErr: "requires pooling capability",
		},
		{
			name: "connect exceeds total timeout",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Encoder.ConnectTimeoutSeconds = 181
			},
			wantErr: "cannot exceed total_timeout_seconds",
		},
		{
			name: "unpaired Modal proxy credential",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Encoder.ModalSecretEnv = ""
			},
			wantErr: "must be configured together",
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

func TestValidateRaylineARCAlgorithmConfigRejectsResumableWithoutCausalMean(t *testing.T) {
	decision := validRaylineARCDecision()
	decision.Algorithm.RaylineARC.Encoder.RequiredCapabilities = []string{
		RaylineARCCapabilityResumableMean,
	}
	err := validateDecisionAlgorithmConfig(decision.Name, decision.ModelRefs, decision.Algorithm)
	if err == nil || !strings.Contains(err.Error(), "requires \"chunked_causal_mean\"") {
		t.Fatalf("error = %v, want resumable dependency error", err)
	}
}

func TestValidateRaylineARCAlgorithmConfigAcceptsRetainedSession(t *testing.T) {
	decision := validRaylineARCDecision()
	decision.Algorithm.RaylineARC.Encoder.ServingRung = RaylineARCServingRungB
	decision.Algorithm.RaylineARC.Encoder.RequiredCapabilities = []string{
		RaylineARCCapabilityChunkedMean,
		RaylineARCCapabilityResumableMean,
	}
	if err := validateDecisionAlgorithmConfig(
		decision.Name,
		decision.ModelRefs,
		decision.Algorithm,
	); err != nil {
		t.Fatal(err)
	}
}

func TestValidateRaylineARCAlgorithmConfigAcceptsVersionedReplicaMembership(
	t *testing.T,
) {
	decision := validReplicatedRaylineARCDecision()
	if err := validateDecisionAlgorithmConfig(
		decision.Name,
		decision.ModelRefs,
		decision.Algorithm,
	); err != nil {
		t.Fatalf("replicated ARC config rejected: %v", err)
	}
	encoded, err := yaml.Marshal(decision.Algorithm)
	if err != nil {
		t.Fatal(err)
	}
	var decoded AlgorithmConfig
	if err := yaml.UnmarshalStrict(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(*decision.Algorithm, decoded) {
		t.Fatalf("replicated ARC config did not round-trip")
	}
}

func TestValidateRaylineARCAlgorithmConfigRejectsUnsafeReplicaContracts(
	t *testing.T,
) {
	tests := []struct {
		name    string
		mutate  func(*RaylineARCEncoderConfig)
		wantErr string
	}{
		{
			name: "single URL and replicas",
			mutate: func(cfg *RaylineARCEncoderConfig) {
				cfg.BaseURL = "http://single.example:8000"
			},
			wantErr: "exactly one of base_url or replicas",
		},
		{
			name: "duplicate ID",
			mutate: func(cfg *RaylineARCEncoderConfig) {
				cfg.Replicas[1].ID = cfg.Replicas[0].ID
			},
			wantErr: "duplicated",
		},
		{
			name: "duplicate URL",
			mutate: func(cfg *RaylineARCEncoderConfig) {
				cfg.Replicas[1].BaseURL = cfg.Replicas[0].BaseURL + "/"
			},
			wantErr: "base_url is duplicated",
		},
		{
			name: "no active member",
			mutate: func(cfg *RaylineARCEncoderConfig) {
				cfg.Replicas[0].State = RaylineARCEncoderDraining
				cfg.Replicas[1].State = RaylineARCEncoderDraining
			},
			wantErr: "at least one active",
		},
		{
			name: "stateless rung",
			mutate: func(cfg *RaylineARCEncoderConfig) {
				cfg.ServingRung = RaylineARCServingRungA
			},
			wantErr: "retained serving rung B",
		},
		{
			name: "internal retries",
			mutate: func(cfg *RaylineARCEncoderConfig) {
				cfg.MaxRetries = 1
			},
			wantErr: "max_retries=0",
		},
		{
			name: "unknown schema",
			mutate: func(cfg *RaylineARCEncoderConfig) {
				cfg.Failover.SchemaVersion = "rayline.arc.encoder-failover.v2"
			},
			wantErr: "schema_version",
		},
		{
			name: "ambiguous remap count",
			mutate: func(cfg *RaylineARCEncoderConfig) {
				cfg.Failover.MaxRemaps = 2
			},
			wantErr: "max_remaps must be 1",
		},
		{
			name: "invalid unavailable status",
			mutate: func(cfg *RaylineARCEncoderConfig) {
				cfg.Failover.UnavailableStatusCodes = []int{200}
			},
			wantErr: "between 400 and 599",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := validReplicatedRaylineARCDecision()
			test.mutate(&decision.Algorithm.RaylineARC.Encoder)
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

func TestValidateRaylineARCAlgorithmConfigRequiresReplicaCloseProtocol(
	t *testing.T,
) {
	decision := validReplicatedRaylineARCDecision()
	decision.Algorithm.RaylineARC.Episode.CloseHeader = ""
	err := validateDecisionAlgorithmConfig(
		decision.Name,
		decision.ModelRefs,
		decision.Algorithm,
	)
	if err == nil || !strings.Contains(err.Error(), "close_header is required") {
		t.Fatalf("error = %v, want required close-header failure", err)
	}
}

func TestValidateRaylineARCAlgorithmConfigAcceptsDynamicReplicaMembership(
	t *testing.T,
) {
	decision := validDynamicRaylineARCDecision()
	if err := validateDecisionAlgorithmConfig(
		decision.Name,
		decision.ModelRefs,
		decision.Algorithm,
	); err != nil {
		t.Fatalf("dynamic ARC config rejected: %v", err)
	}
}

func TestValidateRaylineARCAlgorithmConfigRejectsUnsafeDynamicMembership(
	t *testing.T,
) {
	tests := []struct {
		name    string
		mutate  func(*Decision)
		wantErr string
	}{
		{
			name: "non redis source",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Encoder.Membership.Source = "dns"
			},
			wantErr: "membership.source",
		},
		{
			name: "memory episode store",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Episode.Backend = RaylineARCBackendMemory
				decision.Algorithm.RaylineARC.Episode.DevelopmentMode = true
				decision.Algorithm.RaylineARC.Episode.MaxInMemoryEpisodes = 4
			},
			wantErr: "redis backend",
		},
		{
			name: "missing close header",
			mutate: func(decision *Decision) {
				decision.Algorithm.RaylineARC.Episode.CloseHeader = ""
			},
			wantErr: "close_header is required",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := validDynamicRaylineARCDecision()
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
		{
			name: "auto alias candidate",
			mutate: func(decision *Decision) {
				decision.ModelRefs[1].Model = "auto"
			},
			wantErr: "collides with an auto-routing alias",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := validRaylineARCDecision()
			test.mutate(&decision)
			err := validateRaylineARCDecisionContract(&RouterConfig{}, decision)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestValidateRaylineARCDecisionRejectsEnabledRouterReplay(t *testing.T) {
	decision := validRaylineARCDecision()
	cfg := &RouterConfig{}
	cfg.RouterReplay.Enabled = true
	cfg.Decisions = []Decision{decision}

	err := validateRaylineARCDecisionContract(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "router_replay disabled") {
		t.Fatalf("error = %v, want router_replay rejection", err)
	}

	cfg.Decisions[0].Plugins = []DecisionPlugin{
		{
			Type: DecisionPluginRouterReplay,
			Configuration: MustStructuredPayload(
				map[string]interface{}{"enabled": false},
			),
		},
	}
	if err := validateRaylineARCDecisionContract(cfg, cfg.Decisions[0]); err != nil {
		t.Fatalf("decision-level replay disable was rejected: %v", err)
	}
}

func TestValidateRaylineARCRedisAddressPorts(t *testing.T) {
	tests := []struct {
		address string
		valid   bool
	}{
		{"redis:6379", true},
		{"[::1]:6379", true},
		{"redis:notaport", false},
		{"redis:0", false},
		{"redis:70000", false},
		{"::1:6379", false},
		{":6379", false},
		{"redis", false},
	}
	for _, test := range tests {
		t.Run(test.address, func(t *testing.T) {
			decision := validRaylineARCDecision()
			decision.Algorithm.RaylineARC.Episode.Redis.Address = test.address
			err := validateDecisionAlgorithmConfig(
				decision.Name,
				decision.ModelRefs,
				decision.Algorithm,
			)
			if test.valid && err != nil {
				t.Fatalf("address %q rejected: %v", test.address, err)
			}
			if !test.valid && err == nil {
				t.Fatalf("address %q accepted", test.address)
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
					ServingRung:           RaylineARCServingRungA,
					RequiredCapabilities:  []string{RaylineARCCapabilityPluginMean},
					ModalKeyEnv:           "RAYLINE_ARC_MODAL_KEY",
					ModalSecretEnv:        "RAYLINE_ARC_MODAL_SECRET",
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

func validReplicatedRaylineARCDecision() Decision {
	decision := validRaylineARCDecision()
	encoder := &decision.Algorithm.RaylineARC.Encoder
	encoder.BaseURL = ""
	encoder.Replicas = []RaylineARCEncoderReplicaConfig{
		{
			ID:      "encoder-a",
			BaseURL: "http://rayline-arc-encoder-a:8000",
			State:   RaylineARCEncoderActive,
		},
		{
			ID:      "encoder-b",
			BaseURL: "http://rayline-arc-encoder-b:8000",
			State:   RaylineARCEncoderActive,
		},
	}
	encoder.Failover = RaylineARCEncoderFailoverConfig{
		SchemaVersion:              RaylineARCEncoderFailoverV1,
		UnavailableStatusCodes:     []int{404, 410, 502, 503, 504},
		UnavailableCooldownSeconds: 30,
		MaxRemaps:                  1,
	}
	encoder.ServingRung = RaylineARCServingRungB
	encoder.RequiredCapabilities = []string{
		RaylineARCCapabilityChunkedMean,
		RaylineARCCapabilityResumableMean,
	}
	encoder.MaxRetries = 0
	decision.Algorithm.RaylineARC.Episode.CloseHeader = "x-rayline-episode-close"
	return decision
}

func validDynamicRaylineARCDecision() Decision {
	decision := validReplicatedRaylineARCDecision()
	encoder := &decision.Algorithm.RaylineARC.Encoder
	encoder.Replicas = nil
	encoder.Membership = RaylineARCEncoderMembershipConfig{
		SchemaVersion:  RaylineARCEncoderMembershipV1,
		Source:         RaylineARCEncoderMembershipRedis,
		RefreshSeconds: 5,
	}
	return decision
}
