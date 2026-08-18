package config

import (
	"fmt"
	"net"
	"net/url"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

const (
	RaylineARCAlgorithmType        = "rayline_arc"
	RaylineARCEncoderModel         = "Qwen/Qwen3.5-0.8B"
	RaylineARCEncoderModelRevision = "2fc06364715b967f1860aea9cf38778875588b17"
	RaylineARCSerializerVersion    = "mtrouter-token-blocks-v2"
	RaylineARCServingRungA         = "A"
	RaylineARCServingRungB         = "B"
	RaylineARCBackendRedis         = "redis"
	RaylineARCBackendMemory        = "memory"
	RaylineARCCapabilityPluginMean = "all_plugin_mean"
	// RaylineARCCapabilityChunkedMean is the only Rung B capability the
	// pinned plugin can report; prefix-cached MEAN stays unconfigurable
	// until the Rung C phase gate opens.
	RaylineARCCapabilityChunkedMean   = "chunked_causal_mean"
	RaylineARCCapabilityResumableMean = "resumable_causal_mean"
	maxRaylineARCEncoderRetries       = 3
	maxRaylineARCConfigStringLength   = 512
	maxRaylineARCRequiredCapability   = 8
	maxNetworkPort                    = 65535
	// A router cap above the encoder's own ingress cap cannot protect it: the
	// surplus simply queues inside the encoder container where the router
	// cannot see it, which is the condition this knob exists to remove.
	maxRaylineARCInflightEncoderCalls = 32
)

var (
	raylineARCHeaderNamePattern = regexp.MustCompile(`^[!#$%&'*+.^_` + "`" + `|~0-9a-z-]+$`)
	raylineARCEnvNamePattern    = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
)

// RaylineARCAlgorithmConfig configures the artifact-verified Rayline ARC
// orchestrator. The mounted manifest remains authoritative for arm identities,
// weights, policy, and encoder contract.
type RaylineARCAlgorithmConfig struct {
	ArtifactDir      string                  `yaml:"artifact_dir"`
	ArtifactRevision string                  `yaml:"artifact_revision"`
	Encoder          RaylineARCEncoderConfig `yaml:"encoder"`
	Episode          RaylineARCEpisodeConfig `yaml:"episode"`
	// IncludeSystemText shows the selector the conversation-opening system
	// prompt instead of discarding it: the Anthropic system field, the
	// Responses instructions field, and the leading run of system or developer
	// messages. The text is folded into the user turn it governs, which leaves
	// the turn count, the roles, and the encoder wire contract untouched.
	//
	// Off by default. Every consultation the trained selector has served to
	// date was made without the opening prompt, so turning this on changes
	// what the model is asked, and that belongs behind a deliberate opt-in
	// until a measurement says which way is better.
	IncludeSystemText bool `yaml:"include_system_text,omitempty"`
	// DropMidConversationSystemText discards system text that arrives after
	// the conversation has started: an Anthropic mid_conv_system block, or a
	// system or developer message that follows a user or assistant message.
	//
	// Off by default, so that text reaches the selector. It is a correction
	// aimed at the very next reply, which is the signal the router exists to
	// read, and it is short enough that folding it in barely moves the token
	// count. This field is the kill switch for that, worded as a negative so
	// the zero value keeps the text.
	DropMidConversationSystemText bool `yaml:"drop_mid_conversation_system_text,omitempty"`
}

// RaylineARCEncoderConfig pins the dedicated vLLM pooling service contract.
type RaylineARCEncoderConfig struct {
	BaseURL               string                            `yaml:"base_url"`
	Replicas              []RaylineARCEncoderReplicaConfig  `yaml:"replicas,omitempty"`
	Membership            RaylineARCEncoderMembershipConfig `yaml:"membership,omitempty"`
	Failover              RaylineARCEncoderFailoverConfig   `yaml:"failover,omitempty"`
	Model                 string                            `yaml:"model"`
	ModelRevision         string                            `yaml:"model_revision"`
	ExpectedBuildID       string                            `yaml:"expected_build_id"`
	ExpectedPluginVersion string                            `yaml:"expected_io_plugin_version"`
	SerializerVersion     string                            `yaml:"serializer_version"`
	ServingRung           string                            `yaml:"serving_rung"`
	RequiredCapabilities  []string                          `yaml:"required_pooling_capabilities"`
	ModalKeyEnv           string                            `yaml:"modal_key_env,omitempty"`
	ModalSecretEnv        string                            `yaml:"modal_secret_env,omitempty"`
	ConnectTimeoutSeconds int                               `yaml:"connect_timeout_seconds"`
	TotalTimeoutSeconds   int                               `yaml:"total_timeout_seconds"`
	MaxRetries            int                               `yaml:"max_retries"`
	// MaxInflightEncoderCalls caps concurrent encoder calls per router process
	// and sheds the surplus immediately rather than queueing it. Zero, the
	// default, leaves admission control off so that existing deployments keep
	// their current behaviour until an operator opts in.
	MaxInflightEncoderCalls int `yaml:"max_inflight_encoder_calls,omitempty"`
}

// RaylineARCEpisodeConfig configures serialized, fenced episode state.
// Memory is permitted only when DevelopmentMode is explicitly true.
type RaylineARCEpisodeConfig struct {
	IDHeader              string                `yaml:"id_header"`
	CloseHeader           string                `yaml:"close_header,omitempty"`
	Backend               string                `yaml:"backend"`
	KeyPrefix             string                `yaml:"key_prefix"`
	AcquireTimeoutSeconds int                   `yaml:"acquire_timeout_seconds"`
	LeaseTTLSeconds       int                   `yaml:"lease_ttl_seconds"`
	IdleTTLSeconds        int                   `yaml:"idle_ttl_seconds"`
	MaxInMemoryEpisodes   int                   `yaml:"max_in_memory_episodes"`
	DevelopmentMode       bool                  `yaml:"development_mode,omitempty"`
	Redis                 RaylineARCRedisConfig `yaml:"redis,omitempty"`
}

// RaylineARCRedisConfig carries non-secret Redis connection settings. Passwords
// are resolved only from PasswordEnv so canonical serialization never contains
// credential material.
type RaylineARCRedisConfig struct {
	Address     string `yaml:"address,omitempty"`
	DB          int    `yaml:"db,omitempty"`
	PasswordEnv string `yaml:"password_env,omitempty"`
	UseTLS      bool   `yaml:"use_tls,omitempty"`
	PoolSize    int    `yaml:"pool_size,omitempty"`
}

func validateRaylineARCAlgorithmConfig(cfg *RaylineARCAlgorithmConfig) error {
	if cfg == nil {
		return fmt.Errorf("configuration is required")
	}
	if err := validateRaylineARCArtifactConfig(cfg); err != nil {
		return err
	}
	if err := validateRaylineARCEncoderConfig(cfg.Encoder); err != nil {
		return fmt.Errorf("encoder: %w", err)
	}
	if err := validateRaylineARCEpisodeConfig(cfg.Episode); err != nil {
		return fmt.Errorf("episode: %w", err)
	}
	if cfg.Encoder.usesReplicaMembership() && cfg.Episode.CloseHeader == "" {
		return fmt.Errorf("episode: close_header is required with encoder replicas")
	}
	if !cfg.Encoder.usesReplicaMembership() && cfg.Episode.CloseHeader != "" {
		return fmt.Errorf("episode: close_header requires encoder replicas")
	}
	if cfg.Encoder.usesDynamicMembership() &&
		cfg.Episode.Backend != RaylineARCBackendRedis {
		return fmt.Errorf("episode: redis backend is required with dynamic encoder membership")
	}
	return nil
}

func validateRaylineARCArtifactConfig(cfg *RaylineARCAlgorithmConfig) error {
	artifactDir := strings.TrimSpace(cfg.ArtifactDir)
	if artifactDir == "" {
		return fmt.Errorf("artifact_dir cannot be empty")
	}
	if len(artifactDir) > maxRaylineARCConfigStringLength || !filepath.IsAbs(artifactDir) {
		return fmt.Errorf("artifact_dir must be an absolute path of at most %d bytes", maxRaylineARCConfigStringLength)
	}
	return validateImmutableRaylineARCPin("artifact_revision", cfg.ArtifactRevision)
}

func validateRaylineARCEncoderConfig(cfg RaylineARCEncoderConfig) error {
	if err := validateRaylineARCEncoderMembership(cfg); err != nil {
		return err
	}
	if cfg.Model != RaylineARCEncoderModel {
		return fmt.Errorf("model must be %q", RaylineARCEncoderModel)
	}
	if cfg.ModelRevision != RaylineARCEncoderModelRevision {
		return fmt.Errorf("model_revision must be %q", RaylineARCEncoderModelRevision)
	}
	if err := validateRaylineARCEncoderPins(cfg); err != nil {
		return err
	}
	if cfg.SerializerVersion != RaylineARCSerializerVersion {
		return fmt.Errorf("serializer_version must be %q", RaylineARCSerializerVersion)
	}
	if err := validateRaylineARCModalAuth(cfg); err != nil {
		return err
	}
	if err := validateRaylineARCCapabilities(cfg.RequiredCapabilities); err != nil {
		return err
	}
	if err := validateRaylineARCServingRung(
		cfg.ServingRung,
		cfg.RequiredCapabilities,
	); err != nil {
		return err
	}
	return validateRaylineARCEncoderTimeouts(cfg)
}

func validateRaylineARCServingRung(
	servingRung string,
	capabilities []string,
) error {
	requiredCapability := ""
	switch servingRung {
	case RaylineARCServingRungA:
		requiredCapability = RaylineARCCapabilityPluginMean
	case RaylineARCServingRungB:
		requiredCapability = RaylineARCCapabilityChunkedMean
	default:
		return fmt.Errorf(
			"serving_rung must be %q or %q",
			RaylineARCServingRungA,
			RaylineARCServingRungB,
		)
	}
	for _, capability := range capabilities {
		if capability == requiredCapability {
			return nil
		}
	}
	return fmt.Errorf(
		"serving_rung %q requires pooling capability %q",
		servingRung,
		requiredCapability,
	)
}

func validateRaylineARCModalAuth(cfg RaylineARCEncoderConfig) error {
	hasKey := cfg.ModalKeyEnv != ""
	hasSecret := cfg.ModalSecretEnv != ""
	if hasKey != hasSecret {
		return fmt.Errorf(
			"modal_key_env and modal_secret_env must be configured together",
		)
	}
	if !hasKey {
		return nil
	}
	if !raylineARCEnvNamePattern.MatchString(cfg.ModalKeyEnv) ||
		!raylineARCEnvNamePattern.MatchString(cfg.ModalSecretEnv) {
		return fmt.Errorf(
			"modal proxy credential fields must name valid environment variables",
		)
	}
	return nil
}

func validateRaylineARCEncoderPins(cfg RaylineARCEncoderConfig) error {
	if err := validateImmutableRaylineARCPin("expected_build_id", cfg.ExpectedBuildID); err != nil {
		return err
	}
	return validateImmutableRaylineARCPin("expected_io_plugin_version", cfg.ExpectedPluginVersion)
}

func validateRaylineARCEncoderTimeouts(cfg RaylineARCEncoderConfig) error {
	if cfg.ConnectTimeoutSeconds <= 0 {
		return fmt.Errorf("connect_timeout_seconds must be positive")
	}
	if cfg.TotalTimeoutSeconds <= 0 {
		return fmt.Errorf("total_timeout_seconds must be positive")
	}
	if cfg.ConnectTimeoutSeconds > cfg.TotalTimeoutSeconds {
		return fmt.Errorf("connect_timeout_seconds cannot exceed total_timeout_seconds")
	}
	if cfg.MaxRetries < 0 || cfg.MaxRetries > maxRaylineARCEncoderRetries {
		return fmt.Errorf("max_retries must be between 0 and %d", maxRaylineARCEncoderRetries)
	}
	if cfg.MaxInflightEncoderCalls < 0 ||
		cfg.MaxInflightEncoderCalls > maxRaylineARCInflightEncoderCalls {
		return fmt.Errorf(
			"max_inflight_encoder_calls must be between 0 and %d",
			maxRaylineARCInflightEncoderCalls,
		)
	}
	return nil
}

func validateRaylineARCBaseURL(raw string) error {
	if len(raw) > maxRaylineARCConfigStringLength {
		return fmt.Errorf("base_url exceeds %d bytes", maxRaylineARCConfigStringLength)
	}
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("base_url must be an absolute HTTP(S) URL")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return fmt.Errorf("base_url cannot contain credentials, a query, or a fragment")
	}
	return nil
}

func validateImmutableRaylineARCPin(name string, value string) error {
	pin := strings.TrimSpace(value)
	if pin == "" {
		return fmt.Errorf("%s cannot be empty", name)
	}
	if len(pin) > maxRaylineARCConfigStringLength || strings.ContainsAny(pin, " \t\r\n") {
		return fmt.Errorf("%s must be a bounded immutable identifier without whitespace", name)
	}
	switch strings.ToLower(pin) {
	case "latest", "main", "master", "head":
		return fmt.Errorf("%s cannot use mutable value %q", name, pin)
	}
	return nil
}

func validateRaylineARCCapabilities(capabilities []string) error {
	if len(capabilities) == 0 || len(capabilities) > maxRaylineARCRequiredCapability {
		return fmt.Errorf("required_pooling_capabilities must contain between 1 and %d entries", maxRaylineARCRequiredCapability)
	}
	allowed := map[string]bool{
		RaylineARCCapabilityPluginMean:    true,
		RaylineARCCapabilityChunkedMean:   true,
		RaylineARCCapabilityResumableMean: true,
	}
	seen := make(map[string]bool, len(capabilities))
	for _, capability := range capabilities {
		if !allowed[capability] {
			return fmt.Errorf("required_pooling_capabilities contains unsupported capability %q", capability)
		}
		if seen[capability] {
			return fmt.Errorf("required_pooling_capabilities contains duplicate capability %q", capability)
		}
		seen[capability] = true
	}
	if seen[RaylineARCCapabilityResumableMean] &&
		!seen[RaylineARCCapabilityChunkedMean] {
		return fmt.Errorf(
			"pooling capability %q requires %q",
			RaylineARCCapabilityResumableMean,
			RaylineARCCapabilityChunkedMean,
		)
	}
	return nil
}

func validateRaylineARCEpisodeConfig(cfg RaylineARCEpisodeConfig) error {
	if err := validateRaylineARCEpisodeFields(cfg); err != nil {
		return err
	}
	switch cfg.Backend {
	case RaylineARCBackendRedis:
		return validateRaylineARCRedisConfig(cfg.Redis)
	case RaylineARCBackendMemory:
		return validateRaylineARCMemoryConfig(cfg)
	default:
		return fmt.Errorf("backend must be %q or %q", RaylineARCBackendRedis, RaylineARCBackendMemory)
	}
}

func validateRaylineARCEpisodeFields(cfg RaylineARCEpisodeConfig) error {
	if !raylineARCHeaderNamePattern.MatchString(cfg.IDHeader) {
		return fmt.Errorf("id_header must be a nonempty lowercase HTTP field name")
	}
	if cfg.CloseHeader != "" {
		if !raylineARCHeaderNamePattern.MatchString(cfg.CloseHeader) {
			return fmt.Errorf("close_header must be a lowercase HTTP field name")
		}
		if cfg.CloseHeader == cfg.IDHeader {
			return fmt.Errorf("close_header must differ from id_header")
		}
	}
	if strings.TrimSpace(cfg.KeyPrefix) == "" || len(cfg.KeyPrefix) > 128 {
		return fmt.Errorf("key_prefix must contain between 1 and 128 bytes")
	}
	if cfg.AcquireTimeoutSeconds <= 0 || cfg.LeaseTTLSeconds <= 0 || cfg.IdleTTLSeconds <= 0 {
		return fmt.Errorf("acquire_timeout_seconds, lease_ttl_seconds, and idle_ttl_seconds must be positive")
	}
	if cfg.IdleTTLSeconds < cfg.LeaseTTLSeconds {
		return fmt.Errorf("idle_ttl_seconds cannot be less than lease_ttl_seconds")
	}
	return nil
}

func validateRaylineARCMemoryConfig(cfg RaylineARCEpisodeConfig) error {
	if !cfg.DevelopmentMode {
		return fmt.Errorf("backend=memory requires development_mode=true")
	}
	if cfg.MaxInMemoryEpisodes <= 0 {
		return fmt.Errorf("max_in_memory_episodes must be positive for backend=memory")
	}
	return nil
}

func validateRaylineARCRedisConfig(cfg RaylineARCRedisConfig) error {
	host, port, err := net.SplitHostPort(strings.TrimSpace(cfg.Address))
	if err != nil || host == "" {
		return fmt.Errorf("redis.address must be a host:port pair")
	}
	portNumber, err := strconv.Atoi(port)
	if err != nil || portNumber < 1 || portNumber > maxNetworkPort {
		return fmt.Errorf("redis.address port must be between 1 and %d", maxNetworkPort)
	}
	if cfg.DB < 0 {
		return fmt.Errorf("redis.db cannot be negative")
	}
	if cfg.PasswordEnv != "" && !raylineARCEnvNamePattern.MatchString(cfg.PasswordEnv) {
		return fmt.Errorf("redis.password_env must be a valid environment variable name")
	}
	if cfg.PoolSize < 0 {
		return fmt.Errorf("redis.pool_size cannot be negative")
	}
	return nil
}

func validateRaylineARCDecisionContract(cfg *RouterConfig, decision Decision) error {
	if decision.Algorithm == nil || strings.TrimSpace(decision.Algorithm.Type) != RaylineARCAlgorithmType {
		return nil
	}
	if decision.Adaptations.EffectiveMode() != DecisionAdaptationModeBypass {
		return fmt.Errorf(
			"decision '%s': algorithm.type=%s requires adaptations.mode=%s so Router Learning cannot override the artifact decision",
			decision.Name,
			RaylineARCAlgorithmType,
			DecisionAdaptationModeBypass,
		)
	}
	if len(decision.ModelRefs) < 2 {
		return fmt.Errorf("decision '%s': algorithm.type=%s requires at least two modelRefs", decision.Name, RaylineARCAlgorithmType)
	}
	seen := make(map[string]bool, len(decision.ModelRefs))
	for _, modelRef := range decision.ModelRefs {
		if seen[modelRef.Model] {
			return fmt.Errorf("decision '%s': algorithm.type=%s requires unique modelRefs in artifact order", decision.Name, RaylineARCAlgorithmType)
		}
		seen[modelRef.Model] = true
		if cfg != nil && cfg.IsAutoModelName(modelRef.Model) {
			return fmt.Errorf(
				"decision '%s': algorithm.type=%s modelRef %q collides with an auto-routing alias, which would bypass artifact-owned dispatch",
				decision.Name,
				RaylineARCAlgorithmType,
				modelRef.Model,
			)
		}
	}
	if replay := cfg.EffectiveRouterReplayConfigForDecision(decision.Name); replay != nil && replay.Enabled {
		return fmt.Errorf(
			"decision '%s': algorithm.type=%s requires router_replay disabled for this decision; episode requests must not be persisted",
			decision.Name,
			RaylineARCAlgorithmType,
		)
	}
	return nil
}
