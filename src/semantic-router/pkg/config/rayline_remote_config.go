package config

import (
	"fmt"
	"regexp"
	"strings"
)

const (
	RaylineRemoteAlgorithmType = "rayline_remote"

	minRaylineRemoteTimeoutMS = 1
	maxRaylineRemoteConnectMS = 10_000
	maxRaylineRemoteRequestMS = 30_000
	minRaylineRemoteLeaseTTL  = 5
	maxRaylineRemoteLeaseTTL  = 300
	maxRaylineRemoteRetries   = 2
	maxRaylineRemoteWorkers   = 128
	maxRaylineRemoteWorkerID  = 256
)

var raylineRemoteWorkerIDPattern = regexp.MustCompile(
	`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$`,
)

// RaylineRemoteAlgorithmConfig configures the Pathfinder-owned remote policy.
// Credential fields name environment variables; secret values are never part
// of the canonical configuration.
type RaylineRemoteAlgorithmConfig struct {
	BaseURL       string `yaml:"base_url"`
	BundleVersion string `yaml:"bundle_version"`
	// AllowInsecureTransport opts a decision into a plaintext base_url. The
	// hop carries the full prompt body and tool schemas, so http:// stays a
	// startup error until an operator writes this flag down.
	AllowInsecureTransport bool                        `yaml:"allow_insecure_transport"`
	APIKeyEnv              string                      `yaml:"api_key_env"`
	EpisodeIDHeader        string                      `yaml:"episode_id_header"`
	EpisodeHMACKeyEnv      string                      `yaml:"episode_hmac_key_env"`
	DecisionIDHeader       string                      `yaml:"decision_id_header"`
	ConnectTimeoutMS       int                         `yaml:"connect_timeout_ms"`
	RequestTimeoutMS       int                         `yaml:"request_timeout_ms"`
	LeaseTTLSeconds        int                         `yaml:"lease_ttl_seconds"`
	MaxRetries             int                         `yaml:"max_retries"`
	Workers                []RaylineRemoteWorkerConfig `yaml:"workers"`
}

// RaylineRemoteWorkerConfig is the one-to-one worker-to-ModelRef dispatch map.
type RaylineRemoteWorkerConfig struct {
	ID    string `yaml:"id"`
	Model string `yaml:"model"`
}

func validateRaylineRemoteAlgorithmConfig(
	cfg *RaylineRemoteAlgorithmConfig,
	modelRefs []ModelRef,
) error {
	if cfg == nil {
		return fmt.Errorf("configuration is required")
	}
	if err := validateRaylineARCBaseURL(cfg.BaseURL); err != nil {
		return err
	}
	if err := validateRaylineRemoteTransport(cfg); err != nil {
		return err
	}
	if err := validateImmutableRaylineARCPin(
		"bundle_version",
		cfg.BundleVersion,
	); err != nil {
		return err
	}
	if err := validateRaylineRemoteAuth(cfg); err != nil {
		return err
	}
	if err := validateRaylineRemoteHeaders(cfg); err != nil {
		return err
	}
	if err := validateRaylineRemoteTimeouts(cfg); err != nil {
		return err
	}
	return validateRaylineRemoteWorkers(cfg.Workers, modelRefs)
}

// validateRaylineRemoteTransport keeps the prompt-carrying hop encrypted
// unless the decision opts in. base_url has already been parsed as an
// absolute http or https URL by this point.
func validateRaylineRemoteTransport(
	cfg *RaylineRemoteAlgorithmConfig,
) error {
	if cfg.AllowInsecureTransport {
		return nil
	}
	if strings.HasPrefix(
		strings.ToLower(strings.TrimSpace(cfg.BaseURL)),
		"http://",
	) {
		return fmt.Errorf(
			"base_url must use https; plaintext requires allow_insecure_transport: true",
		)
	}
	return nil
}

func validateRaylineRemoteAuth(cfg *RaylineRemoteAlgorithmConfig) error {
	for _, field := range []struct {
		name  string
		value string
	}{
		{name: "api_key_env", value: cfg.APIKeyEnv},
		{name: "episode_hmac_key_env", value: cfg.EpisodeHMACKeyEnv},
	} {
		if !raylineARCEnvNamePattern.MatchString(field.value) {
			return fmt.Errorf(
				"%s must name a valid nonempty environment variable",
				field.name,
			)
		}
	}
	if cfg.APIKeyEnv == cfg.EpisodeHMACKeyEnv {
		return fmt.Errorf(
			"api_key_env and episode_hmac_key_env must name distinct secrets",
		)
	}
	return nil
}

func validateRaylineRemoteHeaders(cfg *RaylineRemoteAlgorithmConfig) error {
	for _, field := range []struct {
		name  string
		value string
	}{
		{name: "episode_id_header", value: cfg.EpisodeIDHeader},
		{name: "decision_id_header", value: cfg.DecisionIDHeader},
	} {
		if !raylineARCHeaderNamePattern.MatchString(field.value) {
			return fmt.Errorf(
				"%s must be a nonempty lowercase HTTP field name",
				field.name,
			)
		}
	}
	if cfg.EpisodeIDHeader == cfg.DecisionIDHeader {
		return fmt.Errorf(
			"episode_id_header and decision_id_header must be distinct",
		)
	}
	return nil
}

func validateRaylineRemoteTimeouts(cfg *RaylineRemoteAlgorithmConfig) error {
	if cfg.ConnectTimeoutMS < minRaylineRemoteTimeoutMS ||
		cfg.ConnectTimeoutMS > maxRaylineRemoteConnectMS {
		return fmt.Errorf(
			"connect_timeout_ms must be between %d and %d",
			minRaylineRemoteTimeoutMS,
			maxRaylineRemoteConnectMS,
		)
	}
	if cfg.RequestTimeoutMS < minRaylineRemoteTimeoutMS ||
		cfg.RequestTimeoutMS > maxRaylineRemoteRequestMS {
		return fmt.Errorf(
			"request_timeout_ms must be between %d and %d",
			minRaylineRemoteTimeoutMS,
			maxRaylineRemoteRequestMS,
		)
	}
	if cfg.ConnectTimeoutMS > cfg.RequestTimeoutMS {
		return fmt.Errorf(
			"connect_timeout_ms cannot exceed request_timeout_ms",
		)
	}
	if cfg.LeaseTTLSeconds < minRaylineRemoteLeaseTTL ||
		cfg.LeaseTTLSeconds > maxRaylineRemoteLeaseTTL {
		return fmt.Errorf(
			"lease_ttl_seconds must be between %d and %d",
			minRaylineRemoteLeaseTTL,
			maxRaylineRemoteLeaseTTL,
		)
	}
	if cfg.LeaseTTLSeconds*1000 < cfg.RequestTimeoutMS*2 {
		return fmt.Errorf(
			"lease_ttl_seconds must cover at least two request_timeout_ms budgets",
		)
	}
	if cfg.MaxRetries < 0 || cfg.MaxRetries > maxRaylineRemoteRetries {
		return fmt.Errorf(
			"max_retries must be between 0 and %d",
			maxRaylineRemoteRetries,
		)
	}
	return nil
}

func validateRaylineRemoteWorkers(
	workers []RaylineRemoteWorkerConfig,
	modelRefs []ModelRef,
) error {
	if len(workers) < 2 || len(workers) > maxRaylineRemoteWorkers {
		return fmt.Errorf(
			"workers must contain between 2 and %d entries",
			maxRaylineRemoteWorkers,
		)
	}
	if len(workers) != len(modelRefs) {
		return fmt.Errorf(
			"workers must map the decision's complete modelRefs set one-to-one",
		)
	}
	modelSet := make(map[string]bool, len(modelRefs))
	for _, modelRef := range modelRefs {
		model := strings.TrimSpace(modelRef.Model)
		if model == "" || modelSet[model] {
			return fmt.Errorf("decision modelRefs must be nonempty and unique")
		}
		modelSet[model] = true
	}
	workerIDs := make(map[string]bool, len(workers))
	mappedModels := make(map[string]bool, len(workers))
	for _, worker := range workers {
		if err := validateRaylineRemoteWorker(
			worker,
			modelSet,
			workerIDs,
			mappedModels,
		); err != nil {
			return err
		}
	}
	return nil
}

func validateRaylineRemoteWorker(
	worker RaylineRemoteWorkerConfig,
	modelSet map[string]bool,
	workerIDs map[string]bool,
	mappedModels map[string]bool,
) error {
	if len(worker.ID) > maxRaylineRemoteWorkerID ||
		!raylineRemoteWorkerIDPattern.MatchString(worker.ID) {
		return fmt.Errorf("worker id must be a bounded portable identifier")
	}
	if workerIDs[worker.ID] {
		return fmt.Errorf("worker ids must be unique")
	}
	if !modelSet[worker.Model] {
		return fmt.Errorf("worker model must reference a decision modelRef")
	}
	if mappedModels[worker.Model] {
		return fmt.Errorf("worker models must be unique")
	}
	workerIDs[worker.ID] = true
	mappedModels[worker.Model] = true
	return nil
}

func validateRaylineRemoteDecisionContract(
	cfg *RouterConfig,
	decision Decision,
) error {
	if decision.Algorithm == nil ||
		strings.TrimSpace(decision.Algorithm.Type) !=
			RaylineRemoteAlgorithmType {
		return nil
	}
	if decision.Adaptations.EffectiveMode() != DecisionAdaptationModeBypass {
		return fmt.Errorf(
			"decision '%s': algorithm.type=%s requires adaptations.mode=%s",
			decision.Name,
			RaylineRemoteAlgorithmType,
			DecisionAdaptationModeBypass,
		)
	}
	for _, modelRef := range decision.ModelRefs {
		if modelRef.LoRAName != "" {
			return fmt.Errorf(
				"decision '%s': algorithm.type=%s does not support LoRA or virtual modelRefs",
				decision.Name,
				RaylineRemoteAlgorithmType,
			)
		}
		if cfg != nil && cfg.IsAutoModelName(modelRef.Model) {
			return fmt.Errorf(
				"decision '%s': algorithm.type=%s modelRef collides with an auto-routing alias",
				decision.Name,
				RaylineRemoteAlgorithmType,
			)
		}
	}
	if replay := cfg.EffectiveRouterReplayConfigForDecision(
		decision.Name,
	); replay != nil && replay.Enabled {
		return fmt.Errorf(
			"decision '%s': algorithm.type=%s requires router_replay disabled for this decision",
			decision.Name,
			RaylineRemoteAlgorithmType,
		)
	}
	return nil
}

func validateRaylineRemoteSpecializedAlgorithmConfig(
	decisionName string,
	modelRefs []ModelRef,
	algorithm *AlgorithmConfig,
) error {
	if algorithm.OnError != "fail_closed" {
		return fmt.Errorf(
			"decision '%s': algorithm.type=%s requires algorithm.on_error=fail_closed",
			decisionName,
			RaylineRemoteAlgorithmType,
		)
	}
	if err := validateRaylineRemoteAlgorithmConfig(
		algorithm.RaylineRemote,
		modelRefs,
	); err != nil {
		return fmt.Errorf(
			"decision '%s', algorithm.rayline_remote: %w",
			decisionName,
			err,
		)
	}
	return nil
}
