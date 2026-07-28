/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"context"
	"errors"
	"math"
	"net/url"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func createRaylineARCSelector(
	cfg *config.RouterConfig,
) (
	selection.Selector,
	raylinearc.EpisodeStore,
	func() error,
	string,
) {
	decisions := configuredRaylineARCDecisions(cfg)
	if len(decisions) == 0 {
		return nil, nil, nil, ""
	}
	arcConfig := decisions[0].Algorithm.RaylineARC
	unavailable := func(class string) (
		selection.Selector,
		raylinearc.EpisodeStore,
		func() error,
		string,
	) {
		return newRaylineARCSelector(
			nil,
			nil,
			arcConfig.ArtifactRevision,
		), nil, nil, class
	}
	for index := 1; index < len(decisions); index++ {
		if !reflect.DeepEqual(
			arcConfig,
			decisions[index].Algorithm.RaylineARC,
		) {
			return unavailable("conflicting_config")
		}
	}
	runtime, err := raylinearc.LoadRuntime(arcConfig.ArtifactDir)
	if err != nil {
		return unavailable("artifact")
	}
	if failureClass := raylineARCLoadedContractFailure(
		cfg,
		runtime,
		arcConfig,
		decisions,
	); failureClass != "" {
		return unavailable(failureClass)
	}
	encoderConfig, err := raylineARCEncoderClientConfig(arcConfig)
	if err != nil {
		return unavailable("encoder_auth")
	}
	encoder, err := raylinearc.NewEncoderClient(encoderConfig)
	if err != nil {
		return unavailable("encoder_config")
	}
	if probeErr := encoder.Probe(
		context.Background(),
		"semantic-router-startup-readiness",
	); probeErr != nil {
		return unavailable("encoder_probe")
	}
	episodeStore, closeStore, err := createRaylineARCEpisodeStore(
		arcConfig.Episode,
	)
	if err != nil {
		return unavailable("episode_store")
	}
	return newRaylineARCSelector(
		&runtimeARCScorer{runtime: runtime, policy: runtime.Policy()},
		encoder,
		arcConfig.ArtifactRevision,
	), episodeStore, closeStore, ""
}

func raylineARCLoadedContractFailure(
	cfg *config.RouterConfig,
	runtime *raylinearc.Runtime,
	arcConfig *config.RaylineARCAlgorithmConfig,
	decisions []*config.Decision,
) string {
	if runtime.ArtifactID() != arcConfig.ArtifactRevision {
		return "artifact_revision"
	}
	if !raylineARCEncoderContractMatches(runtime, arcConfig) {
		return "artifact_encoder_contract"
	}
	workerIDs := runtime.WorkerIDs()
	for index := range decisions {
		if !raylineARCCandidatesMatch(decisions[index].ModelRefs, workerIDs) {
			return "artifact_arm_mapping"
		}
	}
	if !raylineARCDispatchContractMatches(cfg, runtime, decisions) {
		return "artifact_dispatch_contract"
	}
	return ""
}

func raylineARCDispatchContractMatches(
	cfg *config.RouterConfig,
	runtime *raylinearc.Runtime,
	decisions []*config.Decision,
) bool {
	if cfg == nil || runtime == nil {
		return false
	}
	workers := make([]raylinearc.WorkerManifest, len(runtime.WorkerIDs()))
	for index := range workers {
		worker, ok := runtime.Worker(index)
		if !ok {
			return false
		}
		workers[index] = worker
	}
	return raylineARCDispatchContractsMatch(cfg, workers, decisions)
}

func raylineARCDispatchContractsMatch(
	cfg *config.RouterConfig,
	workers []raylinearc.WorkerManifest,
	decisions []*config.Decision,
) bool {
	if cfg == nil || len(workers) == 0 || len(decisions) == 0 {
		return false
	}
	thinkingByWorker, ok := raylineARCThinkingModes(decisions)
	if !ok {
		return false
	}
	for index := range workers {
		if !raylineARCWorkerDispatchMatches(
			cfg,
			&workers[index],
			thinkingByWorker,
		) {
			return false
		}
	}
	return true
}

func raylineARCThinkingModes(
	decisions []*config.Decision,
) (map[string]bool, bool) {
	thinkingByWorker := make(map[string]bool)
	for _, decision := range decisions {
		for _, modelRef := range decision.ModelRefs {
			useReasoning := modelRef.UseReasoning != nil &&
				*modelRef.UseReasoning
			if previous, exists := thinkingByWorker[modelRef.Model]; exists &&
				previous != useReasoning {
				return nil, false
			}
			thinkingByWorker[modelRef.Model] = useReasoning
		}
	}
	return thinkingByWorker, true
}

func raylineARCWorkerDispatchMatches(
	cfg *config.RouterConfig,
	worker *raylinearc.WorkerManifest,
	thinkingByWorker map[string]bool,
) bool {
	thinking, exists := thinkingByWorker[worker.ID]
	if !exists ||
		thinking != (worker.ThinkingMode == "on") ||
		cfg.GetModelAPIFormat(worker.ID) != config.APIFormatOpenAI ||
		!raylineARCPriceIdentityMatches(cfg, worker) {
		return false
	}
	endpoints := cfg.GetEndpointsForModel(worker.ID)
	if len(endpoints) == 0 {
		return false
	}
	for endpointIndex := range endpoints {
		if !raylineARCEndpointIdentityMatches(
			cfg,
			worker,
			&endpoints[endpointIndex],
		) {
			return false
		}
	}
	return true
}

func raylineARCEndpointIdentityMatches(
	cfg *config.RouterConfig,
	worker *raylinearc.WorkerManifest,
	endpoint *config.VLLMEndpoint,
) bool {
	if endpoint == nil || endpoint.Type != "openai" ||
		cfg.ResolveExternalModelID(worker.ID, endpoint.Name) != worker.Model {
		return false
	}
	if !raylineARCEndpointCredentialMatches(worker, endpoint) {
		return false
	}
	profile, err := cfg.GetProviderProfileForEndpoint(endpoint.Name)
	if err != nil || profile == nil || profile.Type != "openai" {
		return false
	}
	parsed, err := url.Parse(profile.BaseURL)
	return err == nil &&
		strings.EqualFold(parsed.Hostname(), "openrouter.ai") &&
		parsed.Scheme == "https"
}

// raylineARCEndpointCredentialMatches enforces the artifact-owned credential
// identity: the endpoint key must come from exactly the environment variable
// the worker manifest declares, and that credential must be present. This
// keeps a wrong-tenant or absent key from passing readiness and letting a
// caller-supplied Authorization header reach the provider.
func raylineARCEndpointCredentialMatches(
	worker *raylinearc.WorkerManifest,
	endpoint *config.VLLMEndpoint,
) bool {
	return worker.APIKeyEnv != "" &&
		endpoint.APIKeyEnvName == worker.APIKeyEnv &&
		endpoint.APIKey != ""
}

func raylineARCPriceIdentityMatches(
	cfg *config.RouterConfig,
	worker *raylinearc.WorkerManifest,
) bool {
	pricing, ok := cfg.GetFullModelPricing(worker.ID)
	if !ok || pricing.Currency != "USD" || pricing.CacheWritePer1M == nil {
		return false
	}
	const tokenScale = 1_000_000
	return raylineARCPriceEqual(
		pricing.PromptPer1M,
		worker.EstimatedInputCostPerToken*tokenScale,
	) &&
		raylineARCPriceEqual(
			pricing.CachedInputPer1M,
			worker.EstimatedCacheReadCostPerToken*tokenScale,
		) &&
		raylineARCPriceEqual(
			*pricing.CacheWritePer1M,
			worker.EstimatedCacheWriteCostPerToken*tokenScale,
		) &&
		raylineARCPriceEqual(
			pricing.CompletionPer1M,
			worker.EstimatedOutputCostPerToken*tokenScale,
		)
}

func raylineARCPriceEqual(configured, artifact float64) bool {
	if math.IsNaN(configured) || math.IsInf(configured, 0) ||
		math.IsNaN(artifact) || math.IsInf(artifact, 0) {
		return false
	}
	tolerance := math.Max(1e-12, math.Abs(artifact)*1e-9)
	return math.Abs(configured-artifact) <= tolerance
}

func createRaylineARCEpisodeStore(
	episodeConfig config.RaylineARCEpisodeConfig,
) (raylinearc.EpisodeStore, func() error, error) {
	var store raylinearc.EpisodeStore
	var closeStore func() error
	switch episodeConfig.Backend {
	case config.RaylineARCBackendMemory:
		memoryStore, err := raylinearc.NewMemoryEpisodeStore(
			raylinearc.MemoryEpisodeStoreConfig{
				MaxEpisodes: episodeConfig.MaxInMemoryEpisodes,
				IdleTTL: time.Duration(
					episodeConfig.IdleTTLSeconds,
				) * time.Second,
			},
		)
		if err != nil {
			return nil, nil, err
		}
		store = memoryStore
	case config.RaylineARCBackendRedis:
		password, err := raylineARCRedisPassword(episodeConfig.Redis)
		if err != nil {
			return nil, nil, err
		}
		redisStore, err := raylinearc.NewRedisEpisodeStore(
			raylinearc.RedisEpisodeStoreConfig{
				Address:   episodeConfig.Redis.Address,
				DB:        episodeConfig.Redis.DB,
				Password:  password,
				UseTLS:    episodeConfig.Redis.UseTLS,
				PoolSize:  episodeConfig.Redis.PoolSize,
				KeyPrefix: episodeConfig.KeyPrefix,
				LeaseTTL: time.Duration(
					episodeConfig.LeaseTTLSeconds,
				) * time.Second,
				IdleTTL: time.Duration(
					episodeConfig.IdleTTLSeconds,
				) * time.Second,
			},
		)
		if err != nil {
			return nil, nil, err
		}
		store = redisStore
		closeStore = redisStore.Close
	default:
		return nil, nil, errors.New("unsupported ARC episode backend")
	}
	readinessContext, cancel := context.WithTimeout(
		context.Background(),
		raylineARCDurationMin(
			time.Duration(episodeConfig.AcquireTimeoutSeconds)*time.Second,
			5*time.Second,
		),
	)
	defer cancel()
	if err := store.(raylinearc.EpisodeStoreReadiness).Ready(
		readinessContext,
	); err != nil {
		if closeStore != nil {
			_ = closeStore()
		}
		return nil, nil, err
	}
	return store, closeStore, nil
}

func raylineARCRedisPassword(
	redisConfig config.RaylineARCRedisConfig,
) (string, error) {
	if redisConfig.PasswordEnv == "" {
		return "", nil
	}
	value, ok := os.LookupEnv(redisConfig.PasswordEnv)
	if !ok {
		return "", errors.New("ARC Redis password environment is unset")
	}
	return value, nil
}

func raylineARCEncoderClientConfig(
	arcConfig *config.RaylineARCAlgorithmConfig,
) (raylinearc.EncoderClientConfig, error) {
	modalKey, err := raylineARCOptionalSecret(
		arcConfig.Encoder.ModalKeyEnv,
	)
	if err != nil {
		return raylinearc.EncoderClientConfig{}, err
	}
	modalSecret, err := raylineARCOptionalSecret(
		arcConfig.Encoder.ModalSecretEnv,
	)
	if err != nil {
		return raylinearc.EncoderClientConfig{}, err
	}
	return raylinearc.EncoderClientConfig{
		BaseURL:               arcConfig.Encoder.BaseURL,
		Model:                 arcConfig.Encoder.Model,
		ModelRevision:         arcConfig.Encoder.ModelRevision,
		TokenizerRevision:     arcConfig.Encoder.ModelRevision,
		TokenizerSHA256:       raylinearc.EncoderTokenizerSHA256,
		EOSTokenID:            raylinearc.EncoderEOSTokenID,
		ExpectedBuildID:       arcConfig.Encoder.ExpectedBuildID,
		ExpectedPluginVersion: arcConfig.Encoder.ExpectedPluginVersion,
		SerializerVersion:     arcConfig.Encoder.SerializerVersion,
		ServingRung:           arcConfig.Encoder.ServingRung,
		RequiredCapabilities: append(
			[]string(nil),
			arcConfig.Encoder.RequiredCapabilities...,
		),
		ModalKey:    modalKey,
		ModalSecret: modalSecret,
		ConnectTimeout: time.Duration(
			arcConfig.Encoder.ConnectTimeoutSeconds,
		) * time.Second,
		TotalTimeout: time.Duration(
			arcConfig.Encoder.TotalTimeoutSeconds,
		) * time.Second,
		MaxRetries: arcConfig.Encoder.MaxRetries,
	}, nil
}

func raylineARCOptionalSecret(name string) (string, error) {
	if name == "" {
		return "", nil
	}
	value, ok := os.LookupEnv(name)
	if !ok || value == "" {
		return "", errors.New("ARC encoder credential environment is unset")
	}
	return value, nil
}

func configuredRaylineARCDecisions(
	cfg *config.RouterConfig,
) []*config.Decision {
	if cfg == nil {
		return nil
	}
	result := make([]*config.Decision, 0)
	for index := range cfg.Decisions {
		decision := &cfg.Decisions[index]
		if decision.Algorithm != nil &&
			decision.Algorithm.Type == config.RaylineARCAlgorithmType &&
			decision.Algorithm.RaylineARC != nil {
			result = append(result, decision)
		}
	}
	return result
}

func raylineARCEncoderContractMatches(
	runtime *raylinearc.Runtime,
	arcConfig *config.RaylineARCAlgorithmConfig,
) bool {
	contract := runtime.EncoderContract()
	return contract.Model == arcConfig.Encoder.Model &&
		contract.Revision == arcConfig.Encoder.ModelRevision &&
		contract.Dimension == 1024 &&
		contract.Serialization == arcConfig.Encoder.SerializerVersion
}

func raylineARCCandidatesMatch(
	modelRefs []config.ModelRef,
	workerIDs []string,
) bool {
	if len(modelRefs) != len(workerIDs) {
		return false
	}
	for index := range workerIDs {
		if modelRefs[index].Model != workerIDs[index] {
			return false
		}
	}
	return true
}
