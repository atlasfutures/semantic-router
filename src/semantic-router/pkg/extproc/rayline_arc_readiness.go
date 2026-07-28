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
	"reflect"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func createRaylineARCSelector(
	cfg *config.RouterConfig,
) (selection.Selector, string) {
	decisions := configuredRaylineARCDecisions(cfg)
	if len(decisions) == 0 {
		return nil, ""
	}
	arcConfig := decisions[0].Algorithm.RaylineARC
	unavailable := func(class string) (selection.Selector, string) {
		return newRaylineARCSelector(
			nil,
			nil,
			arcConfig.ArtifactRevision,
		), class
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
	if runtime.ArtifactID() != arcConfig.ArtifactRevision {
		return unavailable("artifact_revision")
	}
	if !raylineARCEncoderContractMatches(runtime, arcConfig) {
		return unavailable("artifact_encoder_contract")
	}
	workerIDs := runtime.WorkerIDs()
	for index := range decisions {
		if !raylineARCCandidatesMatch(decisions[index].ModelRefs, workerIDs) {
			return unavailable("artifact_arm_mapping")
		}
	}
	encoder, err := raylinearc.NewEncoderClient(
		raylineARCEncoderClientConfig(arcConfig),
	)
	if err != nil {
		return unavailable("encoder_config")
	}
	if err := encoder.Probe(
		context.Background(),
		"semantic-router-startup-readiness",
	); err != nil {
		return unavailable("encoder_probe")
	}
	return newRaylineARCSelector(
		&runtimeARCScorer{runtime: runtime, policy: runtime.Policy()},
		encoder,
		arcConfig.ArtifactRevision,
	), ""
}

func raylineARCEncoderClientConfig(
	arcConfig *config.RaylineARCAlgorithmConfig,
) raylinearc.EncoderClientConfig {
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
		RequiredCapabilities: append(
			[]string(nil),
			arcConfig.Encoder.RequiredCapabilities...,
		),
		ConnectTimeout: time.Duration(
			arcConfig.Encoder.ConnectTimeoutSeconds,
		) * time.Second,
		TotalTimeout: time.Duration(
			arcConfig.Encoder.TotalTimeoutSeconds,
		) * time.Second,
		MaxRetries: arcConfig.Encoder.MaxRetries,
	}
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
