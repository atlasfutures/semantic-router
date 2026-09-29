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

package config

import (
	"fmt"
	"regexp"
)

// RaylineARCPolicyServiceConfig selects the policy-service mode: an external
// service runs the encoder, head and selection rule and returns the decision
// (website/docs/proposals/rayline-arc-policy-service.md). The mounted v3
// artifact and the in-process encoder are not used in this mode.
type RaylineARCPolicyServiceConfig struct {
	BaseURL             string `yaml:"base_url"`
	ModalKeyEnv         string `yaml:"modal_key_env,omitempty"`
	ModalSecretEnv      string `yaml:"modal_secret_env,omitempty"`
	TotalTimeoutSeconds int    `yaml:"total_timeout_seconds"`
	// PackageAlias and PackageSHA256 name the one package this decision
	// serves. Readiness refuses a service that has not loaded exactly it.
	PackageAlias  string `yaml:"package_alias"`
	PackageSHA256 string `yaml:"package_sha256"`
	// Bindings map each package action to a worker (a modelRef). Several
	// actions of one model differ only in thinking level, so they share the
	// worker and carry the level the lever applies.
	Bindings []RaylineARCPolicyBinding `yaml:"bindings"`
	// ModelSchedule names the package's model-switch schedule, which VSR
	// enforces: between boundaries the model is held and only its level
	// varies. Empty lets every turn choose any action.
	ModelSchedule string `yaml:"model_schedule,omitempty"`
}

// RaylineARCPolicyBinding binds one package action to a worker and level.
type RaylineARCPolicyBinding struct {
	ActionID string `yaml:"action_id"`
	Worker   string `yaml:"worker"`
	Level    string `yaml:"level,omitempty"`
}

const (
	maxRaylineARCPolicyTimeoutSeconds = 900
	// RaylineARCModelScheduleTaskTurnCompaction is pathfinder's
	// task_turn_compaction_v1 (training/sdk/routing_schedule.py).
	RaylineARCModelScheduleTaskTurnCompaction = "task_turn_compaction_v1"
)

var raylineARCSHA256Pattern = regexp.MustCompile(`^[a-f0-9]{64}$`)

func validateRaylineARCPolicyServiceConfig(cfg *RaylineARCPolicyServiceConfig) error {
	if err := validateRaylineARCBaseURL(cfg.BaseURL); err != nil {
		return err
	}
	if err := validateRaylineARCModalAuth(RaylineARCEncoderConfig{
		ModalKeyEnv:    cfg.ModalKeyEnv,
		ModalSecretEnv: cfg.ModalSecretEnv,
	}); err != nil {
		return err
	}
	if cfg.TotalTimeoutSeconds <= 0 || cfg.TotalTimeoutSeconds > maxRaylineARCPolicyTimeoutSeconds {
		return fmt.Errorf("total_timeout_seconds must be between 1 and %d", maxRaylineARCPolicyTimeoutSeconds)
	}
	if cfg.PackageAlias == "" || len(cfg.PackageAlias) > maxRaylineARCConfigStringLength {
		return fmt.Errorf("package_alias is required")
	}
	if !raylineARCSHA256Pattern.MatchString(cfg.PackageSHA256) {
		return fmt.Errorf("package_sha256 must be 64 lowercase hex characters")
	}
	if cfg.ModelSchedule != "" && cfg.ModelSchedule != RaylineARCModelScheduleTaskTurnCompaction {
		return fmt.Errorf("model_schedule must be empty or %s", RaylineARCModelScheduleTaskTurnCompaction)
	}
	if len(cfg.Bindings) == 0 {
		return fmt.Errorf("bindings are required")
	}
	seen := make(map[string]bool, len(cfg.Bindings))
	for _, binding := range cfg.Bindings {
		if !raylineARCSHA256Pattern.MatchString(binding.ActionID) {
			return fmt.Errorf("binding action_id must be 64 lowercase hex characters")
		}
		if seen[binding.ActionID] {
			return fmt.Errorf("binding action_id %s is bound twice", binding.ActionID)
		}
		seen[binding.ActionID] = true
		if binding.Worker == "" {
			return fmt.Errorf("binding for action %s names no worker", binding.ActionID)
		}
	}
	return nil
}

// validateRaylineARCPolicyBindings checks every binding names a modelRef and
// every modelRef serves at least one action, so no arm is unreachable.
func validateRaylineARCPolicyBindings(decision Decision) error {
	cfg := decision.Algorithm.RaylineARC.PolicyService
	refs := make(map[string]bool, len(decision.ModelRefs))
	for _, modelRef := range decision.ModelRefs {
		refs[modelRef.Model] = false
	}
	for _, binding := range cfg.Bindings {
		if _, ok := refs[binding.Worker]; !ok {
			return fmt.Errorf("policy_service binding names worker %q, which is not a modelRef", binding.Worker)
		}
		refs[binding.Worker] = true
	}
	for worker, bound := range refs {
		if !bound {
			return fmt.Errorf("modelRef %q serves no policy_service action", worker)
		}
	}
	return nil
}
