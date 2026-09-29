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
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf16"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkinglever"
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
	// ConnectTimeoutSeconds bounds the dial and TLS handshake to the
	// service; zero selects 5 s, the encoder's.
	ConnectTimeoutSeconds int `yaml:"connect_timeout_seconds,omitempty"`
	// MaxInflightCalls caps concurrent decide calls per router process and
	// sheds the surplus with a retryable 429, as max_inflight_encoder_calls
	// does in the artifact mode. Zero disables the cap.
	MaxInflightCalls int `yaml:"max_inflight_calls,omitempty"`
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
//
// A binding that names Model also declares what the action dispatches, as
// the package's catalog states it: the model, the native effort, the
// reasoning budget, and (through Level) the lever bytes of its steering
// suffix. The loader recomputes the package's action_id from exactly these,
// so a binding that would send anything else refuses to load, and the router
// puts them on the provider request.
type RaylineARCPolicyBinding struct {
	ActionID string `yaml:"action_id"`
	Worker   string `yaml:"worker"`
	Level    string `yaml:"level,omitempty"`
	// Model is the action's model as the package names it: the trained
	// name, which the action_id digests. The worker is the remappable part,
	// so any worker may serve it; selection logs both names.
	Model string `yaml:"model,omitempty"`
	// Effort is the native reasoning effort; absent is the package's null,
	// which sends no effort. "none" is a thinking-off action.
	Effort *string `yaml:"effort,omitempty"`
	// ReasoningMaxTokens is the native reasoning budget; absent is null.
	ReasoningMaxTokens *int64 `yaml:"reasoning_max_tokens,omitempty"`
}

// DeclaresDispatch reports whether the binding states what its action sends.
func (binding RaylineARCPolicyBinding) DeclaresDispatch() bool {
	return binding.Model != ""
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
	if cfg.ConnectTimeoutSeconds < 0 || cfg.ConnectTimeoutSeconds > cfg.TotalTimeoutSeconds {
		return fmt.Errorf("connect_timeout_seconds must be between 0 and total_timeout_seconds")
	}
	if cfg.MaxInflightCalls < 0 || cfg.MaxInflightCalls > maxRaylineARCInflightEncoderCalls {
		return fmt.Errorf("max_inflight_calls must be between 0 and %d", maxRaylineARCInflightEncoderCalls)
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
		if binding.Effort != nil && !plainReasoningEffortName(*binding.Effort) {
			return fmt.Errorf("binding for action %s: effort must be a plain effort name", binding.ActionID)
		}
		if binding.ReasoningMaxTokens != nil && *binding.ReasoningMaxTokens <= 0 {
			return fmt.Errorf("binding for action %s: reasoning_max_tokens must be positive", binding.ActionID)
		}
		if !binding.DeclaresDispatch() && (binding.Effort != nil || binding.ReasoningMaxTokens != nil) {
			return fmt.Errorf("binding for action %s: effort and reasoning_max_tokens need model", binding.ActionID)
		}
		if binding.DeclaresDispatch() != cfg.Bindings[0].DeclaresDispatch() {
			return fmt.Errorf("either every binding declares model, effort and reasoning_max_tokens, or none does")
		}
	}
	return nil
}

// RaylineARCPolicyActionID is pathfinder's action identity
// (rayline_router.serving.arc_policy_contract.action_id): the sha256 of the
// canonical JSON of the dispatch fields, with the steering suffix as the
// sha256 of its raw UTF-8. Canonical JSON is Python's json.dumps with
// sort_keys, ensure_ascii and compact separators.
func RaylineARCPolicyActionID(model string, effort *string, reasoningMaxTokens *int64, steeringSuffix string) string {
	suffixDigest := sha256.Sum256([]byte(steeringSuffix))
	var canonical strings.Builder
	canonical.WriteString(`{"effort":`)
	if effort == nil {
		canonical.WriteString("null")
	} else {
		writePythonASCIIJSONString(&canonical, *effort)
	}
	canonical.WriteString(`,"model":`)
	writePythonASCIIJSONString(&canonical, model)
	canonical.WriteString(`,"reasoning_max_tokens":`)
	if reasoningMaxTokens == nil {
		canonical.WriteString("null")
	} else {
		canonical.WriteString(strconv.FormatInt(*reasoningMaxTokens, 10))
	}
	canonical.WriteString(`,"steering_suffix_sha256":`)
	writePythonASCIIJSONString(&canonical, hex.EncodeToString(suffixDigest[:]))
	canonical.WriteString("}")
	digest := sha256.Sum256([]byte(canonical.String()))
	return hex.EncodeToString(digest[:])
}

// writePythonASCIIJSONString encodes value as Python's json.dumps does with
// ensure_ascii: the two-character escapes, \u00XX for other control
// characters, and \uXXXX (surrogate pairs above the BMP) for non-ASCII.
func writePythonASCIIJSONString(builder *strings.Builder, value string) {
	builder.WriteByte('"')
	for _, character := range value {
		switch character {
		case '"':
			builder.WriteString(`\"`)
		case '\\':
			builder.WriteString(`\\`)
		case '\n':
			builder.WriteString(`\n`)
		case '\r':
			builder.WriteString(`\r`)
		case '\t':
			builder.WriteString(`\t`)
		case '\b':
			builder.WriteString(`\b`)
		case '\f':
			builder.WriteString(`\f`)
		default:
			switch {
			case character < 0x20 || (character >= 0x7f && character <= 0xffff):
				fmt.Fprintf(builder, `\u%04x`, character)
			case character > 0xffff:
				high, low := utf16.EncodeRune(character)
				fmt.Fprintf(builder, `\u%04x\u%04x`, high, low)
			default:
				builder.WriteRune(character)
			}
		}
	}
	builder.WriteByte('"')
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

// PolicyBinding returns the binding for a package action.
func (cfg *RaylineARCPolicyServiceConfig) PolicyBinding(actionID string) (RaylineARCPolicyBinding, bool) {
	if cfg == nil {
		return RaylineARCPolicyBinding{}, false
	}
	for _, binding := range cfg.Bindings {
		if binding.ActionID == actionID {
			return binding, true
		}
	}
	return RaylineARCPolicyBinding{}, false
}

// policyBindingSuffix is the steering suffix a binding's level puts on the
// wire: the lever level's bytes on a worker the lever binds, and nothing on
// the neutral level. A level the worker's lever cannot express is refused.
func policyBindingSuffix(arc *RaylineARCAlgorithmConfig, binding RaylineARCPolicyBinding) (string, error) {
	lever := arc.ThinkingLever
	var leverBinding *RaylineARCThinkingBindingConfig
	if lever != nil && lever.Enabled && lever.Source == RaylineARCThinkingSourcePolicy {
		if bound, ok := lever.Workers[binding.Worker]; ok {
			leverBinding = &bound
		}
	}
	if leverBinding == nil {
		if binding.Level == "" || binding.Level == raylineARCPolicyNeutralLevel {
			return "", nil
		}
		return "", fmt.Errorf("level %q needs a thinking_lever binding with source policy for worker %q", binding.Level, binding.Worker)
	}
	levelName := binding.Level
	if levelName == "" {
		levelName = leverBinding.NeutralLevel
	}
	for _, level := range leverBinding.Levels {
		if level.Level != levelName {
			continue
		}
		if leverBinding.Lever != string(thinkinglever.LeverSteeringSuffix) {
			return "", fmt.Errorf("worker %q expresses levels with %s; a package action steers only with a suffix", binding.Worker, leverBinding.Lever)
		}
		return level.Suffix, nil
	}
	return "", fmt.Errorf("level %q is not in worker %q's thinking_lever binding", levelName, binding.Worker)
}

// raylineARCPolicyNeutralLevel is the package's no-steer level label.
const raylineARCPolicyNeutralLevel = "none"

// validateRaylineARCPolicyDispatch refuses, at load, every binding the router
// could not dispatch as the package's action: a native effort or budget the
// worker cannot carry, a level its lever cannot express, or a thinking-off
// action with a steer. Then it recomputes the action_id from what would
// travel, so a binding that loads is exactly its action. Which worker serves
// the trained model is configuration, and is not checked here.
func validateRaylineARCPolicyDispatch(cfg *RouterConfig, decision Decision) error {
	arc := decision.Algorithm.RaylineARC
	policy := arc.PolicyService
	if len(policy.Bindings) == 0 || !policy.Bindings[0].DeclaresDispatch() {
		if lever := arc.ThinkingLever; lever != nil && lever.Enabled {
			return fmt.Errorf("thinking_lever with source policy needs bindings that declare model, effort and reasoning_max_tokens")
		}
		return nil
	}
	if len(arc.WorkerThinking) > 0 {
		return fmt.Errorf("worker_thinking cannot be set when policy_service bindings declare dispatch: each action owns its reasoning wire")
	}
	reasons := make(map[string]bool, len(decision.ModelRefs))
	for _, modelRef := range decision.ModelRefs {
		reasons[modelRef.Model] = modelRef.UseReasoning != nil && *modelRef.UseReasoning
	}
	for _, binding := range policy.Bindings {
		if err := validateRaylineARCPolicyBindingDispatch(cfg, arc, binding, reasons[binding.Worker]); err != nil {
			return fmt.Errorf("policy_service binding for action %s: %w", binding.ActionID, err)
		}
	}
	return nil
}

func validateRaylineARCPolicyBindingDispatch(
	cfg *RouterConfig,
	arc *RaylineARCAlgorithmConfig,
	binding RaylineARCPolicyBinding,
	useReasoning bool,
) error {
	suffix, err := policyBindingSuffix(arc, binding)
	if err != nil {
		return err
	}
	thinkingOff := binding.Effort != nil && *binding.Effort == raylineARCPolicyNeutralLevel
	if thinkingOff && suffix != "" {
		return fmt.Errorf("a thinking-off action (effort none) cannot be steered, and level %q carries a steering suffix", binding.Level)
	}
	if thinkingOff && binding.ReasoningMaxTokens != nil {
		return fmt.Errorf("a thinking-off action (effort none) cannot carry reasoning_max_tokens")
	}
	if thinkingOff && useReasoning {
		return fmt.Errorf("worker %q reasons (use_reasoning is true) but the action is thinking-off (effort none)", binding.Worker)
	}
	if !thinkingOff && (binding.Effort != nil || binding.ReasoningMaxTokens != nil) && !useReasoning {
		return fmt.Errorf("worker %q does not reason (use_reasoning is false), so it cannot carry the action's effort or budget", binding.Worker)
	}
	if binding.Effort != nil && binding.ReasoningMaxTokens != nil &&
		strings.EqualFold(strings.TrimSpace(cfg.GetModelAPIFormat(binding.Worker)), APIFormatOpenAI) {
		return fmt.Errorf("worker %q dispatches Chat, where OpenRouter refuses an effort and a reasoning budget together", binding.Worker)
	}
	if want := RaylineARCPolicyActionID(binding.Model, binding.Effort, binding.ReasoningMaxTokens, suffix); want != binding.ActionID {
		return fmt.Errorf("the declared dispatch does not reproduce the action: it digests to %s", want)
	}
	return nil
}
