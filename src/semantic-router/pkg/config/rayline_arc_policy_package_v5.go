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
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkingcontrol"
)

// A v5 package (pathfinder docs/arc_serving_contract.md, "Policy package
// v5"; ADR 0109) names each action by a model and a format-agnostic thinking
// control. VSR reads the manifest, recomputes each control_id, requires the
// control in the registry it serves from (thinkingcontrol.Embedded), and
// requires each bound control to be admitted for its worker's (model,
// provider, format). A v5 binding is only {action_id, worker}.
//
// A v6 package (rayline.arc-policy-package.v6) takes this same path: only
// its encoding profile differs, which the policy service applies. Images
// reach only vision-capable workers by the selector's request-time gating
// (no_vision_arm), so readiness does not refuse a v6 binding to a text-only
// worker: such a worker still serves the package's text turns.

// raylineARCPolicyPackageV5 is a loaded v5 manifest.
type raylineARCPolicyPackageV5 struct {
	path     string
	manifest *raylinearc.PolicyPackageManifestV5
	actions  map[string]RaylineARCPolicyActionV5
	// registryDiffers records that the manifest names another registry
	// artifact than the one VSR serves from. It is informational:
	// controls are content-addressed, and every control was resolved.
	registryDiffers bool
}

// RaylineARCPolicyActionV5 is one v5 action as VSR dispatches it.
type RaylineARCPolicyActionV5 struct {
	ActionID string
	// Model is the trained model the package names; the registry's cells
	// and the action_id both key on it.
	Model   string
	Control thinkingcontrol.Control
}

var raylineARCPolicyPackageV5Lock sync.Mutex

// IsPackageV5 reports whether the policy service serves a v5 package or a
// later one: a v6 package (ADR 0122, an encoder that reads images) is a v5
// package with an image encoding profile, and loads, binds, readies and
// dispatches exactly as v5 does. Which schema it is, is the manifest's
// (loadPackageV5).
func (cfg *RaylineARCPolicyServiceConfig) IsPackageV5() bool {
	return cfg != nil && cfg.PackageManifest != ""
}

// PackageV5Action returns a v5 action's model and control.
func (cfg *RaylineARCPolicyServiceConfig) PackageV5Action(actionID string) (RaylineARCPolicyActionV5, bool) {
	pkg, err := cfg.loadPackageV5()
	if err != nil {
		return RaylineARCPolicyActionV5{}, false
	}
	action, ok := pkg.actions[actionID]
	return action, ok
}

// PackageFallbackActionID is the package's declared fallback action
// (decision.fallback_action_id), or "" for a package without a manifest.
func (cfg *RaylineARCPolicyServiceConfig) PackageFallbackActionID() string {
	pkg, err := cfg.loadPackageV5()
	if err != nil {
		return ""
	}
	return pkg.manifest.Decision.FallbackActionID
}

// PackageV5RegistryDiffers reports whether the manifest's
// thinking_controls_sha256 names another registry artifact than VSR's.
func (cfg *RaylineARCPolicyServiceConfig) PackageV5RegistryDiffers() bool {
	pkg, err := cfg.loadPackageV5()
	return err == nil && pkg.registryDiffers
}

// loadPackageV5 reads, verifies and decodes the manifest once per path.
func (cfg *RaylineARCPolicyServiceConfig) loadPackageV5() (*raylineARCPolicyPackageV5, error) {
	raylineARCPolicyPackageV5Lock.Lock()
	defer raylineARCPolicyPackageV5Lock.Unlock()
	if cfg.packageV5 != nil && cfg.packageV5.path == cfg.PackageManifest {
		return cfg.packageV5, nil
	}
	raw, err := os.ReadFile(cfg.PackageManifest)
	if err != nil {
		return nil, fmt.Errorf("package_manifest: %w", err)
	}
	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != cfg.PackageSHA256 {
		return nil, fmt.Errorf("package_manifest's sha256 is %s, not package_sha256", got)
	}
	if schema, schemaErr := raylinearc.PolicyPackageSchemaOf(raw); schemaErr != nil ||
		(schema != raylinearc.PolicyPackageSchemaV5 && schema != raylinearc.PolicyPackageSchemaV6) {
		return nil, fmt.Errorf("package_manifest must be a %s or %s package (a v4 package's bindings declare their dispatch)",
			raylinearc.PolicyPackageSchemaV5, raylinearc.PolicyPackageSchemaV6)
	}
	manifest, err := raylinearc.DecodePolicyPackageManifestV5OrLater(raw)
	if err != nil {
		return nil, fmt.Errorf("package_manifest: %w", err)
	}
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		return nil, fmt.Errorf("the thinking-control registry: %w", err)
	}
	pkg := &raylineARCPolicyPackageV5{
		path:            cfg.PackageManifest,
		manifest:        manifest,
		actions:         make(map[string]RaylineARCPolicyActionV5, len(manifest.Actions)),
		registryDiffers: manifest.ThinkingControlsSHA256 != registry.SHA256,
	}
	for _, action := range manifest.Actions {
		control, ok := registry.Control(action.ControlID)
		if !ok {
			return nil, fmt.Errorf("package action %s names control %s, which is not in thinking-control registry %s",
				action.ActionID, action.ControlID, registry.SHA256)
		}
		pkg.actions[action.ActionID] = RaylineARCPolicyActionV5{
			ActionID: action.ActionID, Model: action.Model, Control: control,
		}
	}
	cfg.packageV5 = pkg
	return pkg, nil
}

// validateRaylineARCPolicyPackageV5Bindings refuses a v5 binding that states
// anything but its action and worker, and a v4-only setting. It runs with the
// rest of the policy_service block.
func validateRaylineARCPolicyPackageV5Bindings(cfg *RaylineARCPolicyServiceConfig) error {
	if _, err := cfg.loadPackageV5(); err != nil {
		return err
	}
	if cfg.DispatchEffort != "" {
		return fmt.Errorf("dispatch_effort serves v4 packages; a v5 control names its own base")
	}
	// The package's schedule is part of what it was trained under, and VSR
	// enforces the configured one; they must be the same.
	pkg, _ := cfg.loadPackageV5()
	declared := ""
	if schedule := pkg.manifest.Decision.ModelSchedule; schedule != nil {
		declared = *schedule
	}
	if cfg.ModelSchedule != declared {
		return fmt.Errorf("model_schedule is %q, and the package declares %q", cfg.ModelSchedule, declared)
	}
	bound := make(map[string]bool, len(cfg.Bindings))
	for _, binding := range cfg.Bindings {
		if binding.Level != "" || binding.Model != "" || binding.Effort != nil || binding.ReasoningMaxTokens != nil {
			return fmt.Errorf("binding for action %s: a v5 binding is only action_id and worker; the package names the model and control",
				binding.ActionID)
		}
		bound[binding.Worker] = true
		action, ok := pkg.actions[binding.ActionID]
		if !ok {
			continue // validateRaylineARCPolicyPackageV5Dispatch names it
		}
		trained, declared := cfg.TrainedModels[binding.Worker]
		if !declared {
			return fmt.Errorf("binding for action %s: trained_models declares no trained model for worker %q, and the action's is %q",
				binding.ActionID, binding.Worker, action.Model)
		}
		if trained != action.Model {
			return fmt.Errorf("binding for action %s: worker %q serves trained model %q, and the action's is %q",
				binding.ActionID, binding.Worker, trained, action.Model)
		}
	}
	for worker := range cfg.TrainedModels {
		if !bound[worker] {
			return fmt.Errorf("trained_models names worker %q, which no binding uses", worker)
		}
	}
	return nil
}

// validateRaylineARCPolicyPackageV5Dispatch checks a v5 package against the
// decision: every package action is bound, every binding names a package
// action, and every bound control is admitted for its worker's (model,
// provider, format). The thinking lever and worker_thinking are v4
// mechanisms; the registry renders each control.
func validateRaylineARCPolicyPackageV5Dispatch(cfg *RouterConfig, decision Decision) error {
	arc := decision.Algorithm.RaylineARC
	policy := arc.PolicyService
	pkg, err := policy.loadPackageV5()
	if err != nil {
		return err
	}
	if lever := arc.ThinkingLever; lever != nil && lever.Enabled {
		return fmt.Errorf("thinking_lever does not apply to a v5 package: the registry renders each action's control")
	}
	if len(arc.WorkerThinking) > 0 {
		return fmt.Errorf("worker_thinking does not apply to a v5 package: each action's control owns its reasoning wire")
	}
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		return err
	}
	reasons := make(map[string]bool, len(decision.ModelRefs))
	for _, modelRef := range decision.ModelRefs {
		reasons[modelRef.Model] = modelRef.UseReasoning != nil && *modelRef.UseReasoning
	}
	bound := make(map[string]bool, len(policy.Bindings))
	// An episode keeps one placer per worker, wire format and control shape,
	// up to a bound; more than that would evict a placer whose instructions
	// still need replaying.
	shapes := map[string]bool{}
	for _, binding := range policy.Bindings {
		if action, ok := pkg.actions[binding.ActionID]; ok {
			control := action.Control
			budget := int64(0)
			if control.BudgetTokens != nil {
				budget = *control.BudgetTokens
			}
			for _, format := range cfg.GetModelAcceptedFormats(binding.Worker) {
				shapes[fmt.Sprintf("%s|%s|%s|%d|%t", binding.Worker, format, control.Native, budget, control.Instruction != nil)] = true
			}
		}
	}
	if len(shapes) > raylinearc.MaxControlPlacements {
		return fmt.Errorf("the bindings put %d control shapes on their workers' formats, and an episode keeps %d",
			len(shapes), raylinearc.MaxControlPlacements)
	}
	for _, binding := range policy.Bindings {
		action, ok := pkg.actions[binding.ActionID]
		if !ok {
			return fmt.Errorf("policy_service binding names action %s, which is not in the package", binding.ActionID)
		}
		bound[binding.ActionID] = true
		provider, err := RaylineARCRegistryProvider(cfg, binding.Worker)
		if err != nil {
			return fmt.Errorf("policy_service binding for action %s: %w", binding.ActionID, err)
		}
		served, err := RaylineARCRegistryModel(cfg, binding.Worker)
		if err != nil {
			return fmt.Errorf("policy_service binding for action %s: %w", binding.ActionID, err)
		}
		// The target format is chosen per request from the worker's accepted
		// formats, so the control must be admitted on the cell of each one.
		for _, apiFormat := range cfg.GetModelAcceptedFormats(binding.Worker) {
			format, err := RaylineARCRegistryFormat(apiFormat)
			if err != nil {
				return fmt.Errorf("policy_service binding for action %s on worker %q: %w", binding.ActionID, binding.Worker, err)
			}
			if _, err := registry.Admit(served, provider, format, action.Control, policy.AllowExperimentalControls); err != nil {
				return fmt.Errorf("policy_service binding for action %s on worker %q: %w", binding.ActionID, binding.Worker, err)
			}
		}
		off := action.Control.Native == raylineARCRegistryBaseOff
		if off == reasons[binding.Worker] {
			return fmt.Errorf("policy_service binding for action %s: worker %q has use_reasoning %v, and the control's base is %q",
				binding.ActionID, binding.Worker, reasons[binding.Worker], action.Control.Native)
		}
	}
	for _, action := range pkg.manifest.Actions {
		if !bound[action.ActionID] {
			return fmt.Errorf("package action %s is not bound to a worker", action.ActionID)
		}
	}
	if pkg.manifest.SchemaVersion == raylinearc.PolicyPackageSchemaV6 {
		return validateRaylineARCPolicyPackageV6Vision(cfg, policy)
	}
	return nil
}

// validateRaylineARCPolicyPackageV6Vision requires every arm a v6 package's
// bindings dispatch to state vision on its model card, true or false. A v6
// encoder reads images, and an image turn leaves the offer only on an arm
// whose card says vision: false, since an unmarked card counts as
// vision-capable (ModelParams.SupportsVision, #215). Under a v6 package that
// default would send images to a text-only arm, so it is refused: the error
// names every unmarked arm. v4 and v5 packages keep the default.
func validateRaylineARCPolicyPackageV6Vision(cfg *RouterConfig, policy *RaylineARCPolicyServiceConfig) error {
	var unmarked []string
	seen := map[string]bool{}
	for _, binding := range policy.Bindings {
		worker := strings.TrimSpace(binding.Worker)
		if seen[worker] {
			continue
		}
		seen[worker] = true
		if params, ok := cfg.ModelConfig[worker]; !ok || params.Vision == nil {
			unmarked = append(unmarked, worker)
		}
	}
	if len(unmarked) == 0 {
		return nil
	}
	sort.Strings(unmarked)
	return fmt.Errorf("a v6 package needs every bound arm's model card to state vision (true or false); unmarked: %s",
		strings.Join(unmarked, ", "))
}

// raylineARCRegistryBaseOff is the registry's thinking-off base.
const raylineARCRegistryBaseOff = "off"

// RaylineARCRegistryFormat is the registry's name for an api_format.
func RaylineARCRegistryFormat(apiFormat string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(apiFormat)) {
	case "", APIFormatOpenAI, "openai.chat", string(llmprotocol.OpenAIChatV1):
		return thinkingcontrol.FormatChat, nil
	case APIFormatAnthropic, "anthropic.messages", string(llmprotocol.AnthropicMessagesV1):
		return thinkingcontrol.FormatMessages, nil
	case APIFormatResponses, "openai.responses", string(llmprotocol.OpenAIResponsesV1):
		return thinkingcontrol.FormatResponses, nil
	default:
		return "", fmt.Errorf("api_format %q is not a format the thinking-control registry names", apiFormat)
	}
}

// RaylineARCRegistryProvider is the registry provider a worker dispatches
// to: the one every one of its endpoints reaches (openrouter, anthropic or
// openai). With the request's target format it names the worker's admission
// cell.
// RaylineARCRegistryModel is the model a worker serves, as the registry's
// admission cells name it: the provider model id its dispatch sends. An
// action's model is the trained name its action_id digests, decoupled from
// providers, so it is logged but never admitted on.
func RaylineARCRegistryModel(cfg *RouterConfig, worker string) (string, error) {
	endpoints := cfg.GetEndpointsForModel(worker)
	if len(endpoints) == 0 {
		return "", fmt.Errorf("worker %q has no endpoint", worker)
	}
	served := ""
	for _, endpoint := range endpoints {
		model := cfg.ResolveExternalModelID(worker, endpoint.Name)
		if served != "" && served != model {
			return "", fmt.Errorf("worker %q serves both %s and %s; its admission cell is ambiguous", worker, served, model)
		}
		served = model
	}
	return served, nil
}

func RaylineARCRegistryProvider(cfg *RouterConfig, worker string) (provider string, err error) {
	endpoints := cfg.GetEndpointsForModel(worker)
	if len(endpoints) == 0 {
		return "", fmt.Errorf("worker %q has no endpoint", worker)
	}
	for _, endpoint := range endpoints {
		profile, err := cfg.GetProviderProfileForEndpoint(endpoint.Name)
		if err != nil {
			return "", fmt.Errorf("worker %q endpoint %q: %w", worker, endpoint.Name, err)
		}
		name := ""
		switch {
		case raylineARCOpenRouterProfile(profile):
			name = "openrouter"
		case strings.EqualFold(strings.TrimSpace(profile.Type), "anthropic"):
			name = "anthropic"
		case strings.EqualFold(strings.TrimSpace(profile.Type), "openai"):
			name = "openai"
		default:
			return "", fmt.Errorf("worker %q endpoint %q is provider type %q, which the thinking-control registry does not name",
				worker, endpoint.Name, profile.Type)
		}
		if provider != "" && provider != name {
			return "", fmt.Errorf("worker %q reaches both %s and %s; its admission cell is ambiguous", worker, provider, name)
		}
		provider = name
	}
	return provider, nil
}
