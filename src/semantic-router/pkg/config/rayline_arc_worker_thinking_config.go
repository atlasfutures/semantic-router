package config

import (
	"fmt"
	"strings"
)

// Worker thinking wires: how a base reasoning level travels on OpenRouter's
// reasoning object. Each is exactly one control, because OpenRouter refuses
// an effort and a token budget together.
const (
	// RaylineARCWorkerThinkingEffort sends reasoning.effort and no budget.
	RaylineARCWorkerThinkingEffort = "effort"
	// RaylineARCWorkerThinkingBudget sends reasoning.max_tokens and no
	// effort: a simulated level for a model with no native ladder, such as a
	// low rung made of a 4096-token cap.
	RaylineARCWorkerThinkingBudget = "budget"
	// RaylineARCWorkerThinkingProviderDefault sends no reasoning control at
	// all, so the provider's default applies, uncapped.
	RaylineARCWorkerThinkingProviderDefault = "provider_default"
)

// RaylineARCWorkerThinkingConfig is one worker's base reasoning level.
//
// Without it, a thinking worker's reasoning is whatever the router derives:
// a reasoning.max_tokens bound from the client's output allowance, with the
// configured effort removed, because OpenRouter refuses the two together.
// Every agent turn states an output allowance, so a named effort never
// reached the model. A worker listed here gets the registry's wire instead,
// and nothing derived.
type RaylineARCWorkerThinkingConfig struct {
	// Level is the registry's logical level name, logged as it is; the wire
	// below is what the router sends.
	Level     string `yaml:"level"`
	Wire      string `yaml:"wire"`
	Effort    string `yaml:"effort,omitempty"`
	MaxTokens int64  `yaml:"max_tokens,omitempty"`
}

func validateRaylineARCWorkerThinking(workers map[string]RaylineARCWorkerThinkingConfig) error {
	for worker, cfg := range workers {
		if err := cfg.validate(); err != nil {
			return fmt.Errorf("%q: %w", worker, err)
		}
	}
	return nil
}

func (cfg RaylineARCWorkerThinkingConfig) validate() error {
	if strings.TrimSpace(cfg.Level) == "" {
		return fmt.Errorf("level is required")
	}
	switch cfg.Wire {
	case RaylineARCWorkerThinkingEffort:
		if !plainReasoningEffortName(cfg.Effort) || cfg.MaxTokens != 0 {
			return fmt.Errorf("wire effort needs a plain effort name and no max_tokens")
		}
	case RaylineARCWorkerThinkingBudget:
		if cfg.MaxTokens <= 0 || cfg.Effort != "" {
			return fmt.Errorf("wire budget needs a positive max_tokens and no effort")
		}
	case RaylineARCWorkerThinkingProviderDefault:
		if cfg.Effort != "" || cfg.MaxTokens != 0 {
			return fmt.Errorf("wire provider_default carries no effort and no max_tokens")
		}
	default:
		return fmt.Errorf("wire %q must be effort, budget or provider_default", cfg.Wire)
	}
	return nil
}

func plainReasoningEffortName(effort string) bool {
	if effort == "" {
		return false
	}
	for _, character := range effort {
		if (character < 'a' || character > 'z') && character != '_' {
			return false
		}
	}
	return true
}

// validateRaylineARCWorkerThinkingRefs refuses a base for a worker the
// decision cannot select, or one that does not reason: a thinking-off arm's
// off-signal is the router's own and a base would contradict it.
func validateRaylineARCWorkerThinkingRefs(cfg *RaylineARCAlgorithmConfig, modelRefs []ModelRef) error {
	if cfg == nil || len(cfg.WorkerThinking) == 0 {
		return nil
	}
	reasons := make(map[string]bool, len(modelRefs))
	for _, modelRef := range modelRefs {
		reasons[modelRef.Model] = modelRef.UseReasoning != nil && *modelRef.UseReasoning
	}
	for worker := range cfg.WorkerThinking {
		thinking, selectable := reasons[worker]
		if !selectable {
			return fmt.Errorf("%q is not one of the decision's modelRefs", worker)
		}
		if !thinking {
			return fmt.Errorf("%q does not reason (use_reasoning is false)", worker)
		}
	}
	return nil
}
