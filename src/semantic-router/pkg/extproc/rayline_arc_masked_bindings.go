package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// raylineARCMaskedBindingIDs logs each binding the config masked and
// returns their action ids. A masked binding is a steered control on a
// worker served by Vertex AI that the worker's registry cell does not admit
// (config.RaylineARCMaskedBindings): the policy is never offered it, and its
// model is served through its other actions. The log, one event per binding
// when the selector is built, is how an operator sees which actions a Vertex
// AI install leaves out.
func raylineARCMaskedBindingIDs(decision *config.Decision, masked []config.RaylineARCMaskedBinding) map[string]bool {
	if len(masked) == 0 {
		return nil
	}
	ids := make(map[string]bool, len(masked))
	for _, binding := range masked {
		ids[binding.ActionID] = true
		logging.ComponentEvent("extproc", "rayline_arc_policy_binding_masked", map[string]interface{}{
			"decision":  decision.Name,
			"action_id": binding.ActionID,
			"model":     binding.Model,
			"worker":    binding.Worker,
			"cell":      binding.Cell,
			"reason":    binding.Reason,
		})
	}
	return ids
}
