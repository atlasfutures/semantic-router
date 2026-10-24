package execution

import (
	"fmt"
	"strings"
)

func (server *Server) validateTestAlias() error {
	alias := server.config.TestAlias
	if alias == nil {
		return nil
	}
	if server.config.Decisions.apiMode() != DecisionsCurrent {
		return fmt.Errorf("test_alias is only supported by current Decisions with cache evidence retained locally")
	}
	target, ok := server.actions[alias.ExecutionActionID]
	if !ok || alias.SelectedModel == "" || alias.SelectedModel == target.action.Model {
		return fmt.Errorf("test_alias must name a different selected model and an admitted execution action")
	}
	if target.deployment.Kind != "local" || strings.TrimPrefix(target.deployment.RegistryModel, "qwen/") != "qwen3.8-27b" || strings.TrimPrefix(target.action.Model, "qwen/") != "qwen3.8-27b" || target.control.Native != "default" || target.control.Instruction != nil || target.control.BudgetTokens != nil || target.deployment.PromptCounter == nil {
		return fmt.Errorf("test_alias target must be qualified local Qwen 27B with the native default control")
	}
	for _, candidate := range server.actions {
		if candidate.action.Model == alias.SelectedModel {
			return nil
		}
	}
	return fmt.Errorf("test_alias selected model has no pinned action binding")
}

func (server *Server) executionFor(selection Decision) (admittedAction, Decision, string, error) {
	selected, ok := server.actions[selection.ActionID]
	if !ok || selected.action.Worker != selection.Worker || selected.action.Model != selection.Model || selected.action.ControlID != selection.ControlID || selection.PackageSHA256 != server.config.PackageSHA256 {
		return admittedAction{}, Decision{}, "", fmt.Errorf("selected action does not match the pinned package and deployment")
	}
	alias := server.config.TestAlias
	if alias == nil || alias.SelectedModel != selection.Model {
		if selected.deployment.ID == "" {
			return admittedAction{}, Decision{}, "", fmt.Errorf("selection-only action cannot execute without its explicit test alias")
		}
		return selected, selection, "", nil
	}
	target := server.actions[alias.ExecutionActionID]
	execution := selection
	execution.ActionID, execution.Worker, execution.Model, execution.ControlID = target.action.ID, target.action.Worker, target.action.Model, target.action.ControlID
	execution.ArmID = nil // A selected model's route arm cannot attest the aliased execution.
	return target, execution, "explicit_test_model_alias", nil
}
