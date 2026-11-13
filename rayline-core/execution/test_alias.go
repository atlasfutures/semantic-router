package execution

import (
	"fmt"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

func (server *Server) validateTestAlias() error {
	alias := server.config.TestAlias
	if alias == nil {
		return nil
	}
	if server.config.Decisions.apiMode() != DecisionsCurrent {
		return fmt.Errorf("test_alias is only supported by current Decisions with cache evidence retained locally")
	}
	targets := 0
	selected := false
	for _, actions := range server.actions {
		for _, candidate := range actions {
			selected = selected || candidate.action.Model == alias.SelectedModel
		}
		target, ok := actions[alias.ExecutionActionID]
		if !ok {
			continue
		}
		targets++
		if alias.SelectedModel == "" || alias.SelectedModel == target.action.Model {
			return fmt.Errorf("test_alias must name a different selected model and an admitted execution action")
		}
		if err := qualifyTestAliasTarget(target); err != nil {
			return err
		}
	}
	if targets == 0 {
		return fmt.Errorf("test_alias must name a different selected model and an admitted execution action")
	}
	if !selected {
		return fmt.Errorf("test_alias selected model has no pinned action binding")
	}
	return nil
}

func qualifyTestAliasTarget(target admittedAction) error {
	if target.deployment.Kind != "local" || strings.TrimPrefix(target.deployment.RegistryModel, "qwen/") != "qwen3.8-27b" || strings.TrimPrefix(target.action.Model, "qwen/") != "qwen3.8-27b" || target.control.Native != "default" || target.control.Instruction != nil || target.control.BudgetTokens != nil || target.deployment.PromptCounter == nil {
		return fmt.Errorf("test_alias target must be qualified local Qwen 27B with the native default control")
	}
	return nil
}

func (server *Server) executionFor(format llmprotocol.WireFormat, selection Decision) (admittedAction, Decision, string, error) {
	selected, ok := server.actions[format][selection.ActionID]
	if !ok || selected.action.Worker != selection.Worker || selected.action.Model != selection.Model || selected.action.ControlID != selection.ControlID || (server.config.PackageSHA256 != "" && selection.PackageSHA256 != server.config.PackageSHA256) {
		return admittedAction{}, Decision{}, "", fmt.Errorf("selected action does not match the pinned package and a %s deployment", format)
	}
	alias := server.config.TestAlias
	if alias == nil || alias.SelectedModel != selection.Model {
		if selected.deployment.ID == "" {
			return admittedAction{}, Decision{}, "", fmt.Errorf("selection-only action cannot execute without its explicit test alias")
		}
		return selected, selection, "", nil
	}
	target, ok := server.actions[format][alias.ExecutionActionID]
	if !ok {
		return admittedAction{}, Decision{}, "", fmt.Errorf("test_alias execution action %s has no %s deployment", alias.ExecutionActionID, format)
	}
	execution := selection
	execution.ActionID, execution.Worker, execution.Model, execution.ControlID = target.action.ID, target.action.Worker, target.action.Model, target.action.ControlID
	execution.ArmID = nil // A selected model's route arm cannot attest the aliased execution.
	return target, execution, "explicit_test_model_alias", nil
}
