package execution

import "fmt"

// Capabilities is the common, explicitly qualified envelope of every possible
// execution target. It advertises limits; the harness gateway never clamps a
// caller request to make it fit.
type Capabilities struct {
	ContextWindow   int64    `json:"context_window"`
	MaxOutputTokens int64    `json:"max_output_tokens"`
	InputModalities []string `json:"input_modalities"`
}

func (server *Server) executionCapabilities() (Capabilities, error) {
	var result Capabilities
	images := true
	seen := map[string]bool{}
	for _, action := range server.actions {
		if alias := server.config.TestAlias; alias != nil && action.action.Model == alias.SelectedModel {
			continue
		}
		deployment := action.deployment
		if seen[deployment.ID] {
			continue
		}
		seen[deployment.ID] = true
		if deployment.ContextWindow <= 0 || deployment.MaxOutputTokens <= 0 || deployment.MaxOutputTokens > deployment.ContextWindow {
			return result, fmt.Errorf("deployment %s requires explicit positive context_window and max_output_tokens within that context", deployment.ID)
		}
		modalities := map[string]bool{}
		for _, modality := range deployment.InputModalities {
			if (modality != "text" && modality != "image") || modalities[modality] {
				return result, fmt.Errorf("deployment %s has unsupported or duplicate input_modalities", deployment.ID)
			}
			modalities[modality] = true
		}
		if !modalities["text"] {
			return result, fmt.Errorf("deployment %s must explicitly qualify text input", deployment.ID)
		}
		images = images && modalities["image"]
		if result.ContextWindow == 0 || deployment.ContextWindow < result.ContextWindow {
			result.ContextWindow = deployment.ContextWindow
		}
		if result.MaxOutputTokens == 0 || deployment.MaxOutputTokens < result.MaxOutputTokens {
			result.MaxOutputTokens = deployment.MaxOutputTokens
		}
	}
	result.InputModalities = []string{"text"}
	if images {
		result.InputModalities = append(result.InputModalities, "image")
	}
	return result, nil
}
