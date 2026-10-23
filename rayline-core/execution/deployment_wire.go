package execution

import (
	"fmt"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/tidwall/sjson"
)

func hasExecutionHistory(request llmprotocol.Request) bool {
	for _, message := range request.Messages {
		if message.Role == llmprotocol.RoleAssistant || message.Role == llmprotocol.RoleTool {
			return true
		}
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentToolCall || content.Kind == llmprotocol.ContentToolResult {
				return true
			}
		}
	}
	return false
}

func validateProviderPin(deployment Deployment) error {
	openRouter := deployment.Kind == "cloud" && deployment.RegistryProvider == "openrouter"
	if !openRouter {
		if deployment.ProviderPin != nil {
			return fmt.Errorf("provider_pin is only valid for an OpenRouter cloud deployment")
		}
		return nil
	}
	pin := deployment.ProviderPin
	if pin == nil || len(pin.Only) != 1 || strings.TrimSpace(pin.Only[0]) == "" || pin.Only[0] != strings.TrimSpace(pin.Only[0]) || pin.AllowFallbacks {
		return fmt.Errorf("OpenRouter requires one approved provider and allow_fallbacks=false")
	}
	if pin.DataCollection != "" && pin.DataCollection != "deny" {
		return fmt.Errorf("provider data_collection must be deny when specified")
	}
	return nil
}

// applyDeploymentWireBody materializes the executor's approved transport
// settings. The cloud gateway validates these bytes against its profile and
// forwards them unchanged. Private transcript identity stays in headers.
func applyDeploymentWireBody(body []byte, deployment Deployment) ([]byte, error) {
	if deployment.Kind != "cloud" {
		return body, nil
	}
	var err error
	for _, key := range []string{"session_id", "prompt_cache_key"} {
		body, err = sjson.DeleteBytes(body, key)
		if err != nil {
			return nil, err
		}
	}
	if deployment.Protocol == llmprotocol.OpenAIResponsesV1 {
		body, err = sjson.SetBytes(body, "store", false)
		if err != nil {
			return nil, err
		}
	}
	if deployment.ProviderPin != nil {
		body, err = sjson.SetBytes(body, "provider", deployment.ProviderPin)
		if err != nil {
			return nil, err
		}
	}
	return body, nil
}
