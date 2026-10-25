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
	if deployment.Kind == "cloud" && cloudTransport(deployment) == cloudModelPin {
		if deployment.ProviderPin != nil {
			return fmt.Errorf("model_pin uses catalog provider routing, not an approved provider pin")
		}
		return nil
	}
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
// settings. Exact dispatch validates these bytes against its profile;
// model_pin leaves provider routing to the existing gateway's catalog.
// Private transcript identity stays in headers.
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
	if cloudTransport(deployment) == cloudModelPin {
		// Harness routing preferences cannot displace the deployment selected by
		// ARC. This transport explicitly delegates provider choice to the gateway.
		body, err = sjson.DeleteBytes(body, "provider")
		if err != nil {
			return nil, err
		}
		if err := validateModelPinBody(body, deployment); err != nil {
			return nil, err
		}
	}
	return body, nil
}
