package execution

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

const (
	cloudExactDispatch = "exact_dispatch"
	cloudModelPin      = "model_pin"
)

func cloudTransport(deployment Deployment) string {
	if deployment.CloudTransport == "" {
		return cloudExactDispatch
	}
	return deployment.CloudTransport
}

func validateCloudTransport(deployment Deployment) error {
	if deployment.Kind != "cloud" {
		if deployment.CloudTransport != "" {
			return fmt.Errorf("cloud_transport requires a cloud deployment")
		}
		return nil
	}
	switch cloudTransport(deployment) {
	case cloudExactDispatch:
		if !validSHA256(deployment.DispatchProfileSHA256) || !validSHA256(deployment.IssuerSHA256) {
			return fmt.Errorf("cloud deployment %s requires an approved dispatch profile and issuer SHA256", deployment.ID)
		}
	case cloudModelPin:
		// The existing gateway owns credentials and provider routing. Its model
		// receipt does not attest a provider, account, action or cache replica.
		if deployment.DispatchProfileSHA256 != "" || deployment.IssuerSHA256 != "" || deployment.ProviderPin != nil {
			return fmt.Errorf("model_pin cannot claim an exact dispatch profile, issuer or provider pin")
		}
		if (deployment.Protocol != llmprotocol.OpenAIChatV1 && deployment.Protocol != llmprotocol.AnthropicMessagesV1) || deployment.RegistryProvider != "openrouter" {
			return fmt.Errorf("model_pin currently qualifies OpenRouter Chat Completions and Messages only")
		}
		if deployment.WireModel != deployment.RegistryModel || !strings.Contains(deployment.WireModel, "/") {
			return fmt.Errorf("model_pin requires the concrete registry model as wire_model")
		}
		endpoint, err := url.Parse(deployment.URL)
		path := "/v1/chat/completions"
		if deployment.Protocol == llmprotocol.AnthropicMessagesV1 {
			path = "/v1/messages"
		}
		if err != nil || !strings.HasSuffix(endpoint.Path, path) {
			return fmt.Errorf("model_pin requires the gateway %s endpoint", path)
		}
		if deployment.AuthEnv == "" {
			return fmt.Errorf("model_pin requires a gateway credential environment")
		}
	default:
		return fmt.Errorf("unknown cloud_transport %q", deployment.CloudTransport)
	}
	return nil
}

func validateModelPinControl(deployment Deployment, control thinkingcontrol.Control) error {
	if deployment.Kind != "cloud" || cloudTransport(deployment) != cloudModelPin {
		return nil
	}
	nativeAllowed := control.Native == "default"
	if deployment.Protocol == llmprotocol.AnthropicMessagesV1 {
		switch control.Native {
		case "low", "medium", "high", "max":
			nativeAllowed = true
		}
	}
	if !nativeAllowed || control.BudgetTokens != nil {
		return fmt.Errorf("model_pin does not preserve this native thinking configuration")
	}
	return nil
}

// The legacy Messages adapter folds in-array system messages and recursively
// coerces the exact string xhigh. Refuse those requests before generation:
// neither changing a control nor rewriting a cached transcript is acceptable.
func validateModelPinBody(body []byte, deployment Deployment) error {
	if cloudTransport(deployment) != cloudModelPin || deployment.Protocol != llmprotocol.AnthropicMessagesV1 {
		return nil
	}
	var wire map[string]any
	if err := json.Unmarshal(body, &wire); err != nil {
		return err
	}
	if messages, ok := wire["messages"].([]any); ok {
		for _, item := range messages {
			if message, ok := item.(map[string]any); ok && (message["role"] == "system" || message["role"] == "developer") {
				return fmt.Errorf("model_pin Messages cannot preserve in-array system or developer messages")
			}
		}
	}
	var hasXhigh func(any) bool
	hasXhigh = func(value any) bool {
		switch value := value.(type) {
		case string:
			return value == "xhigh"
		case []any:
			for _, item := range value {
				if hasXhigh(item) {
					return true
				}
			}
		case map[string]any:
			for _, item := range value {
				if hasXhigh(item) {
					return true
				}
			}
		}
		return false
	}
	if hasXhigh(wire) {
		return fmt.Errorf("model_pin Messages cannot preserve a literal xhigh value through the current gateway")
	}
	return nil
}

func modelPinEchoMatches(header http.Header, deployment Deployment) bool {
	return header.Get("X-Rayline-Selected-Model") == deployment.WireModel &&
		header.Get("X-Rayline-Decider") == "rules" && header.Get("X-Rayline-Decider-Reason") == "pin"
}

// The catalog may move a model between provider accounts without changing its
// public model name. Keep readable reasoning, but never replay signatures or
// opaque Chat reasoning_details under an unverified issuer identity.
func stripModelPinReasoning(request *llmprotocol.Request) int {
	dropped := 0
	messages := make([]llmprotocol.Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		contents := make([]llmprotocol.Content, 0, len(message.Content))
		for _, content := range message.Content {
			// A model pin cannot replay ciphertext, but readable thinking remains
			// useful on Chat even when a redacted block shared its original turn.
			if _, opaque := protocolcodec.OpaqueReasoningData(content); opaque {
				dropped++
				continue
			}
			if content.Kind == llmprotocol.ContentReasoning {
				if content.Signature != "" {
					content.Signature = ""
					dropped++
				}
				if extension := content.Extensions; extension != nil && extension.Format == llmprotocol.OpenAIChatV1 {
					if _, exists := extension.Fields["reasoning_details"]; exists {
						fields := make(map[string]json.RawMessage, len(extension.Fields))
						for key, value := range extension.Fields {
							if key != "reasoning_details" {
								fields[key] = value
							}
						}
						content.Extensions = nil
						if len(fields) > 0 {
							content.Extensions = &llmprotocol.UnmodeledFields{Format: extension.Format, Fields: fields}
						}
						dropped++
					}
				}
				if content.Text == "" && content.Extensions == nil {
					continue
				}
			}
			contents = append(contents, content)
		}
		if len(contents) != 0 || len(message.Content) == 0 {
			message.Content = contents
			messages = append(messages, message)
		}
	}
	request.Messages = messages
	return dropped
}
