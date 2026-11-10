package execution

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"unicode"

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
		// registry_model names the registry cell whose controls were measured;
		// wire_model is the gateway's catalog id for that same model, which
		// may name it without the registry's provider prefix (the gateway's
		// claude-sonnet-5-5 for anthropic/claude-sonnet-5.5, served by
		// Anthropic directly). The echo check below holds the gateway to
		// serving exactly wire_model as a rules pin.
		if !strings.Contains(deployment.RegistryModel, "/") || deployment.WireModel == "" ||
			strings.ContainsFunc(deployment.WireModel, unicode.IsSpace) {
			return fmt.Errorf("model_pin requires a concrete registry model and the gateway's model id as wire_model")
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

// No literal-xhigh scan lives here. placer.Render owns the root control fields
// and strips thinking, reasoning, reasoning_effort and output_config.effort
// before this runs, so a client value there never reaches dispatch; a value
// nested in message content is forwarded, as it already is on every path that
// does not use this transport. A scan of the rest of the body refused ordinary
// client data, such as a tool schema that enumerates effort levels. On this
// transport the caller supplies the credential and pays per token, so the
// thinking level they ask for is their own cost to carry and is not a mutation
// to refuse. Any normalization the serving gateway performs applies equally to
// the paths that reach it without this check.
//
// An in-band system row is not refused. Folding one is opt-in per deployment
// (`inband_system`), applies only to a local Chat deployment
// (validateInbandSystem) and is the single caller of FoldInbandSystem, so no
// fold can reach a model_pin Messages dispatch: the row is forwarded as the
// client sent it.
//
// A developer row is still refused, because Messages has no developer
// authority to carry it (codec_anthropic_content.go). That role is read from
// clientRoles, the roles of the *client's* body: the decoder canonicalizes an
// Anthropic developer row to system before dispatch
// (codec_anthropic_messages.go), so the encoded body alone cannot distinguish
// it from a system row the client actually sent, and allowing system rows
// would otherwise dispatch developer content with changed authority. An empty
// clientRoles (a caller that cannot supply them) refuses nothing extra.
func validateModelPinBody(clientRoles []string, deployment Deployment) error {
	if cloudTransport(deployment) != cloudModelPin || deployment.Protocol != llmprotocol.AnthropicMessagesV1 {
		return nil
	}
	for _, role := range clientRoles {
		// The codec canonicalizes a role by lowercasing and trimming it
		// (canonicalRole), so " Developer " decodes to developer just as
		// "developer" does. Match the same way or the check is bypassable.
		if strings.ToLower(strings.TrimSpace(role)) == "developer" {
			return fmt.Errorf("model_pin Messages cannot preserve in-array developer messages")
		}
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
	dropped := protocolcodec.DropOpaqueReasoning(request, func(string) bool { return true })
	messages := make([]llmprotocol.Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		contents := make([]llmprotocol.Content, 0, len(message.Content))
		for _, content := range message.Content {
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
