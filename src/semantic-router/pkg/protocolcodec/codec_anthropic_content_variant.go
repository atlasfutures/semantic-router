package protocolcodec

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Anthropic content blocks are a discriminated union whose members overlap.
// This file holds the rules a block must satisfy before it is decoded: which
// type name it may use, which members that type may carry, and which members
// the neutral contract refuses. Keeping them apart from the decode and encode
// paths keeps one reviewable statement of the union contract.

// anthropicBlockLocation names where a refused block sat. A refused body is
// never stored, so a refusal that says only which feature is unsupported
// leaves an operator with no way to find the block in a conversation of
// hundreds.
func anthropicBlockLocation(typeName string, index int) string {
	return fmt.Sprintf("content block %d of type %q", index, typeName)
}

func validateAnthropicContentVariant(body json.RawMessage, typeName string, providerOutput bool) error {
	allowedByType := map[string][]string{
		"text":        {"cache_control", "citations", "text", "type"},
		"thinking":    {"signature", "thinking", "type"},
		"image":       {"cache_control", "source", "transformations", "type"},
		"document":    {"cache_control", "citations", "context", "source", "title", "type"},
		"tool_use":    {"cache_control", "caller", "id", "input", "name", "toolset_name", "type"},
		"tool_result": {"cache_control", "content", "is_error", "tool_use_id", "toolset_name", "type"},
	}
	if providerOutput {
		allowedByType = map[string][]string{
			"text":     {"citations", "text", "type"},
			"thinking": {"signature", "thinking", "type"},
			"tool_use": {"caller", "id", "input", "name", "toolset_name", "type"},
		}
	}
	allowed, recognized := allowedByType[typeName]
	if !recognized {
		return nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return err
	}
	if err := requireAnthropicContentFields(object, typeName, providerOutput); err != nil {
		return err
	}
	return rejectAnthropicContentVariantFields(object, allowed, providerOutput)
}

func requireAnthropicContentFields(object map[string]json.RawMessage, typeName string, providerOutput bool) error {
	requiredByType := map[string][]string{
		"text":        {"text"},
		"thinking":    {"thinking"},
		"image":       {"source"},
		"document":    {"source"},
		"tool_use":    {"id", "input", "name"},
		"tool_result": {"content", "tool_use_id"},
	}
	for _, name := range requiredByType[typeName] {
		if _, present := object[name]; present {
			continue
		}
		category := llmprotocol.ErrorInvalidRequest
		code := "invalid_content_variant"
		message := "Anthropic content is missing the required field: " + name
		if providerOutput {
			category = llmprotocol.ErrorUpstreamUnavailable
			code = "invalid_response_content"
			message = "Anthropic provider output is missing the required field: " + name
		}
		return llmprotocol.NewError(category, code, message, nil)
	}
	return nil
}

func rejectAnthropicContentVariantFields(
	object map[string]json.RawMessage,
	allowed []string,
	providerOutput bool,
) error {
	known := []string{
		"cache_control", "caller", "citations", "content", "context", "data", "file_id", "id", "input",
		"is_error", "name", "signature", "source", "text", "thinking", "title", "tool_use_id",
		"toolset_name", "transformations", "type",
	}
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, name := range allowed {
		allowedSet[name] = struct{}{}
	}
	for _, name := range known {
		if _, present := object[name]; !present {
			continue
		}
		if _, valid := allowedSet[name]; valid {
			continue
		}
		category := llmprotocol.ErrorInvalidRequest
		code := "invalid_content_variant"
		message := "Anthropic content includes a field from a different union variant"
		if providerOutput {
			category = llmprotocol.ErrorUpstreamUnavailable
			code = "invalid_response_content"
			message = "Anthropic provider output mixes content union variants"
		}
		return llmprotocol.NewError(category, code, message+": "+name, nil)
	}
	return nil
}

func anthropicContentDiscriminator(body json.RawMessage) (string, error) {
	var discriminator struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(body, &discriminator); err != nil || discriminator.Type == "" {
		return "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "content_type_required", "Anthropic content type is required", err)
	}
	return discriminator.Type, nil
}

func anthropicRequestContentType(body json.RawMessage) (string, error) {
	typeName, err := anthropicContentDiscriminator(body)
	if err != nil {
		return "", err
	}
	return typeName, nil
}

func anthropicResponseContentType(body json.RawMessage) (string, error) {
	typeName, err := anthropicContentDiscriminator(body)
	if err != nil {
		return "", err
	}
	switch typeName {
	case "text", "thinking", "tool_use":
		return typeName, nil
	case "redacted_thinking":
		return "", llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "redacted_reasoning", "redacted reasoning cannot be translated", nil)
	default:
		return "", llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Anthropic response content type is unsupported", nil)
	}
}

// validateAnthropicContentExtensions refuses the block members the neutral
// contract does not carry. Every refusal here names the block it came from, so
// the ingress_request_refused line says which block and which field failed.
func validateAnthropicContentExtensions(block anthropicContentWire, location string, providerOutput bool) error {
	// A request block carries its citations unread; only provider output is
	// refused, because the Router would have to generate the response-side
	// spans it cannot derive.
	if providerOutput && len(block.Citations) > 0 {
		return llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_citations",
			"Anthropic citations are not supported by the neutral contract",
			unsupportedFieldCause(location, "content.citations"),
		)
	}
	return rejectUnsupportedRequestFieldsAt(location, map[string]json.RawMessage{
		"content.context":         block.Context,
		"content.title":           block.Title,
		"content.toolset_name":    block.ToolsetName,
		"content.transformations": block.Transformations,
	})
}
