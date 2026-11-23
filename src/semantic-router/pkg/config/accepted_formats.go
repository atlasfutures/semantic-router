package config

import (
	"fmt"
	"slices"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// validateAcceptedFormats checks providers.models[].accepted_formats: known
// formats, each once, and an api_format, when stated, that is the first one,
// since the first is what the model is sent in when the client's format is
// not accepted.
func validateAcceptedFormats(modelName, apiFormat string, formats []string) error {
	if len(formats) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(formats))
	for index, format := range formats {
		if catalogProtocolForAPIFormat(format) == "" {
			return fmt.Errorf(
				"providers.models[%s].accepted_formats[%d] %q is unsupported; use openai, responses, or anthropic",
				modelName, index, format,
			)
		}
		if _, duplicate := seen[format]; duplicate {
			return fmt.Errorf("providers.models[%s].accepted_formats lists %q twice", modelName, format)
		}
		seen[format] = struct{}{}
	}
	if apiFormat != "" && apiFormat != formats[0] {
		return fmt.Errorf(
			"providers.models[%s].api_format %q must be the first of accepted_formats (%q)",
			modelName, apiFormat, formats[0],
		)
	}
	return nil
}

// validateAcceptedFormatBindings refuses a format some backend of the model
// cannot take: the endpoint is chosen after the request has been encoded, so
// every backend must accept every format the request may be sent in. A
// backend with a catalog mapping must list the format on that mapping, since
// another format could name a different native model.
func validateAcceptedFormatBindings(
	modelName string,
	formats []string,
	providers []modelcatalog.EffectiveModelProvider,
) error {
	if len(formats) > 1 {
		for index, provider := range providers {
			// A Vertex AI or Amazon Bedrock base URL names one API: Vertex
			// AI's Messages publisher path or OpenAI-compatible
			// /endpoints/openapi, Bedrock's /anthropic or /openai/v1. The other
			// format's path would resolve under the wrong base.
			switch provider.Provider.Definition.ID {
			case "vertex-ai":
				return fmt.Errorf(
					"providers.models[%s].accepted_formats: backend_refs[%d] is Vertex AI, whose base URL serves one API; list one format",
					modelName, index,
				)
			case "bedrock":
				return fmt.Errorf(
					"providers.models[%s].accepted_formats: backend_refs[%d] is Amazon Bedrock, whose base URL serves one API; list one format",
					modelName, index,
				)
			}
		}
	}
	for _, format := range formats {
		protocol := catalogProtocolForAPIFormat(format)
		for index, provider := range providers {
			if !slices.Contains(provider.Provider.Definition.Protocols, protocol) ||
				!slices.Contains(provider.Provider.Definition.SupportedOperations, protocol+"#create") {
				return fmt.Errorf(
					"providers.models[%s].accepted_formats %q: backend_refs[%d] provider %q does not serve %s",
					modelName, format, index, provider.Provider.Definition.ID, protocol,
				)
			}
			if provider.CatalogBinding != nil && !slices.Contains(provider.CatalogBinding.Protocols, protocol) {
				return fmt.Errorf(
					"providers.models[%s].accepted_formats %q: provider %q's catalog mapping of %q does not list %s",
					modelName, format, provider.Provider.Definition.ID, provider.CatalogBinding.ID, protocol,
				)
			}
		}
	}
	return nil
}

// GetModelAcceptedFormats returns the wire formats the model's backends
// accept, in preference order: accepted_formats when configured, otherwise
// its single api_format.
func (c *RouterConfig) GetModelAcceptedFormats(modelName string) []string {
	if c != nil && c.ModelConfig != nil {
		if modelConfig, ok := c.ModelConfig[modelName]; ok && len(modelConfig.AcceptedFormats) > 0 {
			return modelConfig.AcceptedFormats
		}
		if _, baseConfig, ok := c.resolveLoRABaseModel(modelName); ok && len(baseConfig.AcceptedFormats) > 0 {
			return baseConfig.AcceptedFormats
		}
	}
	return []string{strings.ToLower(strings.TrimSpace(c.GetModelAPIFormat(modelName)))}
}

// ResolveModelTargetAPIFormat returns the format a request in clientFormat is
// sent to the model in: the client's own when the model accepts it, otherwise
// the model's first accepted format.
func (c *RouterConfig) ResolveModelTargetAPIFormat(modelName, clientFormat string) string {
	formats := c.GetModelAcceptedFormats(modelName)
	if clientFormat != "" && slices.Contains(formats, clientFormat) {
		return clientFormat
	}
	return formats[0]
}

// ModelAcceptsOnlyAPIFormat reports whether every request to the model is
// sent in format, whatever the client's format.
func (c *RouterConfig) ModelAcceptsOnlyAPIFormat(modelName, format string) bool {
	for _, accepted := range c.GetModelAcceptedFormats(modelName) {
		if accepted != format {
			return false
		}
	}
	return true
}

// ModelAcceptsAPIFormat reports whether some request to the model may be sent
// in format.
func (c *RouterConfig) ModelAcceptsAPIFormat(modelName, format string) bool {
	return slices.Contains(c.GetModelAcceptedFormats(modelName), format)
}

// validateBedrockAuth refuses an Amazon Bedrock backend whose auth_header or
// auth_prefix moves the key off Bedrock's own shape (Authorization: Bearer).
// Bedrock reads API keys, long- or short-term, only there, so another header
// would carry the key to the wrong place and leave a caller's Authorization
// header in force.
func validateBedrockAuth(modelName string, index int, provider modelcatalog.EffectiveModelProvider) error {
	if provider.Provider.Definition.ID != "bedrock" {
		return nil
	}
	want := provider.Provider.Definition.Auth
	header, prefix := provider.Provider.Instance.AuthHeader, provider.Provider.Instance.AuthPrefix
	if (header != "" && !strings.EqualFold(strings.TrimSpace(header), want.Header)) ||
		((provider.Provider.Instance.AuthPrefixSet || prefix != "") && prefix != want.Prefix) {
		return fmt.Errorf(
			"providers.models[%s].backend_refs[%d] is Amazon Bedrock, which reads its key only from %s: %s; remove auth_header and auth_prefix",
			modelName, index, want.Header, want.Prefix,
		)
	}
	return nil
}
