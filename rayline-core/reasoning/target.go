package reasoning

import (
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
)

// EncryptedIssuer names the account and target able to read a Responses blob.
// A varying serving provider (including OpenRouter) or unresolved credential
// has unknown provenance. Secrets are represented only by a truncated digest.
func EncryptedIssuer(worker, backend, model, credential string, credentialKnown, varyingProvider bool) string {
	if varyingProvider || !credentialKnown {
		return ReasoningIssuerUnknown
	}
	return shortDigest(strings.Join([]string{worker, backend, model, credential}, "\x00"))
}

// OpaqueIssuer names the serving model/provider for Anthropic-format opaque
// reasoning. For OpenRouter, pinnedProvider must name its sole provider with
// fallbacks disabled; an empty pin means the serving provider is unknown.
func OpaqueIssuer(worker, backend, model string, openRouter bool, pinnedProvider string) string {
	if openRouter {
		if pinnedProvider == "" {
			return ReasoningIssuerUnknown
		}
		return strings.Join([]string{worker, "openrouter", pinnedProvider, model}, "\x00")
	}
	return strings.Join([]string{worker, backend, model}, "\x00")
}

// ApplyEncryptedReasoning marks whether held blobs can reach this target and
// returns the issuer set to stage. Persist it only with successful execution.
func ApplyEncryptedReasoning(request *llmprotocol.Request, previous []string, issuer string, target llmprotocol.WireFormat) []string {
	held := protocolcodec.HoldsEncryptedReasoning(*request)
	issues := target == llmprotocol.OpenAIResponsesV1
	forwarded := held && issues && ReasoningIssuersAre(previous, issuer)
	request.ForwardsEncryptedReasoning = forwarded
	return NextReasoningIssuers(previous, held, forwarded, issues, issuer)
}

// DropOpaqueReasoning removes only blocks recorded as unreadable by issuer.
// Unknown blocks retain the codec's existing handling of client history.
func DropOpaqueReasoning(request *llmprotocol.Request, provenance []ReasoningProvenance, issuer string) int {
	return protocolcodec.DropOpaqueReasoning(request, func(data string) bool {
		return ReasoningIssuedElsewhere(provenance, data, issuer)
	})
}
