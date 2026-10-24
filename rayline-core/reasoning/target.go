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

// DropOpaqueReasoning removes private Messages groups without recorded matching
// issuance. An unrecorded earlier block invalidates later private groups too:
// their signatures may bind the prefix being removed. Other wire formats retain
// their existing disposition for unknown history.
func DropOpaqueReasoning(request *llmprotocol.Request, provenance []ReasoningProvenance, issuer string, target llmprotocol.WireFormat) int {
	unknownPrefix := false
	return protocolcodec.DropOpaqueReasoning(request, func(data string) bool {
		if target == llmprotocol.AnthropicMessagesV1 {
			matches := ReasoningIssuedBy(provenance, data, issuer)
			if !matches && !ReasoningIssuedElsewhere(provenance, data, issuer) {
				// The missing record may be an older same-worker block. Removing it
				// changes the prefix that later signatures could be bound to, so issuer
				// equality alone cannot authorize replay after this point.
				unknownPrefix = true
			}
			return unknownPrefix || !matches
		}
		return ReasoningIssuedElsewhere(provenance, data, issuer)
	})
}
