package llmprotocol

import (
	"regexp"
	"strings"
)

// providerAccountMessage is what a client is told when the provider refused
// the Router's own credentials or account. The client's request was not at
// fault and the provider's words describe an account the client cannot see.
const providerAccountMessage = "model service unavailable"

// providerSecretPattern matches what a provider message may carry about the
// Router's provider account: a link (OpenRouter's 402 links the key's settings
// page, which names the key hash) or a key itself.
var providerSecretPattern = regexp.MustCompile(`(?i)\bhttps?://\S+|\bsk-[A-Za-z0-9_-]{8,}`)

// PublicUpstreamError is the form of a provider's error a client may see. A
// provider error describes the provider call the Router made with its own
// key, so it can name that key's account, budget and settings page.
//
//   - A refusal of the Router's credentials or account (401, 402, an
//     authentication, billing, quota or key-limit error) becomes a generic upstream
//     unavailability: none of its text is the client's business.
//   - Any other error keeps its category, code and message, which a client
//     acts on (a context-length 400 tells it to compact), with links and
//     key-shaped tokens removed from the message.
//
// status is the provider's HTTP status, or 0 when the error arrived in a
// stream. The argument is not modified.
func PublicUpstreamError(protocolError *ProtocolError, status int) *ProtocolError {
	if protocolError == nil {
		return nil
	}
	if providerAccountError(protocolError, status) {
		return NewError(ErrorUpstreamUnavailable, "upstream_unavailable", providerAccountMessage, nil)
	}
	public := *protocolError
	public.Cause = nil
	public.Message = RedactProviderText(protocolError.Message)
	return &public
}

// RedactProviderText removes links and key-shaped tokens from provider text.
func RedactProviderText(text string) string {
	if text == "" {
		return text
	}
	return providerSecretPattern.ReplaceAllString(text, "[redacted]")
}

func providerAccountError(protocolError *ProtocolError, status int) bool {
	if status == 401 || status == 402 || protocolError.Category == ErrorAuthentication {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(protocolError.Code)) {
	case "401", "402", "insufficient_quota", "payment_required", "billing_error", "insufficient_credits":
		return true
	}
	// Some account refusals come with another status: OpenRouter states its
	// key limit as a 403, Anthropic a low credit balance as a 400.
	message := strings.ToLower(protocolError.Message)
	for _, account := range providerAccountPhrases {
		if strings.Contains(message, account) {
			return true
		}
	}
	return false
}

var providerAccountPhrases = []string{"key limit", "credit balance", "more credits", "current quota", "billing"}
