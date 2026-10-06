package llmprotocol

import (
	"strings"
	"testing"
)

const openRouterCreditsMessage = "This request requires more credits, or fewer max_tokens. You requested up to 131072 tokens, " +
	"but can only afford 54095. To increase, visit https://openrouter.ai/settings/keys/0123abcd and create a key with a higher limit"

func TestPublicUpstreamErrorHidesProviderAccountErrors(t *testing.T) {
	tests := []struct {
		name   string
		err    ProtocolError
		status int
	}{
		{"openrouter 402", ProtocolError{Category: ErrorUpstreamUnavailable, Code: "402", Message: openRouterCreditsMessage}, 402},
		{"402 in a stream", ProtocolError{Category: ErrorUpstreamUnavailable, Code: "402", Message: openRouterCreditsMessage}, 0},
		{"401", ProtocolError{Category: ErrorUpstreamUnavailable, Code: "401", Message: "No auth credentials found"}, 401},
		{"authentication", ProtocolError{Category: ErrorAuthentication, Code: "authentication_error", Message: "invalid x-api-key"}, 0},
		{"billing", ProtocolError{Category: ErrorUpstreamUnavailable, Code: "billing_error", Message: "credit balance is too low"}, 400},
		{"quota", ProtocolError{Category: ErrorRateLimited, Code: "insufficient_quota", Message: "exceeded your current quota"}, 429},
		{"openrouter key limit 403", ProtocolError{
			Category: ErrorPermission, Code: "403",
			Message: "Key limit exceeded (daily limit). Manage it using https://openrouter.ai/settings/keys",
		}, 403},
		{"anthropic credit 400", ProtocolError{
			Category: ErrorInvalidRequest, Code: "invalid_request_error",
			Message: "Your credit balance is too low to access the Anthropic API.",
		}, 400},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			public := PublicUpstreamError(&test.err, test.status)
			if public.Category != ErrorUpstreamUnavailable || public.Code != "upstream_unavailable" ||
				public.Message != "model service unavailable" {
				t.Fatalf("public = %+v", public)
			}
		})
	}
}

func TestPublicUpstreamErrorKeepsActionableErrorsWithoutLinksOrKeys(t *testing.T) {
	upstream := &ProtocolError{
		Category: ErrorInvalidRequest, Code: "context_length_exceeded", Parameter: "messages",
		Message: "prompt is too long: 210000 tokens > 200000 maximum (see https://openrouter.ai/docs/limits; key sk-or-v1-0123456789abcdef)",
		Cause:   errString("internal"),
	}
	public := PublicUpstreamError(upstream, 400)
	if public.Category != ErrorInvalidRequest || public.Code != "context_length_exceeded" || public.Parameter != "messages" {
		t.Fatalf("semantics changed: %+v", public)
	}
	if !strings.HasPrefix(public.Message, "prompt is too long: 210000 tokens > 200000 maximum") {
		t.Fatalf("actionable text lost: %q", public.Message)
	}
	if strings.Contains(public.Message, "openrouter.ai") || strings.Contains(public.Message, "sk-or") || public.Cause != nil {
		t.Fatalf("provider detail kept: %+v", public)
	}
	if !strings.Contains(upstream.Message, "openrouter.ai") {
		t.Fatal("the provider's error was modified")
	}
	if PublicUpstreamError(nil, 500) != nil {
		t.Fatal("nil error must stay nil")
	}
}

type errString string

func (err errString) Error() string { return string(err) }
