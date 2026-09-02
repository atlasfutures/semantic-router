package protocolcodec

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// testdata/provider/openrouter/transport-error.json is the envelope OpenRouter
// returned on 2026-09-03 when no endpoint of the routed model accepted an
// image. It is the actionable answer to the request, and it names three things
// the OpenAI error contract does not expect: a routing_funnel object under
// metadata, a numeric code, and no type at all.
const openRouterTransportError = "transport-error.json"

const openRouterTransportErrorMessage = "No endpoints found that support image input"

// The client must be told what the upstream said. Replacing this envelope with
// a generic failure loses the only sentence that tells the caller to stop
// sending an image to this model.
func TestOpenRouterTransportErrorReachesTheClient(t *testing.T) {
	engine := NewBuiltinEngine()
	for name, target := range map[string]llmprotocol.WireFormat{
		"chat":      llmprotocol.OpenAIChatV1,
		"anthropic": llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(name, func(t *testing.T) {
			result, err := engine.TranslateTransportError(
				llmprotocol.OpenAIChatV1, target,
				loadProviderFixture(t, openRouterTransportError), nil,
			)
			if err != nil {
				t.Fatalf("translate transport error: %v", err)
			}
			if !bytes.Contains(result.Body, []byte(openRouterTransportErrorMessage)) {
				t.Fatalf("client body does not carry the upstream message: %s", result.Body)
			}
		})
	}
}

// The neutral error keeps the upstream code so status mapping sees the same
// 404 the provider reported, and the category follows from it.
func TestOpenRouterTransportErrorKeepsCodeAndCategory(t *testing.T) {
	transportError, _, err := NewBuiltinEngine().DecodeTransportError(
		llmprotocol.OpenAIChatV1, loadProviderFixture(t, openRouterTransportError),
	)
	if err != nil {
		t.Fatalf("decode transport error: %v", err)
	}
	if transportError.Error == nil {
		t.Fatal("decoded transport error carries no details")
	}
	if transportError.Error.Message != openRouterTransportErrorMessage {
		t.Fatalf("message = %q, want the upstream message", transportError.Error.Message)
	}
	if transportError.Error.Code != "404" {
		t.Fatalf("code = %q, want the upstream code as reported", transportError.Error.Code)
	}
	if transportError.Error.Category != llmprotocol.ErrorNotFound {
		t.Fatalf("category = %q, want not_found for a 404", transportError.Error.Category)
	}
}

// An envelope with no message says nothing a client can act on, so it is still
// refused. Tolerance covers the parts the contract does not need, not the one
// part that carries the answer.
func TestTransportErrorWithoutMessageStaysRefused(t *testing.T) {
	_, _, err := NewBuiltinEngine().DecodeTransportError(
		llmprotocol.OpenAIChatV1, []byte(`{"error":{"type":"server_error","code":500}}`),
	)
	if err == nil {
		t.Fatal("an upstream error envelope with no message was accepted")
	}
}
