package protocolcodec

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// service_tier "auto" is every source API's default and asks for nothing, so
// a request carrying it is served, and the member is not dispatched
// (semantic-router#163: OpenRouter's benchmark harness sends it on every
// Responses call). Control: an explicit tier is still refused.
func TestServiceTierAutoIsServedAndExplicitTiersAreRefused(t *testing.T) {
	engine := NewBuiltinEngine()
	cases := []struct {
		format   llmprotocol.WireFormat
		auto     string
		explicit string
	}{
		{llmprotocol.OpenAIResponsesV1, `{"model":"m","input":"hi","service_tier":"auto"}`, `{"model":"m","input":"hi","service_tier":"flex"}`},
		{llmprotocol.OpenAIChatV1, `{"model":"m","messages":[{"role":"user","content":"hi"}],"service_tier":"auto"}`, `{"model":"m","messages":[{"role":"user","content":"hi"}],"service_tier":"priority"}`},
		{llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"service_tier":"auto"}`, `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"service_tier":"standard_only"}`},
	}
	for _, tc := range cases {
		request, _, _, err := engine.DecodeRequest(tc.format, []byte(tc.auto))
		if err != nil {
			t.Fatalf("%s: service_tier auto refused: %v", tc.format, err)
		}
		for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
			encoded, err := engine.EncodeRequest(target, request, llmprotocol.Envelope{})
			if err != nil {
				t.Fatalf("%s -> %s: %v", tc.format, target, err)
			}
			if bytes.Contains(encoded.Body, []byte("service_tier")) {
				t.Fatalf("%s -> %s: dispatched the tier: %s", tc.format, target, encoded.Body)
			}
		}
		_, _, _, err = engine.DecodeRequest(tc.format, []byte(tc.explicit))
		assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_service_tier")
	}
}
