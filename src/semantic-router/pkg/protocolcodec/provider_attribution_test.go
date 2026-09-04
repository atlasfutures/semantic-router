package protocolcodec

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// OpenRouter names the upstream that served the turn in a top-level "provider"
// member. Caching, thinking-off handling and empty completions differ by that
// provider rather than by model, so the name has to survive decoding. It is
// telemetry: the Router reads it and never hands it to a client.
func TestUpstreamProviderIsDecodedFromTheOpenRouterResponse(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		loadProviderFixture(t, openRouterResponse), renameModel("public-model"),
	)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	response := result.Response
	if response.UpstreamProvider != "Ionstream" {
		t.Fatalf("upstream provider = %q, want the name OpenRouter reported", response.UpstreamProvider)
	}
}

// The provider name is not part of any client contract. No target may publish
// it, on any encode path.
func TestUpstreamProviderIsNeverEncodedForAClient(t *testing.T) {
	response := llmprotocol.Response{
		Generation: 1, ID: "gen-1", Model: "public-model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_0", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hi"}},
		}},
		StopReason:       llmprotocol.StopEndTurn,
		SourceStopReason: "stop",
		UpstreamProvider: "Ionstream",
		Usage: llmprotocol.Usage{
			State:       llmprotocol.UsageAvailable,
			InputTotal:  llmprotocol.TokenCount{Value: llmprotocol.Int64(11), Provenance: llmprotocol.UsageAuthoritative},
			OutputTotal: llmprotocol.TokenCount{Value: llmprotocol.Int64(8), Provenance: llmprotocol.UsageAuthoritative},
			Total:       llmprotocol.TokenCount{Value: llmprotocol.Int64(19), Provenance: llmprotocol.UsageAuthoritative},
		},
	}
	engine := NewBuiltinEngine()
	for name, target := range map[string]llmprotocol.WireFormat{
		"chat":      llmprotocol.OpenAIChatV1,
		"anthropic": llmprotocol.AnthropicMessagesV1,
		"responses": llmprotocol.OpenAIResponsesV1,
	} {
		t.Run(name, func(t *testing.T) {
			result, err := engine.EncodeResponse(target, response, llmprotocol.Envelope{})
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if strings.Contains(string(result.Body), "Ionstream") {
				t.Fatalf("the provider name reached the client body: %s", result.Body)
			}
		})
	}
}

// A streamed turn is the one the empties were measured on, so the provider name
// and the upstream finish reason have to reach the neutral events the Router
// reconstructs a stream from.
func TestUpstreamAttributionReachesTheChatStreamEvents(t *testing.T) {
	body, events := runProviderStream(t, openRouterStream, llmprotocol.OpenAIChatV1)
	provider, sourceStop := "", ""
	for _, event := range events {
		if event.UpstreamProvider != "" {
			provider = event.UpstreamProvider
		}
		if event.SourceStopReason != "" {
			sourceStop = event.SourceStopReason
		}
	}
	if strings.Contains(string(body), "Venice") {
		t.Fatalf("the provider name reached the client stream: %s", body)
	}
	if provider != "Venice" {
		t.Fatalf("streamed upstream provider = %q, want the name OpenRouter reported", provider)
	}
	if sourceStop != "stop" {
		t.Fatalf("streamed upstream finish reason = %q, want the reason the upstream sent", sourceStop)
	}
}
