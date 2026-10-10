package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
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

// OpenRouter's Messages surface names the upstream the same way: a top-level
// "provider" on a buffered response, and on the message_start message of a
// stream (semantic-router#257). Every v5 cell dispatches Messages first.
const openRouterMessagesResponse = `{"id":"msg_1","type":"message","role":"assistant",` +
	`"model":"moonshotai/kimi-k3","provider":"Moonshot AI",` +
	`"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,` +
	`"usage":{"input_tokens":3,"output_tokens":1}}`

func openRouterMessagesStream(provider string) []byte {
	start := `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",` +
		`"model":"moonshotai/kimi-k3",` + provider + `"content":[],"stop_reason":null,"stop_sequence":null,` +
		`"usage":{"input_tokens":3,"output_tokens":0}}}`
	return []byte("event: message_start\ndata: " + start + "\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
}

func TestUpstreamProviderIsDecodedFromAnOpenRouterMessagesResponse(t *testing.T) {
	for name, target := range map[string]llmprotocol.WireFormat{
		"anthropic": llmprotocol.AnthropicMessagesV1,
		"chat":      llmprotocol.OpenAIChatV1,
	} {
		t.Run(name, func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateResponse(
				llmprotocol.AnthropicMessagesV1, target, []byte(openRouterMessagesResponse), renameModel("public-model"),
			)
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			if result.Response.UpstreamProvider != "Moonshot AI" {
				t.Fatalf("upstream provider = %q, want the name OpenRouter reported", result.Response.UpstreamProvider)
			}
			if strings.Contains(string(result.Body), "Moonshot AI") {
				t.Fatalf("the provider name reached the client body: %s", result.Body)
			}
		})
	}
}

func TestUpstreamProviderIsEmptyWhenAMessagesResponseNamesNone(t *testing.T) {
	body := strings.Replace(openRouterMessagesResponse, `"provider":"Moonshot AI",`, "", 1)
	result, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(body), renameModel("public-model"),
	)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if result.Response.UpstreamProvider != "" {
		t.Fatalf("upstream provider = %q, want none: the upstream named none", result.Response.UpstreamProvider)
	}
}

func TestUpstreamProviderReachesTheMessagesStreamEvents(t *testing.T) {
	for name, provider := range map[string]string{"named": "Moonshot AI", "absent": ""} {
		t.Run(name, func(t *testing.T) {
			member := ""
			if provider != "" {
				member = `"provider":"` + provider + `",`
			}
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public-model", ProviderModel: "moonshotai/kimi-k3",
			})
			if err != nil {
				t.Fatal(err)
			}
			frames, events, _, err := stream.Push(openRouterMessagesStream(member))
			if err != nil {
				t.Fatalf("push: %v", err)
			}
			finalFrames, finalEvents, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatalf("finalize: %v", err)
			}
			events = append(events, finalEvents...)
			body := append(bytes.Join(frames, nil), bytes.Join(finalFrames, nil)...)
			got, completedNamed := "", false
			for _, event := range events {
				if event.UpstreamProvider != "" {
					got = event.UpstreamProvider
				}
				if event.Type == llmprotocol.EventResponseCompleted && event.UpstreamProvider == provider {
					completedNamed = true
				}
			}
			if got != provider {
				t.Fatalf("streamed upstream provider = %q, want %q", got, provider)
			}
			if !completedNamed {
				t.Fatalf("the completion event does not carry the provider %q", provider)
			}
			if provider != "" && strings.Contains(string(body), provider) {
				t.Fatalf("the provider name reached the client stream: %s", body)
			}
		})
	}
}

// A same-format turn with no rename would replay the upstream's bytes; a body
// that names its provider is encoded instead, so the name stays the Router's.
func TestUpstreamProviderIsNotReplayedToASameFormatClient(t *testing.T) {
	cases := map[string]struct {
		format llmprotocol.WireFormat
		body   []byte
		name   string
	}{
		"messages": {llmprotocol.AnthropicMessagesV1, []byte(openRouterMessagesResponse), "Moonshot AI"},
		"chat":     {llmprotocol.OpenAIChatV1, loadProviderFixture(t, openRouterResponse), "Ionstream"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateResponse(test.format, test.format, test.body, nil)
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			if result.Response.UpstreamProvider != test.name {
				t.Fatalf("upstream provider = %q, want %q", result.Response.UpstreamProvider, test.name)
			}
			if strings.Contains(string(result.Body), test.name) {
				t.Fatalf("the provider name was replayed to the client: %s", result.Body)
			}
		})
	}
}

// A stream whose last frame has no trailing blank line completes in Finalize;
// that completion still names the provider.
func TestUpstreamProviderReachesAMessagesStreamCompletedInFinalize(t *testing.T) {
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "moonshotai/kimi-k3",
	})
	if err != nil {
		t.Fatal(err)
	}
	body := bytes.TrimSuffix(openRouterMessagesStream(`"provider":"Moonshot AI",`), []byte("\n\n"))
	if _, _, _, err := stream.Push(body); err != nil {
		t.Fatalf("push: %v", err)
	}
	_, events, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	for _, event := range events {
		if event.Type == llmprotocol.EventResponseCompleted {
			if event.UpstreamProvider != "Moonshot AI" {
				t.Fatalf("completion in Finalize names provider %q, want Moonshot AI", event.UpstreamProvider)
			}
			return
		}
	}
	t.Fatalf("Finalize produced no completion: %+v", events)
}

func TestWithoutTopLevelMemberKeepsEveryOtherByte(t *testing.T) {
	cases := map[string]struct{ in, want string }{
		"middle": {`{"id":"x", "provider":"P","model":"m"}`, `{"id":"x","model":"m"}`},
		"first":  {`{"provider":"P", "id":"x"}`, `{"id":"x"}`},
		"last":   {`{"id":"x","provider":"P"}`, `{"id":"x"}`},
		"only":   {`{"provider":"P"}`, `{}`},
		"nested": {`{"a":{"provider":"keep"},"provider":"P"}`, `{"a":{"provider":"keep"}}`},
		"absent": {`{"id":"x"}`, `{"id":"x"}`},
		"pretty": {"{\n  \"id\": 1,\n  \"provider\": \"P\",\n  \"m\": [1, 2]\n}", "{\n  \"id\": 1,\n  \"m\": [1, 2]\n}"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			got := withoutTopLevelMember([]byte(test.in), "provider")
			if string(got) != test.want {
				t.Fatalf("got %s, want %s", got, test.want)
			}
			if !json.Valid(got) {
				t.Fatalf("result is not JSON: %s", got)
			}
		})
	}
	if got := withoutTopLevelMember([]byte(`[1]`), "provider"); got != nil {
		t.Fatalf("a non-object kept bytes: %s", got)
	}
}

// Stripping the name keeps what only a replay carries: Chat logprobs.
func TestUpstreamProviderStripKeepsAChatReplaysLogprobs(t *testing.T) {
	body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":7,"model":"model","provider":"Moonshot AI",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"},` +
		`"logprobs":{"content":[{"token":"hi","logprob":-0.1,"bytes":[104,105],"top_logprobs":[]}]}}],` +
		`"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if strings.Contains(string(result.Body), "Moonshot AI") || !strings.Contains(string(result.Body), `"logprob":-0.1`) {
		t.Fatalf("client body = %s, want logprobs kept and the provider gone", result.Body)
	}
}
