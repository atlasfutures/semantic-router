package protocolcodec

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The tier an upstream served a turn at is billing telemetry: OpenAI and
// OpenRouter state it as a top-level service_tier on Chat and Responses,
// Anthropic as usage.service_tier. Every decode path keeps it on the neutral
// response or stream events, empty when the upstream did not say.

const servedTierChatResponse = `{"id":"chatcmpl-1","object":"chat.completion","created":7,"model":"model",` +
	`"service_tier":"flex","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],` +
	`"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`

const servedTierResponsesResponse = `{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"m",` +
	`"output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],` +
	`"service_tier":"priority","usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`

const servedTierMessagesResponse = `{"id":"msg_1","type":"message","role":"assistant","model":"claude",` +
	`"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,` +
	`"usage":{"input_tokens":3,"output_tokens":1,"service_tier":"standard"}}`

func TestServedServiceTierIsDecodedFromABufferedResponse(t *testing.T) {
	for name, test := range map[string]struct {
		format     llmprotocol.WireFormat
		body, want string
		member     string
	}{
		"chat":      {llmprotocol.OpenAIChatV1, servedTierChatResponse, "flex", `"service_tier":"flex",`},
		"responses": {llmprotocol.OpenAIResponsesV1, servedTierResponsesResponse, "priority", `"service_tier":"priority",`},
		"messages":  {llmprotocol.AnthropicMessagesV1, servedTierMessagesResponse, "standard", `,"service_tier":"standard"`},
	} {
		t.Run(name, func(t *testing.T) {
			response, _, _, err := NewBuiltinEngine().DecodeResponse(test.format, []byte(test.body))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if response.UpstreamServiceTier != test.want {
				t.Fatalf("served tier = %q, want %q", response.UpstreamServiceTier, test.want)
			}
			absent := strings.Replace(test.body, test.member, "", 1)
			response, _, _, err = NewBuiltinEngine().DecodeResponse(test.format, []byte(absent))
			if err != nil {
				t.Fatalf("decode without a tier: %v", err)
			}
			if response.UpstreamServiceTier != "" {
				t.Fatalf("served tier = %q, want none: the upstream named none", response.UpstreamServiceTier)
			}
		})
	}
}

// OpenRouter documents ultrafast beside OpenAI's tiers; a Chat response that
// names it still decodes.
func TestServedServiceTierAcceptsOpenRoutersUltrafast(t *testing.T) {
	body := strings.Replace(servedTierChatResponse, `"flex"`, `"ultrafast"`, 1)
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if response.UpstreamServiceTier != "ultrafast" {
		t.Fatalf("served tier = %q, want ultrafast", response.UpstreamServiceTier)
	}
}

func servedTierOf(events []llmprotocol.Event) (tier string, completed string) {
	for _, event := range events {
		if event.UpstreamServiceTier != "" {
			tier = event.UpstreamServiceTier
		}
		if event.Type == llmprotocol.EventResponseCompleted {
			completed = event.UpstreamServiceTier
		}
	}
	return tier, completed
}

func TestServedServiceTierReachesTheChatStreamEvents(t *testing.T) {
	chunk := func(tier, rest string) string {
		return `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m",` + tier + rest + "}\n\n"
	}
	for name, test := range map[string]struct{ member, want string }{
		"named":  {`"service_tier":"priority",`, "priority"},
		"absent": {"", ""},
	} {
		t.Run(name, func(t *testing.T) {
			body := chunk(test.member, `"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]`) +
				chunk(test.member, `"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]`) +
				chunk(test.member, `"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}`) +
				"data: [DONE]\n\n"
			_, events := runChatStream(t, name, []byte(body), llmprotocol.OpenAIChatV1)
			tier, completed := servedTierOf(events)
			if tier != test.want || completed != test.want {
				t.Fatalf("streamed tier = %q (completion %q), want %q", tier, completed, test.want)
			}
		})
	}
}

func TestServedServiceTierReachesTheMessagesStreamEvents(t *testing.T) {
	for name, test := range map[string]struct{ usage, want string }{
		"named":  {`"usage":{"input_tokens":3,"output_tokens":0,"service_tier":"priority"}`, "priority"},
		"absent": {`"usage":{"input_tokens":3,"output_tokens":0}`, ""},
	} {
		t.Run(name, func(t *testing.T) {
			body := strings.Replace(string(openRouterMessagesStream("")), `"usage":{"input_tokens":3,"output_tokens":0}`, test.usage, 1)
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public-model", ProviderModel: "claude",
			})
			if err != nil {
				t.Fatal(err)
			}
			_, events, _, err := stream.Push([]byte(body))
			if err != nil {
				t.Fatalf("push: %v", err)
			}
			_, finalEvents, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatalf("finalize: %v", err)
			}
			tier, completed := servedTierOf(append(events, finalEvents...))
			if tier != test.want || completed != test.want {
				t.Fatalf("streamed tier = %q (completion %q), want %q", tier, completed, test.want)
			}
		})
	}
}

func TestServedServiceTierReachesTheResponsesStreamCompletion(t *testing.T) {
	for name, test := range map[string]struct{ member, want string }{
		"named":  {`"service_tier":"flex",`, "flex"},
		"absent": {"", ""},
	} {
		t.Run(name, func(t *testing.T) {
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"})
			if err != nil {
				t.Fatal(err)
			}
			payload := "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"response_1\",\"model\":\"model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
				"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n" +
				"event: response.function_call_arguments.done\ndata: {\"type\":\"response.function_call_arguments.done\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"item_1\",\"name\":\"lookup\",\"arguments\":\"{}\"}\n\n" +
				"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":3,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{}\",\"status\":\"completed\"}}\n\n" +
				"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":4,\"response\":{\"id\":\"response_1\",\"model\":\"model\",\"status\":\"completed\"," + test.member +
				"\"output\":[{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{}\",\"status\":\"completed\"}],\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n"
			_, events, _, err := stream.Push([]byte(payload))
			if err != nil {
				t.Fatalf("push: %v", err)
			}
			_, finalEvents, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatalf("finalize: %v", err)
			}
			events = append(events, finalEvents...)
			if _, completed := servedTierOf(events); completed != test.want {
				t.Fatalf("completion tier = %q, want %q; events %+v", completed, test.want, events)
			}
		})
	}
}
