package protocolcodec

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A provider that ends a turn on its token limit can state more reasoning
// tokens than completion tokens: 64 completion against 79 reasoning. The
// completion is well formed and already billed, but the decoder derives a
// sub-split from those two numbers that the validator then refuses.
// codec_openai_chat_response.go:170-172 writes -1 for the remainder, which
// validate_response.go:247-249 rejects as negative_usage. The two Anthropic
// copies, codec_anthropic_response.go:136-139 and stream_anthropic.go:368-371,
// clamp it to 0 and call that authoritative, so 79 + 0 misses the stated total
// 64 and validate_response.go:266-280 rejects the sum as usage_total_mismatch.
var (
	chatOvershootBody = []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"provider-model",` +
		`"choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","content":"hi"}}],` +
		`"usage":{"prompt_tokens":20,"completion_tokens":64,"total_tokens":84,` +
		`"completion_tokens_details":{"reasoning_tokens":79}}}`)
	anthropicOvershootBody = []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"provider-model",` +
		`"content":[{"type":"text","text":"hi"}],"stop_reason":"max_tokens","stop_sequence":null,` +
		`"usage":{"input_tokens":20,"output_tokens":64,"cache_creation_input_tokens":0,` +
		`"cache_read_input_tokens":0,"output_tokens_details":{"thinking_tokens":79}}}`)
	anthropicOvershootStream = []byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":20,\"output_tokens\":0,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0}}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":20,\"output_tokens\":64,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens_details\":{\"thinking_tokens\":79}}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
)

// TestReasoningOvershootStillServesTheChatCompletion is the ask: a sub-split
// the router cannot reconcile should be left unknown, not synthesised.
func TestReasoningOvershootStillServesTheChatCompletion(t *testing.T) {
	assertOvershootIsServed(t, llmprotocol.OpenAIChatV1, chatOvershootBody)
}

// TestThinkingOvershootStillServesTheAnthropicResponse is the same ask on the
// Anthropic decoders, whose buffered and streamed copies have both drifted.
func TestThinkingOvershootStillServesTheAnthropicResponse(t *testing.T) {
	assertOvershootIsServed(t, llmprotocol.AnthropicMessagesV1, anthropicOvershootBody)
	t.Run("streamed", func(t *testing.T) {
		stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
			llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
		if err != nil {
			t.Fatalf("new stream: %v", err)
		}
		if _, _, _, err := stream.Push(anthropicOvershootStream); err != nil {
			t.Fatalf("push: %v", err)
		}
	})
}

// assertOvershootIsServed keeps the totals the provider did state.
func assertOvershootIsServed(t *testing.T, source llmprotocol.WireFormat, body []byte) {
	t.Helper()
	for name, target := range map[string]llmprotocol.WireFormat{
		"to_chat": llmprotocol.OpenAIChatV1, "to_anthropic": llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(name, func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateResponse(source, target, body, nil)
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			if len(result.Body) == 0 {
				t.Fatal("translated body is empty")
			}
			if total := result.Response.Usage.OutputTotal; total.Value == nil || *total.Value != 64 {
				t.Fatalf("output total = %v, want the stated 64", total.Value)
			}
		})
	}
}

// TestReasoningOvershootIsRejectedToday pins the present behaviour so a change
// to it shows up in the diff rather than only in the tests above.
func TestReasoningOvershootIsRejectedToday(t *testing.T) {
	for name, want := range map[string]struct {
		format llmprotocol.WireFormat
		body   []byte
		code   string
	}{
		"chat":      {llmprotocol.OpenAIChatV1, chatOvershootBody, "negative_usage"},
		"anthropic": {llmprotocol.AnthropicMessagesV1, anthropicOvershootBody, "usage_total_mismatch"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := NewBuiltinEngine().TranslateResponse(want.format, want.format, want.body, nil)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) {
				t.Fatalf("translate = %v, want a protocol error", err)
			}
			if protocolError.Code != want.code {
				t.Fatalf("code = %q, want the recorded %q", protocolError.Code, want.code)
			}
		})
	}
}
