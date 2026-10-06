package protocolcodec

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Claude arm served over Messages answers with signed thinking. A Chat
// client has no member for the signature, so the answer reaches it with the
// thinking text as reasoning_content and without the signature, as a request
// leg's CarryReasoningTo strips one for a Chat target -- not as a refusal
// (atlasfutures/semantic-router#212). A Responses client keeps the signature
// in the reasoning item's OpenRouter members, as before.
const anthropicSignedThinkingResponse = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5.5",` +
	`"content":[{"type":"thinking","thinking":"Check the tool.","signature":"EqQBsigclaude"},{"type":"text","text":"Done."}],` +
	`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":5}}`

func assertSignedThinkingTranslated(t *testing.T, target llmprotocol.WireFormat, body string) {
	t.Helper()
	if !strings.Contains(body, "Check the tool.") || !strings.Contains(body, "Done.") {
		t.Fatalf("%s answer lost the thinking or the text: %s", target, body)
	}
	switch target {
	case llmprotocol.OpenAIChatV1:
		if strings.Contains(body, "EqQBsigclaude") {
			t.Fatalf("a signature reached a Chat client: %s", body)
		}
	case llmprotocol.OpenAIResponsesV1:
		if !strings.Contains(body, "EqQBsigclaude") {
			t.Fatalf("a Responses client lost Claude's signature: %s", body)
		}
	}
}

func TestSignedThinkingResponseReachesChatAndResponsesClients(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for name, mutate := range map[string]ResponseMutation{
			"unmutated": nil,
			"mutated":   func(response *llmprotocol.Response) error { response.Model = "public"; return nil },
		} {
			t.Run(string(target)+"/translate/"+name, func(t *testing.T) {
				result, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, target, []byte(anthropicSignedThinkingResponse), mutate)
				if err != nil {
					t.Fatalf("translate: %v", err)
				}
				assertSignedThinkingTranslated(t, target, string(result.Body))
			})
		}
		t.Run(string(target)+"/encode", func(t *testing.T) {
			decoded, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(anthropicSignedThinkingResponse), nil)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			result, err := engine.EncodeResponse(target, decoded.Response, llmprotocol.Envelope{})
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			assertSignedThinkingTranslated(t, target, string(result.Body))
		})
		t.Run(string(target)+"/encode stream", func(t *testing.T) {
			decoded, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(anthropicSignedThinkingResponse), nil)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			body, _, err := engine.EncodeResponseStream(target, decoded.Response, llmprotocol.StreamContext{Context: context.Background()})
			if err != nil {
				t.Fatalf("encode stream: %v", err)
			}
			assertSignedThinkingTranslated(t, target, string(body))
		})
	}
}

// The live stream from a Messages arm carries the signature as a
// signature_delta at the thinking block's end.
func TestSignedThinkingStreamReachesChatAndResponsesClients(t *testing.T) {
	events := []string{
		`event: message_start` + "\n" + `data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5.5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":1}}}`,
		`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Check the tool."}}`,
		`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBsigclaude"}}`,
		`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}`,
		`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Done."}}`,
		`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":1}`,
		`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`,
		`event: message_stop` + "\n" + `data: {"type":"message_stop"}`,
	}
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, target, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public", ProviderModel: "claude-opus-5.5",
			})
			if err != nil {
				t.Fatal(err)
			}
			var wire strings.Builder
			for _, event := range events {
				frames, _, _, pushErr := stream.Push([]byte(event + "\n\n"))
				if pushErr != nil {
					t.Fatalf("push: %v", pushErr)
				}
				for _, frame := range frames {
					wire.Write(frame)
				}
			}
			frames, _, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatalf("finalize: %v", err)
			}
			for _, frame := range frames {
				wire.Write(frame)
			}
			assertSignedThinkingTranslated(t, target, wire.String())
		})
	}
}
