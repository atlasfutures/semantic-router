package protocolcodec

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An OpenRouter-style upstream can emit reasoning, then visible text, then more
// reasoning inside one choice. Anthropic Messages cannot resume a content block
// once another block has started, so this is the stream shape most likely to
// break a cell that answers Anthropic and dispatches to such a provider.
//
// This test records what happens today. It asserts nothing about which
// behaviour is right; it fails only if the outcome stops being one of the two
// the encoder can produce, so the answer stays in the tree.
func TestChatInterleavedReasoningIntoAnthropicStream(t *testing.T) {
	frames := []string{
		`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning":"first "}}]}`,
		`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"visible "}}]}`,
		`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"reasoning":"second"}}]}`,
		`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	var public [][]byte
	for index, frame := range frames {
		emitted, _, _, pushErr := stream.Push([]byte(frame + "\n\n"))
		public = append(public, emitted...)
		if pushErr == nil {
			continue
		}
		var protocolError *llmprotocol.ProtocolError
		if !errors.As(pushErr, &protocolError) {
			t.Fatalf("frame %d failed with a non-protocol error: %v", index, pushErr)
		}
		t.Logf(
			"interleaved reasoning fails closed at frame %d: %s %s %q",
			index, protocolError.Category, protocolError.Code, protocolError.Message,
		)
		if protocolError.Code != "anthropic_content_interleaving" {
			t.Fatalf("interleaving now fails with %q, which this test does not describe", protocolError.Code)
		}
		return
	}
	body := bytes.Join(public, nil)
	if !bytes.Contains(body, []byte("thinking_delta")) || !bytes.Contains(body, []byte("text_delta")) {
		t.Fatalf("interleaved reasoning was accepted but lost a block: %s", body)
	}
	t.Logf("interleaved reasoning is served: %s", body)
}
