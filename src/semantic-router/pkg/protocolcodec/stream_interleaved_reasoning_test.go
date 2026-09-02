package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An OpenRouter-style upstream can emit reasoning, then visible text, then more
// reasoning inside one choice. Anthropic Messages cannot resume a content block
// once another block has started, so the encoder degrades: it closes the open
// block and carries the resumed reasoning onto a fresh thinking block at the
// next index. Nothing is dropped, and a Messages client sees a valid stream.
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
		if pushErr != nil {
			t.Fatalf("frame %d failed: %v", index, pushErr)
		}
		public = append(public, emitted...)
	}
	final, _, _, finalErr := stream.Finalize(nil)
	if finalErr != nil {
		t.Fatalf("finalize failed: %v", finalErr)
	}
	public = append(public, final...)
	body := bytes.Join(public, nil)
	want := []string{
		"message_start",
		"content_block_start 0 thinking",
		"content_block_delta 0 thinking_delta first ",
		"content_block_stop 0",
		"content_block_start 1 text",
		"content_block_delta 1 text_delta visible ",
		"content_block_stop 1",
		"content_block_start 2 thinking",
		"content_block_delta 2 thinking_delta second",
		"content_block_stop 2",
		"message_delta",
		"message_stop",
	}
	got := describeAnthropicStream(t, body)
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("interleaved reasoning stream shape:\nwant:\n%s\ngot:\n%s", strings.Join(want, "\n"), strings.Join(got, "\n"))
	}
}

// describeAnthropicStream reduces an encoded Messages stream to one line per
// event, so a test can pin block ordering and indexes without matching bytes.
func describeAnthropicStream(t *testing.T, body []byte) []string {
	t.Helper()
	var described []string
	for _, line := range strings.Split(string(body), "\n") {
		payload, found := strings.CutPrefix(line, "data: ")
		if !found {
			continue
		}
		var wire anthropicEventWire
		if err := json.Unmarshal([]byte(payload), &wire); err != nil {
			t.Fatalf("stream carried an unparsable frame %q: %v", payload, err)
		}
		described = append(described, describeAnthropicEvent(wire))
	}
	return described
}

func describeAnthropicEvent(wire anthropicEventWire) string {
	if wire.Index == nil {
		return wire.Type
	}
	switch {
	case wire.ContentBlock != nil:
		return fmt.Sprintf("%s %d %s", wire.Type, *wire.Index, wire.ContentBlock.Type)
	case wire.Delta != nil:
		return fmt.Sprintf("%s %d %s %s%s", wire.Type, *wire.Index, wire.Delta.Type, wire.Delta.Text, wire.Delta.Thinking)
	default:
		return fmt.Sprintf("%s %d", wire.Type, *wire.Index)
	}
}
