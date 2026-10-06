package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The OpenAI stream_options.include_usage final chunk carries "choices": []
// beside usage, and strict clients (opencode) reject one without the key.
// Every source -- a Messages or Responses upstream translated to Chat, and a
// Chat upstream re-encoded -- ends with that shape.
func TestChatStreamUsageChunkCarriesEmptyChoices(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.OpenAIChatV1,
	} {
		t.Run(string(source), func(t *testing.T) {
			stream, err := engine.NewStream(source, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public-model",
				Options: llmprotocol.StreamOptions{IncludeUsage: boolPointer(true)},
			})
			if err != nil {
				t.Fatal(err)
			}
			frames, _, _, err := stream.Push(streamFixture(source))
			if err != nil {
				t.Fatal(err)
			}
			final, _, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			var usageChunks int
			for _, frame := range append(frames, final...) {
				for _, line := range strings.Split(string(frame), "\n") {
					data, ok := strings.CutPrefix(line, "data: ")
					if !ok || data == "[DONE]" {
						continue
					}
					var chunk map[string]json.RawMessage
					if err := json.Unmarshal([]byte(data), &chunk); err != nil {
						t.Fatalf("chunk is not JSON: %s", data)
					}
					choices, present := chunk["choices"]
					if !present {
						t.Errorf("chunk has no choices key: %s", data)
					}
					if _, usage := chunk["usage"]; usage {
						usageChunks++
						if !bytes.Equal(choices, []byte("[]")) {
							t.Errorf("usage chunk choices = %s, want []: %s", choices, data)
						}
					}
				}
			}
			if usageChunks != 1 {
				t.Fatalf("usage chunks = %d, want 1", usageChunks)
			}
		})
	}
}
