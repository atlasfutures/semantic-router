package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An upstream completion carrying no content is a legal outcome. A provider
// closes such a turn as an empty content delta then a finish, or a finish alone.
const (
	emptyContentDelta = `"delta":{"role":"assistant","content":""},"finish_reason":null`
	textDelta         = `"delta":{"role":"assistant","content":"hi"},"finish_reason":null`
	finishStop        = `"delta":{},"finish_reason":"stop"`
)

// chatStreamOf builds a Chat SSE body from one choice fragment per chunk.
func chatStreamOf(choices ...string) string {
	body := strings.Builder{}
	for _, choice := range choices {
		body.WriteString(`data: {"id":"chatcmpl_1","object":"chat.completion.chunk",` +
			`"model":"provider-model","choices":[{"index":0,` + choice + "}]}\n\n")
	}
	return body.String() + "data: [DONE]\n\n"
}

// anthropicStreamOutput translates a Chat SSE body to Messages and returns the
// encoded client stream with its event names in the order they were sent.
func anthropicStreamOutput(t *testing.T, payload string) (string, string) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, pushErr := stream.Push([]byte(payload))
	final, _, _, finalErr := stream.Finalize(nil)
	if pushErr != nil || finalErr != nil {
		t.Fatal(pushErr, finalErr)
	}
	output := string(bytes.Join(append(frames, final...), nil))
	var names []string
	for _, line := range strings.Split(output, "\n") {
		if name, found := strings.CutPrefix(line, "event: "); found {
			names = append(names, strings.TrimSpace(name))
		}
	}
	return output, strings.Join(names, ",")
}

// TestEmptyCompletionOpensNoContentBlock is the ask. A choice that finishes
// with no delta is given a synthesized ContentText (stream_chat.go:435-445),
// and an item owning no block key has one opened on completion
// (stream_anthropic.go:668-681): the client is served content_block_start and
// content_block_stop carrying text "" for an answer the model never gave.
func TestEmptyCompletionOpensNoContentBlock(t *testing.T) {
	for name, payload := range map[string]string{
		"empty content then finish": chatStreamOf(emptyContentDelta, finishStop),
		"finish only":               chatStreamOf(finishStop),
	} {
		t.Run(name, func(t *testing.T) {
			output, sequence := anthropicStreamOutput(t, payload)
			if strings.Contains(sequence, "content_block") {
				t.Errorf("an empty completion opened a content block: %s", sequence)
			}
			if !strings.Contains(sequence, "message_start") || !strings.Contains(sequence, "message_delta,message_stop") {
				t.Errorf("an empty completion lost its terminal events: %s", sequence)
			}
			if !strings.Contains(output, `"stop_reason":"end_turn"`) {
				t.Errorf("an empty completion lost its stop reason:\n%s", output)
			}
		})
	}
}

// TestEmptyCompletionOpensAnEmptyTextBlockToday pins the present behaviour so a
// change to it shows up in the diff rather than only in the test above.
func TestEmptyCompletionOpensAnEmptyTextBlockToday(t *testing.T) {
	output, sequence := anthropicStreamOutput(t, chatStreamOf(finishStop))
	if !strings.Contains(sequence, "content_block_start,content_block_stop") ||
		!strings.Contains(output, `"content_block":{"text":"","type":"text"}`) {
		t.Fatalf("the recorded behaviour has changed: %s\n%s", sequence, output)
	}
}

// A completion that does carry text still opens, fills and closes its block.
func TestNonEmptyCompletionStillOpensItsBlock(t *testing.T) {
	_, sequence := anthropicStreamOutput(t, chatStreamOf(textDelta, finishStop))
	for _, want := range []string{"content_block_start", "content_block_delta", "content_block_stop"} {
		if !strings.Contains(sequence, want) {
			t.Fatalf("a completion carrying text lost %s: %s", want, sequence)
		}
	}
}
