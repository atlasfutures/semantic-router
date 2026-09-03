package protocolcodec

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A turn on deepseek/deepseek-v4-flash@thinking-on ended on 2026-09-03 with
// output_tokens 1, thinking_tokens 0 and stop_reason end_turn: the model spent
// its turn on a stop token and wrote nothing. The client was served
// message_start, content_block_start with text "", content_block_stop,
// message_delta, message_stop -- an empty text block the upstream never sent.
// Claude Code rendered nothing and the turn was billed.
//
// The two fixtures below are the upstream shapes that reproduce it. Neither
// carries any content, so neither may produce a content block: an Anthropic
// message with content [] is what actually happened, and inventing a block
// tells the client the model answered when it did not.
const (
	openRouterStreamEmptyContentThenFinish = `data: {"id":"gen-1","object":"chat.completion.chunk","created":1788474769,` +
		`"model":"deepseek/deepseek-v4-flash","choices":[{"index":0,` +
		`"delta":{"content":"","role":"assistant"},"finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1788474769,` +
		`"model":"deepseek/deepseek-v4-flash","choices":[{"index":0,"delta":{},` +
		`"finish_reason":"stop"}],"usage":{"prompt_tokens":220,"completion_tokens":1,"total_tokens":221}}

data: [DONE]

`
	openRouterStreamFinishOnly = `data: {"id":"gen-1","object":"chat.completion.chunk","created":1788474769,` +
		`"model":"deepseek/deepseek-v4-flash","choices":[{"index":0,"delta":{},` +
		`"finish_reason":"stop"}],"usage":{"prompt_tokens":220,"completion_tokens":1,"total_tokens":221}}

data: [DONE]

`
)

func TestEmptyCompletionOpensNoContentBlock(t *testing.T) {
	for name, fixture := range map[string]string{
		"empty content then finish": openRouterStreamEmptyContentThenFinish,
		"finish only":               openRouterStreamFinishOnly,
	} {
		t.Run(name, func(t *testing.T) {
			body, _ := runChatStream(t, name, []byte(fixture), llmprotocol.AnthropicMessagesV1)
			for _, event := range anthropicEventSequence(body) {
				if strings.HasPrefix(event, "content_block") {
					t.Fatalf("an empty completion produced %s:\n%s", event, body)
				}
			}
			// The turn still ends, and it ends honestly.
			assertAnthropicEventsInOrder(t, body, "message_start", "message_delta", "message_stop")
			if !strings.Contains(body2string(body), `"stop_reason":"end_turn"`) {
				t.Fatalf("an empty completion lost its stop reason:\n%s", body)
			}
		})
	}
}

// A completion that does carry content is untouched: the block still opens,
// still receives its delta and still closes.
func TestNonEmptyCompletionStillOpensItsBlock(t *testing.T) {
	body, _ := runProviderStream(t, openRouterStream, llmprotocol.AnthropicMessagesV1)
	sequence := strings.Join(anthropicEventSequence(body), ",")
	for _, want := range []string{"content_block_start", "content_block_delta", "content_block_stop"} {
		if !strings.Contains(sequence, want) {
			t.Fatalf("a completion carrying text lost %s: %s", want, sequence)
		}
	}
}

func assertAnthropicEventsInOrder(t *testing.T, body []byte, want ...string) {
	t.Helper()
	got := anthropicEventSequence(body)
	next := 0
	for _, event := range got {
		if next < len(want) && event == want[next] {
			next++
		}
	}
	if next != len(want) {
		t.Fatalf("events %v do not contain %v in order", got, want)
	}
}

func body2string(body []byte) string { return string(body) }
