package protocolcodec

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// OpenRouter streams Claude's thinking over Responses as a reasoning item
// whose reasoning_text part is announced with content_part.added and closed
// with content_part.done, the same events a message uses for output_text:
//
//	output_item.added    {item: {type: reasoning}}
//	content_part.added   {part: {type: reasoning_text, text: ""}}
//	reasoning_text.delta / reasoning_text.done
//	content_part.done    {part: {type: reasoning_text, text}}
//	output_item.done     {item: {type: reasoning, content: [reasoning_text], signature, format}}
//
// The decoder took every content_part event for a message part and aborted
// the stream with stream_item_kind_mismatch, so every Claude turn that
// thought before calling a tool failed. The fixture is a captured stream of
// that shape with its ids, text, signature and tools replaced.
const openRouterClaudeReasoningStream = "responses-stream-claude-reasoning.sse"

const openRouterClaudeReasoningText = "The user wants 6*7 evaluated with the js tool, so I will call it."

func pushResponsesStream(body []byte, client llmprotocol.WireFormat) ([]byte, error) {
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, client, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "anthropic/claude-sonnet-5.5",
		ProviderModel: "anthropic/claude-sonnet-5.5", WorkerFamily: "anthropic",
	})
	if err != nil {
		return nil, err
	}
	frames, _, _, err := stream.Push(body)
	if err != nil {
		return nil, err
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		return nil, err
	}
	return bytes.Join(append(frames, final...), nil), nil
}

func TestOpenRouterClaudeReasoningStreamReachesEveryClient(t *testing.T) {
	body := loadProviderFixture(t, openRouterClaudeReasoningStream)
	for _, client := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(client), func(t *testing.T) {
			routed, err := pushResponsesStream(body, client)
			if err != nil {
				t.Fatalf("stream failed: %v", err)
			}
			for _, want := range []string{"6*7 evaluated ", `\"code\":`} {
				if !bytes.Contains(routed, []byte(want)) {
					t.Fatalf("routed stream lost %q:\n%s", want, routed)
				}
			}
		})
	}
}

// A reasoning_text part is reasoning content: it belongs to a reasoning item
// and to no other, and a message part still belongs only to a message.
func TestResponsesContentPartMustMatchItsItem(t *testing.T) {
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	for name, mutated := range map[string]string{
		"reasoning_text part on a message": strings.Replace(body,
			`"item":{"id":"rs_fixture_1","type":"reasoning","status":"in_progress","summary":[]}`,
			`"item":{"id":"rs_fixture_1","type":"message","role":"assistant","status":"in_progress","content":[]}`, 1),
		"output_text part on a reasoning item": strings.Replace(body,
			`"part":{"type":"reasoning_text","text":""}`,
			`"part":{"type":"output_text","text":"","annotations":[]}`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			if mutated == body {
				t.Fatal("the fixture no longer has the event this case rewrites")
			}
			_, err := pushResponsesStream([]byte(mutated), llmprotocol.OpenAIResponsesV1)
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Code != "stream_item_kind_mismatch" {
				t.Fatalf("got %v, want stream_item_kind_mismatch", err)
			}
		})
	}
}
