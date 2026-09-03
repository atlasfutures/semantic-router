package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A thinking turn that outruns the platform deadline has its connection closed
// mid-delta. Measured on the dev cell 2026-09-03: 3 of 17 thinking-arm turns
// were cut at 630 s (Envoy stream_idle_timeout 620 s under Cloud Run 630 s),
// and t20 reached the client as message_start, content_block_start, 1,191
// content_block_delta frames, then a clean close. No content_block_stop, no
// message_delta, no message_stop, no error frame. curl exits 0 with empty
// stderr, so a client cannot tell a cut answer from a finished one.
//
// The fixture below ends mid-delta, in the middle of a JSON frame, which is
// what a connection closed at a deadline looks like on the wire.
const openRouterStreamCutMidDelta = `data: {"id":"gen-1","object":"chat.completion.chunk","created":1788474769,` +
	`"model":"deepseek/deepseek-v4-flash","choices":[{"index":0,` +
	`"delta":{"content":"Hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1788474769,` +
	`"model":"deepseek/deepseek-v4-flash","choices":[{"index":0,"delta":{"content":" wor`

func TestCutStreamIsTerminatedHonestly(t *testing.T) {
	body := runCutStream(t, llmprotocol.AnthropicMessagesV1)
	events := anthropicEventSequence(body)
	// What the client already holds must be closed before the turn ends.
	assertAnthropicEventsInOrder(t, body, "content_block_start", "content_block_delta",
		"content_block_stop", "error", "message_stop")
	if events[len(events)-1] != "message_stop" {
		t.Fatalf("a cut stream did not end on message_stop: %v", events)
	}
	if !strings.Contains(string(body), `"type":"error"`) {
		t.Fatalf("a cut stream carried no error frame:\n%s", body)
	}
}

// The Chat leg gets the equivalent: the error chunk it already emits, then the
// sentinel every Chat client waits for. A finish_reason of "length" would say
// the model reached a limit, which is not what happened -- the platform closed
// the connection -- so the error frame carries the outcome and [DONE] ends the
// stream.
func TestCutChatStreamEndsOnTheSentinel(t *testing.T) {
	body := runCutStream(t, llmprotocol.OpenAIChatV1)
	if !bytes.Contains(body, []byte(`"error"`)) {
		t.Fatalf("a cut Chat stream carried no error chunk:\n%s", body)
	}
	if !bytes.HasSuffix(bytes.TrimRight(body, "\n"), []byte("data: [DONE]")) {
		t.Fatalf("a cut Chat stream did not end on the sentinel:\n%s", body)
	}
}

// A stream that completed normally is untouched by any of this.
func TestCompletedStreamKeepsItsTerminalShape(t *testing.T) {
	body, _ := runProviderStream(t, openRouterStream, llmprotocol.AnthropicMessagesV1)
	events := anthropicEventSequence(body)
	if events[len(events)-1] != "message_stop" {
		t.Fatalf("a completed stream lost message_stop: %v", events)
	}
	for _, event := range events {
		if event == "error" {
			t.Fatalf("a completed stream gained an error frame: %v", events)
		}
	}
}

func runCutStream(t *testing.T, target llmprotocol.WireFormat) []byte {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, pushErr := stream.Push([]byte(openRouterStreamCutMidDelta))
	if pushErr != nil {
		t.Fatalf("push: %v", pushErr)
	}
	final, _, _, _ := stream.Finalize(context.DeadlineExceeded)
	return append(bytes.Join(frames, nil), bytes.Join(final, nil)...)
}
