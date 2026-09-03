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
	// What the client already holds is closed, then the failure is named.
	assertAnthropicEventsInOrder(t, body, "content_block_start", "content_block_delta",
		"content_block_stop", "error")
	if events[len(events)-1] != "error" {
		t.Fatalf("a cut stream did not end on the failure: %v", events)
	}
	if !strings.Contains(string(body), `"type":"error"`) {
		t.Fatalf("a cut stream carried no error frame:\n%s", body)
	}
	// message_stop is the terminal of a message that finished. This one did
	// not, and a client that reads message_stop as "the answer is complete"
	// would be told a whole answer arrived. Anthropic documents message_stop
	// only as the last event of the normal flow and never says it follows an
	// error, so the Router does not send one.
	for _, event := range events {
		if event == "message_stop" {
			t.Fatalf("a cut stream published a success terminal: %v", events)
		}
	}
}

// The Chat leg ends on its error chunk for the same reason. [DONE] is the
// sentinel that ends a stream that ran to the end; a finish_reason of "length"
// would claim the model reached a limit it never reached. Neither is true of a
// connection the platform closed, so the error chunk carries the outcome and
// nothing after it claims the turn finished.
func TestCutChatStreamEndsOnItsError(t *testing.T) {
	body := runCutStream(t, llmprotocol.OpenAIChatV1)
	if !bytes.Contains(body, []byte(`"error"`)) {
		t.Fatalf("a cut Chat stream carried no error chunk:\n%s", body)
	}
	if bytes.Contains(body, []byte("data: [DONE]")) {
		t.Fatalf("a cut Chat stream published a success sentinel:\n%s", body)
	}
	if bytes.Contains(body, []byte(`"finish_reason":"length"`)) {
		t.Fatalf("a cut Chat stream claimed the model reached a limit:\n%s", body)
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
