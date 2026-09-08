package protocolcodec

import (
	"bytes"
	"context"
	"io"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// providerStreamCutMidAnswer is one complete provider chunk carrying a text
// delta. The connection is cut after it, so no finish_reason and no [DONE]
// ever arrive from the provider.
const providerStreamCutMidAnswer = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model",` +
	`"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

`

const providerStreamCompleted = providerStreamCutMidAnswer +
	`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"provider-model",` +
	`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`

// encodeStream translates a provider stream to target and finalizes it with
// cut, the way pkg/extproc/processor_res_semantic_stream.go:96 finalizes a
// response stream whose read failed.
func encodeStream(t *testing.T, target llmprotocol.WireFormat, provider string, cut error) string {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, pushErr := stream.Push([]byte(provider))
	if pushErr != nil {
		t.Fatalf("push: %v", pushErr)
	}
	final, _, _, _ := stream.Finalize(cut)
	return string(bytes.Join(append(frames, final...), nil))
}

// TestCutStreamEndsWithATerminalFrame is the ask.
//
// A client must be able to tell a cut answer from a finished one from the
// frame sequence itself. A cut stream stops inside the content block the
// client is still holding open, so the two endings are the same on the wire
// apart from an error frame no Anthropic client treats as terminal.
func TestCutStreamEndsWithATerminalFrame(t *testing.T) {
	t.Run("the Anthropic leg closes its block and ends the message", func(t *testing.T) {
		body := encodeStream(t, llmprotocol.AnthropicMessagesV1, providerStreamCutMidAnswer, io.ErrUnexpectedEOF)
		for _, want := range []string{"content_block_stop", `"type":"error"`, "message_stop"} {
			if !strings.Contains(body, want) {
				t.Errorf("a cut Anthropic stream carried no %s:\n%s", want, body)
			}
		}
	})
	t.Run("the Chat leg ends on the sentinel", func(t *testing.T) {
		body := encodeStream(t, llmprotocol.OpenAIChatV1, providerStreamCutMidAnswer, io.ErrUnexpectedEOF)
		if !strings.HasSuffix(strings.TrimRight(body, "\n"), "data: [DONE]") {
			t.Errorf("a cut Chat stream did not end on the sentinel:\n%s", body)
		}
	})
	t.Run("a stream that finished is untouched", func(t *testing.T) {
		body := encodeStream(t, llmprotocol.AnthropicMessagesV1, providerStreamCompleted, nil)
		if !strings.Contains(body, "message_stop") {
			t.Errorf("a completed stream lost message_stop:\n%s", body)
		}
		if strings.Contains(body, `"type":"error"`) {
			t.Errorf("a completed stream gained an error frame:\n%s", body)
		}
	})
}

// TestCutStreamEndsOpenToday pins the present behaviour so a change to it
// shows up in the diff rather than only in the test above. A cut stream ends
// on the error frame the codec already emits: the block it opened is never
// closed, and the Chat leg never reaches the sentinel its clients wait for.
func TestCutStreamEndsOpenToday(t *testing.T) {
	anthropic := encodeStream(t, llmprotocol.AnthropicMessagesV1, providerStreamCutMidAnswer, io.ErrUnexpectedEOF)
	if !strings.Contains(anthropic, "content_block_start") {
		t.Fatalf("the fixture no longer opens a content block:\n%s", anthropic)
	}
	for _, absent := range []string{"content_block_stop", "message_stop"} {
		if strings.Contains(anthropic, absent) {
			t.Errorf("the recorded behaviour has changed: a cut Anthropic stream now carries %s", absent)
		}
	}
	chat := encodeStream(t, llmprotocol.OpenAIChatV1, providerStreamCutMidAnswer, io.ErrUnexpectedEOF)
	if strings.Contains(chat, "[DONE]") {
		t.Error("the recorded behaviour has changed: a cut Chat stream now ends on the sentinel")
	}
}
