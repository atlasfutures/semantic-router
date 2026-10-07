package protocolcodec

import (
	"bytes"
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Responses stream this Router wrote carries the model's bytes in its
// deltas and the settled object in function_call_arguments.done; a Router
// reading such a stream accepts the pair.
func TestAResponsesStreamWithSettledDoneArgumentsIsAccepted(t *testing.T) {
	if !settledArguments([]byte(`{"a":1,"a":2}`), `{"a":2}`, 32) {
		t.Fatal("the settled form of the streamed arguments was not recognised")
	}
	if settledArguments([]byte(`{"a":1,"a":2}`), `{"a":1}`, 32) {
		t.Fatal("arguments that are not the settled form were accepted")
	}
}

// The Router reads back a Responses stream it wrote for settled arguments:
// the deltas are the model's bytes, the done event the settled object.
func TestTheRoutersOwnSettledResponsesStreamReadsBack(t *testing.T) {
	body := anthropicToolStream([]string{`{"a":1,"a":2}`}, anthropicStop("tool_use"))
	out, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := out.Push([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	final, _, _, err := out.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	in, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := in.Push(bytes.Join(append(frames, final...), nil)); err != nil {
		t.Fatalf("the Router refused its own Responses stream: %v", err)
	}
}
