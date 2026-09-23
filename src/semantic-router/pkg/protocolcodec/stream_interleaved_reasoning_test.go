package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// interleavedReasoningChunks is one upstream turn that reasons, says something
// the user can read, and then goes back to reasoning. The text delta carries
// real visible text, so this is not the separate case of an empty-string delta
// opening a block that says nothing.
var interleavedReasoningChunks = []string{
	"data: {\"id\":\"chatcmpl_interleaved\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"The user asks about the file.\"},\"finish_reason\":null}]}\n\n",
	"data: {\"id\":\"chatcmpl_interleaved\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Let me check.\"},\"finish_reason\":null}]}\n\n",
	"data: {\"id\":\"chatcmpl_interleaved\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"The file is small enough to read whole.\"},\"finish_reason\":null}]}\n\n",
	"data: {\"id\":\"chatcmpl_interleaved\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
}

// encodeInterleavedReasoning encodes that turn to Messages one chunk at a time
// and returns the bytes the client was handed before the first error, with
// that error. Pushing chunk by chunk is what a client sees: every frame
// returned before the error has already left the router.
func encodeInterleavedReasoning(t *testing.T) ([]byte, error) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	var delivered [][]byte
	for _, chunk := range interleavedReasoningChunks {
		frames, _, _, pushErr := stream.Push([]byte(chunk))
		delivered = append(delivered, frames...)
		if pushErr != nil {
			return bytes.Join(delivered, nil), pushErr
		}
	}
	frames, _, _, finalizeErr := stream.Finalize(nil)
	return bytes.Join(append(delivered, frames...), nil), finalizeErr
}

// TestReasoningResumingAfterTextContinuesInAFreshBlock is the ask.
//
// Interleaved reasoning is a shape upstream models produce: reason, answer,
// reason again. Messages cannot reopen a stopped content block, so the resumed
// reasoning should continue in a fresh block of the same kind rather than fail
// the response.
func TestReasoningResumingAfterTextContinuesInAFreshBlock(t *testing.T) {
	output, err := encodeInterleavedReasoning(t)
	if err != nil {
		t.Fatalf("encoding interleaved reasoning failed the stream: %v", err)
	}
	for _, event := range []string{"event: content_block_start", "event: content_block_stop"} {
		if got := strings.Count(string(output), event); got != 3 {
			t.Errorf("%q appears %d times, want 3 (thinking, text, thinking):\n%s", event, got, output)
		}
	}
	for _, event := range []string{"event: message_delta", "event: message_stop"} {
		if !strings.Contains(string(output), event) {
			t.Errorf("the stream never reached %q:\n%s", event, output)
		}
	}
	if !strings.Contains(string(output), `"index":2,"content_block":{"thinking":""`) {
		t.Errorf("the resumed reasoning did not open a third block of the thinking kind:\n%s", output)
	}
}

// TestReasoningResumingAfterTextIsRefusedToday pins the present behaviour so a
// change to it shows up in the diff. The refusal is raised at
// stream_anthropic.go:586, after two whole content blocks have already been
// written to the client, which is the harm: the response is failed in flight,
// with part of it already read.
func TestReasoningResumingAfterTextIsRefusedToday(t *testing.T) {
	output, err := encodeInterleavedReasoning(t)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "anthropic_content_interleaving")
	for _, delivered := range []string{
		`"index":0,"content_block":{"thinking":""`,
		`{"type":"thinking_delta","thinking":"The user asks about the file."}`,
		`{"type":"content_block_stop","index":0}`,
		`"index":1,"content_block":{"text":""`,
		`{"type":"text_delta","text":"Let me check."}`,
	} {
		if !strings.Contains(string(output), delivered) {
			t.Errorf("%s was not delivered before the refusal:\n%s", delivered, output)
		}
	}
	if strings.Contains(string(output), "event: message_stop") {
		t.Errorf("the recorded behaviour has changed: the stream reached message_stop:\n%s", output)
	}
}
