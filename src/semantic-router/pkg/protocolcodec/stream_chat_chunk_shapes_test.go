package protocolcodec

import (
	"bytes"
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A provider closes a Chat stream with two chunks, not one: a chunk that
// carries finish_reason with an empty delta, then a second chunk repeating that
// finish_reason with another empty delta and the usage of the turn.
const (
	chatShapeTextChunk = "data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\"," +
		"\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"
	chatShapeStopChunk = "data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\"," +
		"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"stop\"}]}\n\n"
	chatShapeRepeatedStopChunk = "data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\"," +
		"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"stop\"}]," +
		"\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"total_tokens\":18}}\n\n"
	chatShapeDone = "data: [DONE]\n\n"
)

// chatShapeReasoningChunk is one chunk of a thinking turn from a provider that
// always sends content, filling it with the empty string when it says nothing.
func chatShapeReasoningChunk(text string) string {
	return "data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\"," +
		"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning_content\":\"" + text + "\"},\"finish_reason\":null}]}\n\n"
}

// pushChatToMessages runs the chunks through one Chat-to-Messages stream and
// returns the bytes a client would have received and the first error raised.
func pushChatToMessages(t *testing.T, chunks ...string) ([]byte, error) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1,
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	var frames [][]byte
	var streamErr error
	for _, chunk := range chunks {
		pushed, _, _, pushErr := stream.Push([]byte(chunk))
		frames = append(frames, pushed...)
		if pushErr != nil {
			streamErr = pushErr
			break
		}
	}
	final, _, _, finalErr := stream.Finalize(nil)
	frames = append(frames, final...)
	if streamErr == nil {
		streamErr = finalErr
	}
	return bytes.Join(frames, nil), streamErr
}

// TestRepeatedStopChunkIsANoOp is the ask. The repeated stop chunk names a
// choice the previous chunk closed, and decodeChoice (stream_chat.go:272) has
// no guard for that, so its empty content string reaches the lifecycle check at
// stream_state.go:340 and the stream is refused. An empty delta on a completed
// choice should change nothing while its finish_reason and usage are ingested.
func TestRepeatedStopChunkIsANoOp(t *testing.T) {
	output, err := pushChatToMessages(t, chatShapeTextChunk, chatShapeStopChunk, chatShapeRepeatedStopChunk, chatShapeDone)
	if err != nil {
		t.Fatalf("the repeated stop chunk failed the stream: %v", err)
	}
	if !bytes.Contains(output, []byte(`"output_tokens":7`)) ||
		!bytes.Contains(output, []byte(`"type":"message_stop"`)) {
		t.Errorf("the terminal frame lost the usage and the stop of the turn: %s", output)
	}
}

// TestEmptyStringDeltaDoesNotFragmentAThinkingBlock is the rest of the ask.
// chatChoiceNeedsItem (stream_chat.go:310) and decodeContentDelta
// (stream_chat.go:326) test the delta fields for nil, not for content, so an
// empty content string opens a text block that receives no text. Three
// reasoning chunks are one thinking block, opened by text that says something.
func TestEmptyStringDeltaDoesNotFragmentAThinkingBlock(t *testing.T) {
	output, err := pushChatToMessages(t,
		chatShapeReasoningChunk("a"), chatShapeReasoningChunk("b"), chatShapeReasoningChunk("c"),
		chatShapeStopChunk, chatShapeDone,
	)
	started := bytes.Count(output, []byte(`"type":"content_block_start"`))
	if err != nil || started != 1 || !bytes.Contains(output, []byte(`"type":"thinking"`)) {
		t.Fatalf("want one thinking block, got %d block starts and error %v: %s", started, err, output)
	}
}

// TestRepeatedStopChunkIsRefusedToday pins the present behaviour so a change to
// it shows up in the diff rather than only in the tests above.
func TestRepeatedStopChunkIsRefusedToday(t *testing.T) {
	_, err := pushChatToMessages(t, chatShapeTextChunk, chatShapeStopChunk, chatShapeRepeatedStopChunk, chatShapeDone)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_item_lifecycle")
}
