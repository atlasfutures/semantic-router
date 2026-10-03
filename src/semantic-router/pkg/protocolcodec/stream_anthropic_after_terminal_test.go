package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const anthropicCompleteToolUseStream = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"read\",\"input\":{}}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"a\\\"}\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

// A complete message followed by frames that can add nothing to it -- an
// OpenAI-style [DONE] sentinel, a ping, a late usage message_delta, a repeated
// message_stop -- streams to the end, through every target, with the trailer
// recorded as a diagnostic (semantic-router#165: OpenRouter's Messages
// adapter on opus tool-use and kimi-k3 turns). Control: a content block after
// message_stop still fails, and the error names the event.
func TestAnthropicStreamToleratesFramesThatCannotAddToACompleteMessage(t *testing.T) {
	for name, trailer := range map[string]string{
		"done sentinel": "data: [DONE]\n\n",
		"ping":          "event: ping\ndata: {\"type\":\"ping\"}\n\n",
		"late usage":    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}\n\n",
		"repeated stop": "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	} {
		for _, target := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1} {
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
			if err != nil {
				t.Fatal(err)
			}
			frames, _, diagnostics, err := stream.Push([]byte(anthropicCompleteToolUseStream + trailer))
			if err != nil {
				t.Fatalf("%s -> %s: %v", name, target, err)
			}
			if _, _, _, err := stream.Finalize(nil); err != nil {
				t.Fatalf("%s -> %s: finalize: %v", name, target, err)
			}
			if !bytes.Contains(bytes.Join(frames, nil), []byte("read")) {
				t.Fatalf("%s -> %s: the tool call was not delivered", name, target)
			}
			found := false
			for _, diagnostic := range diagnostics {
				found = found || diagnostic.Field == "stream.after_message_stop"
			}
			if !found {
				t.Fatalf("%s -> %s: no diagnostic recorded the trailer", name, target)
			}
		}
	}
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = stream.Push([]byte(anthropicCompleteToolUseStream +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_event_after_terminal")
	if !strings.Contains(err.Error(), "content_block_start after message_stop") {
		t.Fatalf("the error does not name the event: %v", err)
	}
}
