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
// adapter on opus tool-use and kimi-k3 turns). A late message_delta passes
// only when it restates the usage and stop already published. Controls: a
// content block, new counts, a charge or another stop after message_stop
// still fail.
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
	// The accounting the client was given at message_stop stands: a
	// restating late delta leaves the published usage as it was.
	response, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.AnthropicMessagesV1,
		[]byte(anthropicCompleteToolUseStream+"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}\n\n"),
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil || tokenValue(response.Usage.OutputTotal) != 9 {
		t.Fatalf("usage after a restating late delta: %+v, %v", response.Usage, err)
	}
	for name, trailer := range map[string]string{
		"content block": "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"new counts":    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":12}}\n\n",
		"a charge":      "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":null},\"usage\":{\"output_tokens\":9,\"cost\":0.0042}}\n\n",
		"another stop":  "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
	} {
		stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1,
			llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
		if err != nil {
			t.Fatal(err)
		}
		_, _, _, err = stream.Push([]byte(anthropicCompleteToolUseStream + trailer))
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_event_after_terminal")
		if !strings.Contains(err.Error(), "after message_stop") {
			t.Fatalf("%s: the error does not say where: %v", name, err)
		}
	}
}
