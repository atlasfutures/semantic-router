package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
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
		"content block":       "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"new counts":          "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":12}}\n\n",
		"a charge":            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":null},\"usage\":{\"output_tokens\":9,\"cost\":0.0042}}\n\n",
		"another stop":        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
		"a stop sequence":     "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":\"END\"},\"usage\":{\"output_tokens\":9}}\n\n",
		"a ping with usage":   "event: ping\ndata: {\"type\":\"ping\",\"usage\":{\"output_tokens\":3}}\n\n",
		"a stop with a delta": "event: message_stop\ndata: {\"type\":\"message_stop\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
		"usage_source":        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9},\"usage_source\":\"unknown\"}\n\n",
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

// A terminal message_delta whose usage a Router marked unknown, repeated
// exactly after message_stop, is a restatement like any other.
func TestAnthropicStreamToleratesARepeatedUnknownUsageDelta(t *testing.T) {
	unknown := "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":0},\"usage_source\":\"unknown\"}\n\n"
	stream := strings.Replace(anthropicCompleteToolUseStream,
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}\n\n", unknown, 1)
	if stream == anthropicCompleteToolUseStream {
		t.Fatal("the fixture did not take the unknown-usage delta")
	}
	if _, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.AnthropicMessagesV1, []byte(stream+unknown),
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"}); err != nil {
		t.Fatalf("a repeated unknown-usage delta failed the stream: %v", err)
	}
}

// Counts that differ only beyond float64 precision are still different.
func TestAnthropicLateDeltaComparesCountsExactly(t *testing.T) {
	decoder := &anthropicStreamDecoder{terminalDelta: json.RawMessage(`{"type":"message_delta","usage":{"output_tokens":9007199254740992}}`)}
	if decoder.lateDeltaRestates([]byte(`{"type":"message_delta","usage":{"output_tokens":9007199254740993}}`)) {
		t.Fatal("two counts beyond float64 precision compared equal")
	}
	if !decoder.lateDeltaRestates([]byte(`{"type":"message_delta","usage":{"output_tokens":9007199254740992}}`)) {
		t.Fatal("the same count compared unequal")
	}
}

// A connection dropping inside the trailing sentinel leaves the finished
// answer its terminal.
//
// The sentinel was matched only complete, so a read that stopped partway
// through one of these trailers went to the JSON decoder and the turn ended on
// stream_event_after_terminal -- the client got a bare error where its
// finish_reason and [DONE] belonged, with nothing to close the stream on,
// after the model had already finished (codex round 13). The sentinel whole
// adds nothing to a complete message, so neither could the bytes that never
// came have.
func TestAnthropicStreamToleratesASentinelTheConnectionDroppedInside(t *testing.T) {
	for _, tail := range []string{"data: [", "data: [D", "data: [DO", "data: [DON", "data: [DONE"} {
		for target, closing := range map[llmprotocol.WireFormat]string{
			llmprotocol.AnthropicMessagesV1: "event: message_stop",
			llmprotocol.OpenAIResponsesV1:   "event: response.completed",
			llmprotocol.OpenAIChatV1:        "data: [DONE]",
		} {
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
			requireNoError(t, err)
			served, _, _, err := stream.Push([]byte(anthropicCompleteToolUseStream + tail))
			requireNoError(t, err, "%q -> %s: refused before the cut", tail, target)
			terminal, _, _, err := stream.Finalize(nil)
			requireNoError(t, err, "%q -> %s: a cut sentinel failed the stream", tail, target)

			body := string(joinFrames(append(served, terminal...)))
			wantNotContains(t, body, "stream_event_after_terminal",
				"%q -> %s: a dropped connection was called a protocol violation", tail, target)
			wantContains(t, body, "toolu_1", "%q -> %s: the finished answer did not reach the client", tail, target)
			wantContains(t, body, closing, "%q -> %s: the stream had nothing to close on", tail, target)
		}
	}
}

// A turn the Router then cut keeps the stop and the count the complete message
// had already stated: the cut landed in a trailer, after the answer.
func TestACutInsideATrailingSentinelBillsWhatTheMessageStated(t *testing.T) {
	requested := true
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{
			Context: context.Background(), PublicModel: "public-model",
			Options: llmprotocol.StreamOptions{IncludeUsage: &requested},
		})
	requireNoError(t, err)
	served, _, _, err := stream.Push([]byte(anthropicCompleteToolUseStream + "data: [DON"))
	requireNoError(t, err)
	stream.SetTruncationUsage(estimatedTruncationUsage(), llmprotocol.UsageSourceStreamEstimate)
	stream.SetTruncationCause(llmprotocol.TruncationTransport)
	terminal, _, _, err := stream.Finalize(llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable, "standard_resend_body_cut", "the body was cut", nil))
	requireNoError(t, err)

	body := string(joinFrames(append(served, terminal...)))
	wantContains(t, body, `"finish_reason":"tool_calls"`, "the stop the model reached was overwritten")
	// The message settled its own count at message_stop; the Router's estimate
	// of 268 would charge for output the provider never billed.
	wantContains(t, body, `"completion_tokens":9`, "the message's own settlement was lost")
	wantNotContains(t, body, "268", "the Router's estimate replaced a count the provider had settled")
	wantContains(t, body, "data: [DONE]", "the stream had nothing to close on")
}
