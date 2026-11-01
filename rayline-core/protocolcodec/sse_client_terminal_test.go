package protocolcodec

import (
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

func TestClientTerminalFrameAtParsesSSEEvents(t *testing.T) {
	const delta = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n"
	const chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
	messages, chat := llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1
	cases := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
		want   int
	}{
		{"messages stop", messages, delta + "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n", len(delta)},
		{"messages stop, data before event", messages, delta + "data: {\"type\":\"message_stop\"}\nevent: message_stop\n\n", len(delta)},
		{"messages stop, no space after colon", messages, delta + "event:message_stop\ndata:{\"type\":\"message_stop\"}\n\n", len(delta)},
		{"messages stop, CRLF", messages, "event: ping\r\ndata: {}\r\n\r\nevent: message_stop\r\ndata: {}\r\n\r\n", len("event: ping\r\ndata: {}\r\n\r\n")},
		{"messages stop, CR", messages, "event: ping\rdata: {}\r\revent: message_stop\rdata: {}\r\r", len("event: ping\rdata: {}\r\r")},
		{"messages stop after a comment", messages, ": keepalive\n\nevent: message_stop\ndata: {}\n\n", len(": keepalive\n\n")},
		{"messages stop, unfinished last event", messages, delta + "event: message_stop\ndata: {}", len(delta)},
		{"messages error", messages, delta + "data: {\"type\":\"error\"}\nevent:error\n\n", len(delta)},
		{"messages stop named in data only", messages, delta + "data: {\"type\":\"message_stop\"}\n\n", -1},
		{"messages stop as a data value", messages, "event: content_block_delta\ndata: event: message_stop\n\n", -1},
		{"messages delta only", messages, delta, -1},
		{"chat done", chat, chunk + "data: [DONE]\n\n", len(chunk)},
		{"chat done, no space", chat, chunk + "data:[DONE]\n\n", len(chunk)},
		{"chat done, CRLF", chat, "data: {}\r\n\r\ndata: [DONE]\r\n\r\n", len("data: {}\r\n\r\n")},
		{"chat error, not the first key", chat, chunk + "data: {\"id\":\"x\",\"error\":{\"message\":\"boom\"}}\n\n", len(chunk)},
		{"chat error, no space", chat, chunk + "data:{\"error\":{\"message\":\"boom\"}}\n\n", len(chunk)},
		{"chat error after an event field", chat, chunk + "event: error\ndata: {\"error\":{\"message\":\"boom\"}}\n\n", len(chunk)},
		{"chat null error", chat, "data: {\"error\":null,\"choices\":[]}\n\n" + chunk, -1},
		{"chat error in a comment", chat, ": data: [DONE]\n\n" + chunk, -1},
		{"chat chunk only", chat, chunk, -1},
		{"responses", llmprotocol.OpenAIResponsesV1, "event: response.completed\ndata: {}\n\n", -1},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			if got := ClientTerminalFrameAt(testCase.format, []byte(testCase.body)); got != testCase.want {
				t.Fatalf("ClientTerminalFrameAt = %d, want %d", got, testCase.want)
			}
		})
	}
}
