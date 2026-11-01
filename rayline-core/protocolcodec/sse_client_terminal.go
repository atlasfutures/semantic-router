package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// ClientTerminalFrameAt is where, in body, the first SSE event starts that
// ends a client's stream of format, or -1 when body has none. body is read as
// whole SSE events, as the public stream filters write them; an unfinished
// last event is read as it stands.
//
// The terminal events are those a client's SDK stops on:
//   - Messages: an event named message_stop or error;
//   - Chat: data [DONE], or a data object with a non-null error.
//
// Events are parsed as SSE, so field order, a missing space after the colon,
// and LF, CRLF or CR line endings do not matter. Other formats return -1: a
// Responses terminal carries its usage and goes out with the body's end.
func ClientTerminalFrameAt(format llmprotocol.WireFormat, body []byte) int {
	if format != llmprotocol.AnthropicMessagesV1 && format != llmprotocol.OpenAIChatV1 {
		return -1
	}
	for offset := 0; offset < len(body); {
		end, complete := completeSSEFrame(body[offset:])
		if !complete {
			end = len(body) - offset
		}
		if clientTerminalEvent(format, parseClientSSEEvent(body[offset:offset+end], offset == 0)) {
			return offset
		}
		offset += end
	}
	return -1
}

func parseClientSSEEvent(frame []byte, first bool) sseFrame {
	var event sseFrame
	normalized, err := normalizeSSEFrame(frame, first)
	if err != nil {
		return event
	}
	for _, line := range bytes.Split(normalized, []byte("\n")) {
		parseSSELine(&event, line)
	}
	return event
}

func clientTerminalEvent(format llmprotocol.WireFormat, event sseFrame) bool {
	if format == llmprotocol.AnthropicMessagesV1 {
		return event.Event == "message_stop" || event.Event == "error"
	}
	if !event.HasData {
		return false
	}
	data := bytes.TrimSpace(event.Data)
	if bytes.HasPrefix(data, []byte("[DONE]")) {
		return true
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(data, &object) != nil {
		return false
	}
	failure, found := object["error"]
	return found && !bytes.Equal(bytes.TrimSpace(failure), []byte("null"))
}
