package execution

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"strings"
)

// streamPeekLimit bounds what the core holds back while it looks for a
// streamed refusal; a longer opening passes through unread.
const streamPeekLimit = 64 << 10

// peekStreamContextRejection reads a local runtime's event stream up to its
// first event that is not a stream opening (a role-only Chat chunk, Messages
// message_start or ping, Responses response.created or response.in_progress)
// and reports whether that event refuses the prompt as over the context
// window. Some runtimes (TensorFold, for one) answer 200, open the stream and
// only then send the context-length error, before any output. The returned
// body replays every byte read, so a stream that is not refused is relayed
// unchanged.
func peekStreamContextRejection(body io.ReadCloser) (bool, io.ReadCloser) {
	reader := bufio.NewReader(body)
	var read bytes.Buffer
	var event, data strings.Builder
	replay := func() io.ReadCloser {
		return struct {
			io.Reader
			io.Closer
		}{io.MultiReader(bytes.NewReader(read.Bytes()), reader), body}
	}
	for read.Len() < streamPeekLimit {
		line, err := reader.ReadString('\n')
		read.WriteString(line)
		trimmed := strings.TrimRight(line, "\r\n")
		switch {
		case trimmed == "" && err == nil:
			if data.Len() == 0 && event.Len() == 0 {
				continue
			}
			refused, opening := classifyStreamEvent(event.String(), data.String())
			if !opening {
				return refused, replay()
			}
			event.Reset()
			data.Reset()
			continue
		case strings.HasPrefix(trimmed, ":"):
			// A comment (a keepalive) is the runtime working: no refusal.
			return false, replay()
		case strings.HasPrefix(trimmed, "event:"):
			event.WriteString(strings.TrimSpace(strings.TrimPrefix(trimmed, "event:")))
		case strings.HasPrefix(trimmed, "data:"):
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(strings.TrimPrefix(strings.TrimPrefix(trimmed, "data:"), " "))
		}
		if err != nil {
			return false, replay()
		}
	}
	return false, replay()
}

// classifyStreamEvent reports whether one event refuses the prompt as over the
// context window, and whether it only opens the stream (so the next event
// decides).
func classifyStreamEvent(name, data string) (refused, opening bool) {
	var payload map[string]json.RawMessage
	if json.Unmarshal([]byte(data), &payload) != nil {
		return false, false
	}
	var kind string
	_ = json.Unmarshal(payload["type"], &kind)
	if name == "" {
		name = kind
	}
	switch name {
	case "message_start", "ping", "response.created", "response.in_progress":
		return false, true
	case "response.failed":
		var response struct {
			Error json.RawMessage `json:"error"`
		}
		if json.Unmarshal(payload["response"], &response) != nil || len(response.Error) == 0 {
			return false, false
		}
		envelope, err := json.Marshal(map[string]json.RawMessage{"error": response.Error})
		return err == nil && runtimeContextRefusal(400, envelope), false
	}
	if payload["error"] != nil {
		return runtimeContextRefusal(400, []byte(data)), false
	}
	return false, roleOnlyChatChunk([]byte(data))
}

// roleOnlyChatChunk reports a Chat chunk that only opens the assistant
// message: a role and no content, tool call or finish.
func roleOnlyChatChunk(data []byte) bool {
	var chunk struct {
		Choices []struct {
			Delta        map[string]json.RawMessage `json:"delta"`
			FinishReason *string                    `json:"finish_reason"`
		} `json:"choices"`
	}
	if json.Unmarshal(data, &chunk) != nil || len(chunk.Choices) != 1 || chunk.Choices[0].FinishReason != nil {
		return false
	}
	for key, value := range chunk.Choices[0].Delta {
		if key == "role" {
			continue
		}
		if key == "content" && (string(value) == `""` || string(value) == "null") {
			continue
		}
		return false
	}
	return chunk.Choices[0].Delta["role"] != nil
}
