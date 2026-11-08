package execution

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"sync"
	"time"
)

const (
	// streamPeekLimit bounds what the core holds back while it looks for a
	// streamed refusal; a longer opening passes through.
	streamPeekLimit = 64 << 10
	// streamPeekWait bounds how long it holds a stream back: a runtime that
	// refuses does so as soon as the stream opens, before any prefill.
	streamPeekWait = 2 * time.Second
)

// peekStreamContextRejection reads a local runtime's event stream up to its
// first event that is not a stream opening (a role-only Chat chunk, Messages
// message_start or ping, Responses response.created or response.in_progress)
// and reports whether that event refuses the prompt as over the context
// window. Some runtimes (TensorFold, for one) answer 200, open the stream and
// only then send the context-length error, before any output. It stops
// looking after streamPeekLimit bytes or streamPeekWait. The returned body
// replays every byte read, then the rest of the stream, so a stream that is
// not refused is relayed unchanged.
func peekStreamContextRejection(body io.ReadCloser) (bool, io.ReadCloser) {
	stream := newChunkedStream(body)
	var read []byte
	deadline := time.NewTimer(streamPeekWait)
	defer deadline.Stop()
	for len(read) < streamPeekLimit {
		select {
		case chunk, open := <-stream.chunks:
			if !open {
				refused, _ := classifyStreamPrefix(read, true)
				return refused, stream.replay(read)
			}
			read = append(read, chunk...)
			if refused, decided := classifyStreamPrefix(read, false); decided {
				return refused, stream.replay(read)
			}
		case <-deadline.C:
			return false, stream.replay(read)
		}
	}
	return false, stream.replay(read)
}

// chunkedStream reads its body on one goroutine, so a peek can stop waiting
// while a read is pending and hand that read on to the relay.
type chunkedStream struct {
	body    io.ReadCloser
	chunks  chan []byte
	done    chan struct{}
	pending []byte
	once    sync.Once
	// err is the body's last read error, set before chunks closes.
	err error
}

func newChunkedStream(body io.ReadCloser) *chunkedStream {
	stream := &chunkedStream{body: body, chunks: make(chan []byte), done: make(chan struct{})}
	go func() {
		defer close(stream.chunks)
		for {
			buffer := make([]byte, 4096)
			n, err := body.Read(buffer)
			if n > 0 {
				select {
				case stream.chunks <- buffer[:n]:
				case <-stream.done:
					return
				}
			}
			if err != nil {
				stream.err = err
				return
			}
		}
	}()
	return stream
}

// replay returns a body that yields read, then the rest of the stream.
func (stream *chunkedStream) replay(read []byte) io.ReadCloser {
	stream.pending = read
	return stream
}

func (stream *chunkedStream) Read(p []byte) (int, error) {
	for len(stream.pending) == 0 {
		chunk, open := <-stream.chunks
		if !open {
			if stream.err != nil {
				return 0, stream.err
			}
			return 0, io.EOF
		}
		stream.pending = chunk
	}
	n := copy(p, stream.pending)
	stream.pending = stream.pending[n:]
	return n, nil
}

func (stream *chunkedStream) Close() error {
	var err error
	stream.once.Do(func() {
		close(stream.done)
		err = stream.body.Close()
	})
	return err
}

// classifyStreamPrefix walks the complete events in a stream prefix (at EOF,
// the last unterminated one too). decided is false while every event so far
// only opens the stream.
func classifyStreamPrefix(prefix []byte, eof bool) (refused, decided bool) {
	text := strings.TrimPrefix(string(prefix), "\ufeff")
	var event, data strings.Builder
	hasData := false
	for len(text) > 0 {
		// Each read re-parses the whole prefix, so a CR whose LF arrives in
		// the next read ends the same line either way.
		end := strings.IndexAny(text, "\r\n")
		if end < 0 {
			if !eof {
				break
			}
			end = len(text)
		}
		line := text[:end]
		text = text[end:]
		if strings.HasPrefix(text, "\r\n") {
			text = text[2:]
		} else if len(text) > 0 {
			text = text[1:]
		}
		if line == "" {
			if hasData {
				refused, opening := classifyStreamEvent(event.String(), data.String())
				if !opening {
					return refused, true
				}
			}
			event.Reset()
			data.Reset()
			hasData = false
			continue
		}
		field, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			event.Reset()
			event.WriteString(value)
		case "data":
			if hasData {
				data.WriteByte('\n')
			}
			data.WriteString(value)
			hasData = true
		}
	}
	if eof && hasData {
		refused, _ := classifyStreamEvent(event.String(), data.String())
		return refused, true
	}
	return false, eof
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
		_ = json.Unmarshal(payload["response"], &response)
		return errorRefusesContext(response.Error), false
	}
	if inner := payload["error"]; len(inner) > 0 && string(inner) != "null" {
		return errorRefusesContext(inner), false
	}
	if name == "error" {
		// A flat error event (Responses): code, type and message at the top.
		flat, err := json.Marshal(map[string]json.RawMessage{"code": payload["code"], "type": payload["type"], "message": payload["message"]})
		return err == nil && errorRefusesContext(flat), false
	}
	return false, roleOnlyChatChunk([]byte(data))
}

// errorRefusesContext matches an error object (or string) alone, never the
// output beside it.
func errorRefusesContext(inner json.RawMessage) bool {
	if len(inner) == 0 || string(inner) == "null" {
		return false
	}
	envelope, err := json.Marshal(map[string]json.RawMessage{"error": inner})
	return err == nil && runtimeContextRefusal(400, envelope)
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
		if key == "content" && (bytes.Equal(value, []byte(`""`)) || bytes.Equal(value, []byte("null"))) {
			continue
		}
		return false
	}
	return chunk.Choices[0].Delta["role"] != nil
}
