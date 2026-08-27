package fixture

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// sseStream carries a multi-byte rune so a chunk boundary can fall inside it.
const sseStream = "event: message_start\ndata: {\"text\":\"σπ\"}\n\nevent: message_stop\ndata: {}\n\n"

func TestReplayPreStreamError(t *testing.T) {
	script := scriptHeader + fmt.Sprintf(`expect: {method: POST, path: %s}
steps:
  - {kind: status, status: 429, headers: {content-type: application/json, retry-after: "3"}}
  - {kind: body, file: provider-response.json}
`, PathChatCompletions)

	const errorBody = `{"error":{"type":"rate_limit_error"}}`
	server, _ := start(t, script, map[string]string{"provider-response.json": errorBody})

	answer := post(t, server, PathChatCompletions, `{}`, nil)
	if answer.Status != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429", answer.Status)
	}
	if got := answer.Header.Get("retry-after"); got != "3" {
		t.Errorf("retry-after = %q, want %q", got, "3")
	}
	if string(answer.Body) != errorBody {
		t.Errorf("body = %q, want %q", answer.Body, errorBody)
	}
}

func TestReplayBufferedBodyIsByteExact(t *testing.T) {
	// Trailing whitespace and a non-UTF-8 byte survive only if the fixture replays
	// the file rather than re-encoding a parsed reading of it.
	const body = "{\"id\":\"resp_1\",\"raw\":\"\xff\"}\n\n"

	script := scriptHeader + fmt.Sprintf(`expect: {method: POST, path: %s}
steps:
  - {kind: status, status: 200}
  - {kind: body, file: provider-response.json}
`, PathResponses)
	server, _ := start(t, script, map[string]string{"provider-response.json": body})

	answer := post(t, server, PathResponses, `{}`, nil)
	if !bytes.Equal(answer.Body, []byte(body)) {
		t.Errorf("body = %q, want %q", answer.Body, body)
	}
}

func TestReplaySSESplitsAtArbitraryByteBoundaries(t *testing.T) {
	// Split one byte into the first multi-byte rune, so the boundary falls inside a
	// UTF-8 sequence as well as inside an event.
	chunk := bytes.IndexRune([]byte(sseStream), 'σ') + 1

	script := scriptHeader + fmt.Sprintf(`expect: {method: POST, path: %s}
steps:
  - {kind: status, status: 200, headers: {content-type: text/event-stream}}
  - {kind: sse, file: provider-response.sse, chunk_bytes: %d}
`, PathMessages, chunk)
	server, _ := start(t, script, map[string]string{"provider-response.sse": sseStream})

	chunks, terminated := chunkedBody(t, rawPost(t, server, PathMessages, `{}`))
	if !terminated {
		t.Fatalf("stream did not end with its terminating chunk")
	}

	wantChunks := (len(sseStream) + chunk - 1) / chunk
	if len(chunks) != wantChunks {
		t.Fatalf("wrote %d chunks, want %d", len(chunks), wantChunks)
	}
	for i, got := range chunks[:len(chunks)-1] {
		if len(got) != chunk {
			t.Errorf("chunk %d is %d bytes, want %d", i, len(got), chunk)
		}
	}
	if got := string(bytes.Join(chunks, nil)); got != sseStream {
		t.Errorf("reassembled stream = %q, want %q", got, sseStream)
	}
	if utf8.Valid(chunks[0]) {
		t.Errorf("chunk 0 = %q, want a split inside the UTF-8 sequence", chunks[0])
	}
}

func TestReplaySSEWritesWholeWhenChunkBytesIsZero(t *testing.T) {
	script := scriptHeader + fmt.Sprintf(`expect: {method: POST, path: %s}
steps:
  - {kind: status, status: 200, headers: {content-type: text/event-stream}}
  - {kind: sse, file: provider-response.sse}
`, PathMessages)
	server, _ := start(t, script, map[string]string{"provider-response.sse": sseStream})

	chunks, terminated := chunkedBody(t, rawPost(t, server, PathMessages, `{}`))
	if !terminated {
		t.Fatalf("stream did not end with its terminating chunk")
	}
	if len(chunks) != 1 || string(chunks[0]) != sseStream {
		t.Fatalf("chunks = %q, want the whole stream in one write", chunks)
	}
}

func TestReplayDelayPrecedesTheNextStep(t *testing.T) {
	const millis = 60

	script := scriptHeader + fmt.Sprintf(`expect: {method: POST, path: %s}
steps:
  - {kind: status, status: 200}
  - {kind: delay, millis: %d}
  - {kind: body, file: provider-response.json}
`, PathChatCompletions, millis)
	server, _ := start(t, script, map[string]string{"provider-response.json": `{"ok":true}`})

	started := time.Now()
	answer := post(t, server, PathChatCompletions, `{}`, nil)
	elapsed := time.Since(started)

	if elapsed < millis*time.Millisecond {
		t.Errorf("request took %s, want at least %dms", elapsed, millis)
	}
	if string(answer.Body) != `{"ok":true}` {
		t.Errorf("body = %q", answer.Body)
	}
}

func TestReplayDisconnectTruncatesTheStream(t *testing.T) {
	script := scriptHeader + fmt.Sprintf(`expect: {method: POST, path: %s}
steps:
  - {kind: status, status: 200, headers: {content-type: text/event-stream}}
  - {kind: sse, file: provider-response.sse}
  - {kind: disconnect}
`, PathMessages)
	const prefix = "event: message_start\ndata: {\"i\":0}\n\n"
	server, _ := start(t, script, map[string]string{"provider-response.sse": prefix})

	raw := rawPost(t, server, PathMessages, `{}`)
	if got := statusLine(raw); !strings.HasPrefix(got, "HTTP/1.1 200") {
		t.Fatalf("status line = %q, want a committed 200", got)
	}

	chunks, terminated := chunkedBody(t, raw)
	if terminated {
		t.Errorf("stream ended with a terminating chunk, want a truncated stream")
	}
	if got := string(bytes.Join(chunks, nil)); got != prefix {
		t.Errorf("delivered prefix = %q, want %q", got, prefix)
	}
}

func TestReplayDisconnectSurfacesAsAnUnexpectedEOF(t *testing.T) {
	script := scriptHeader + fmt.Sprintf(`expect: {method: POST, path: %s}
steps:
  - {kind: status, status: 200, headers: {content-type: text/event-stream}}
  - {kind: sse, file: provider-response.sse}
  - {kind: disconnect}
`, PathMessages)
	server, _ := start(t, script, map[string]string{"provider-response.sse": "data: {}\n\n"})

	resp, err := http.Post(server.URL()+PathMessages, "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("Post() error = %v", err)
	}
	defer resp.Body.Close()

	if _, err := io.ReadAll(resp.Body); err == nil {
		t.Errorf("reading a truncated stream succeeded, want an unexpected EOF")
	}
}

func TestReplayPostCommitErrorEvent(t *testing.T) {
	// The schema's post-commit fault family: a committed 200, a good prefix, then a
	// provider error event on the same stream.
	script := scriptHeader + fmt.Sprintf(`expect: {method: POST, path: %s}
steps:
  - {kind: status, status: 200, headers: {content-type: text/event-stream}}
  - {kind: sse, file: provider-response.sse}
  - {kind: sse, file: provider-error.sse}
`, PathMessages)
	const (
		prefix    = "event: message_start\ndata: {\"i\":0}\n\n"
		errEvent  = "event: error\ndata: {\"type\":\"overloaded_error\"}\n\n"
		wantWhole = prefix + errEvent
	)
	server, _ := start(t, script, map[string]string{
		"provider-response.sse": prefix,
		"provider-error.sse":    errEvent,
	})

	answer := post(t, server, PathMessages, `{}`, nil)
	if answer.Status != http.StatusOK {
		t.Fatalf("status = %d, want a committed 200", answer.Status)
	}
	if string(answer.Body) != wantWhole {
		t.Errorf("stream = %q, want %q", answer.Body, wantWhole)
	}
}
