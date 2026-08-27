package fixture

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
)

// scriptHeader is the schema line every test script needs.
const scriptHeader = "schema_version: protocol-conformance-replay-v1\n"

// start brings up a fixture loaded with script, whose artifacts are written into a
// fresh case directory. It returns the running server and that directory.
func start(t *testing.T, script string, artifacts map[string]string) (*Server, string) {
	t.Helper()

	dir := t.TempDir()
	for name, content := range artifacts {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
			t.Fatalf("write artifact %s: %v", name, err)
		}
	}

	server, err := Start("127.0.0.1:0")
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	t.Cleanup(func() {
		// A client connection that was dialed but never used is neither active nor
		// idle to the server, and Shutdown waits five seconds on it. Retiring the
		// client's pooled connections first keeps the teardown quick.
		http.DefaultClient.CloseIdleConnections()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			t.Errorf("Shutdown() error = %v", err)
		}
	})

	if script != "" {
		parsed, err := conformance.ParseReplayScript([]byte(script), dir)
		if err != nil {
			t.Fatalf("ParseReplayScript() error = %v", err)
		}
		server.Reset(parsed, dir)
	}
	return server, dir
}

// reply is one complete provider response, with the body already read and the
// connection already released, so no test has to manage either.
type reply struct {
	Status int
	Header http.Header
	Body   []byte
}

// post sends one provider request through the standard client.
func post(t *testing.T, server *Server, path, body string, headers map[string]string) reply {
	t.Helper()

	req, err := http.NewRequest(http.MethodPost, server.URL()+path, strings.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest() error = %v", err)
	}
	for name, value := range headers {
		req.Header.Set(name, value)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	return reply{Status: resp.StatusCode, Header: resp.Header, Body: raw}
}

// rawPost speaks HTTP/1.1 on a bare socket and returns every byte the server sent
// back, chunk framing included. Only a raw socket can show where a chunk boundary
// fell or that a stream ended without its terminating chunk.
func rawPost(t *testing.T, server *Server, path, body string) []byte {
	t.Helper()

	address := strings.TrimPrefix(server.URL(), "http://")
	conn, err := net.DialTimeout("tcp", address, 5*time.Second)
	if err != nil {
		t.Fatalf("dial %s: %v", address, err)
	}
	defer conn.Close()

	request := fmt.Sprintf(
		"POST %s HTTP/1.1\r\nHost: %s\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s",
		path, address, len(body), body,
	)
	if deadlineErr := conn.SetDeadline(time.Now().Add(10 * time.Second)); deadlineErr != nil {
		t.Fatalf("set deadline: %v", deadlineErr)
	}
	if _, writeErr := conn.Write([]byte(request)); writeErr != nil {
		t.Fatalf("write request: %v", writeErr)
	}

	raw, err := io.ReadAll(conn)
	if err != nil {
		t.Fatalf("read raw response: %v", err)
	}
	return raw
}

// chunkedBody splits a raw HTTP/1.1 response into its transfer-encoding chunks and
// reports whether the terminating zero-length chunk arrived.
func chunkedBody(t *testing.T, raw []byte) (chunks [][]byte, terminated bool) {
	t.Helper()

	_, body, found := bytes.Cut(raw, []byte("\r\n\r\n"))
	if !found {
		t.Fatalf("response has no header terminator: %q", raw)
	}

	reader := bufio.NewReader(bytes.NewReader(body))
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return chunks, false
		}
		size, err := strconv.ParseInt(strings.TrimSpace(line), 16, 32)
		if err != nil {
			t.Fatalf("chunk size %q: %v", line, err)
		}
		if size == 0 {
			return chunks, true
		}

		payload := make([]byte, size)
		if _, err := io.ReadFull(reader, payload); err != nil {
			return append(chunks, payload), false
		}
		chunks = append(chunks, payload)
		if _, err := reader.Discard(2); err != nil { // trailing CRLF
			return chunks, false
		}
	}
}

// statusLine is the first line of a raw response, without its CRLF.
func statusLine(raw []byte) string {
	line, _, _ := bytes.Cut(raw, []byte("\r\n"))
	return string(line)
}
