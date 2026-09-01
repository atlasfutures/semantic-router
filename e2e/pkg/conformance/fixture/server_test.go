package fixture

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"
)

// bufferedScript answers one POST on path with a buffered JSON body.
func bufferedScript(path string) string {
	return scriptHeader + fmt.Sprintf(`expect: {method: POST, path: %s}
steps:
  - {kind: status, status: 200, headers: {content-type: application/json}}
  - {kind: body, file: provider-response.json}
`, path)
}

func TestCapturesRawRequest(t *testing.T) {
	// A body that is not valid UTF-8 and not valid JSON proves the fixture records
	// the bytes it received rather than a re-encoded reading of them.
	const body = "{\"model\":\"m\",\"raw\":\"\xff\xfe\",\"trailing\":true}\n"

	for _, path := range ProviderPaths() {
		t.Run(path, func(t *testing.T) {
			server, _ := start(t, bufferedScript(path), map[string]string{
				"provider-response.json": `{"ok":true}`,
			})

			answer := post(t, server, path, body, map[string]string{
				"X-Case-Id":         "seed-chat-identity",
				"anthropic-version": "2023-06-01",
			})
			if answer.Status != http.StatusOK {
				t.Fatalf("status = %d, want 200", answer.Status)
			}

			assertCaptured(t, only(t, server.Observed()), path, body)
		})
	}
}

// assertCaptured checks that one recorded request is the request that was sent,
// down to the body bytes and the headers the caller set.
func assertCaptured(t *testing.T, got ObservedRequest, path, body string) {
	t.Helper()

	if got.Method != http.MethodPost || got.Path != path {
		t.Errorf("method/path = %s %s, want POST %s", got.Method, got.Path, path)
	}
	if !bytes.Equal(got.Body, []byte(body)) {
		t.Errorf("body = %q, want %q", got.Body, body)
	}
	if value := got.Headers.Get("x-case-id"); value != "seed-chat-identity" {
		t.Errorf("header x-case-id = %q, want %q", value, "seed-chat-identity")
	}
	if value := got.Headers.Get("Anthropic-Version"); value != "2023-06-01" {
		t.Errorf("header anthropic-version = %q, want %q", value, "2023-06-01")
	}
	if got.Mismatch != "" {
		t.Errorf("mismatch = %q, want none", got.Mismatch)
	}
}

// only returns the single recorded request, failing when the fixture saw another
// number of them.
func only(t *testing.T, observed []ObservedRequest) ObservedRequest {
	t.Helper()

	if len(observed) != 1 {
		t.Fatalf("observed %d requests, want 1", len(observed))
	}
	return observed[0]
}

func TestObservedRecordsEveryRequestInOrder(t *testing.T) {
	server, _ := start(t, bufferedScript(PathMessages), map[string]string{
		"provider-response.json": `{"ok":true}`,
	})

	for i := range 3 {
		post(t, server, PathMessages, fmt.Sprintf(`{"turn":%d}`, i), nil)
	}

	observed := server.Observed()
	if len(observed) != 3 {
		t.Fatalf("observed %d requests, want 3", len(observed))
	}
	for i, got := range observed {
		if want := fmt.Sprintf(`{"turn":%d}`, i); string(got.Body) != want {
			t.Errorf("observed[%d].Body = %q, want %q", i, got.Body, want)
		}
	}
}

func TestExpectationMismatch(t *testing.T) {
	script := scriptHeader + `expect:
  method: POST
  path: /v1/messages
  headers:
    anthropic-version: "2023-06-01"
steps:
  - {kind: status, status: 200}
  - {kind: body, file: provider-response.json}
`
	tests := []struct {
		name    string
		path    string
		headers map[string]string
		want    string
	}{
		{
			name:    "matching request replays",
			path:    PathMessages,
			headers: map[string]string{"anthropic-version": "2023-06-01"},
			want:    "",
		},
		{
			name:    "wrong path is refused, not routed",
			path:    PathChatCompletions,
			headers: map[string]string{"anthropic-version": "2023-06-01"},
			want:    `path "/v1/chat/completions", want "/v1/messages"`,
		},
		{
			name: "missing expected header is refused",
			path: PathMessages,
			want: `header anthropic-version = "", want "2023-06-01"`,
		},
		{
			name:    "an unregistered path is recorded, never 404",
			path:    "/v1/complete",
			headers: map[string]string{"anthropic-version": "2023-06-01"},
			want:    `path "/v1/complete", want "/v1/messages"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, _ := start(t, script, map[string]string{"provider-response.json": `{"ok":true}`})

			answer := post(t, server, tt.path, `{}`, tt.headers)
			got := only(t, server.Observed())

			if tt.want == "" {
				if answer.Status != http.StatusOK || got.Mismatch != "" {
					t.Fatalf("status = %d, mismatch = %q, want 200 and none", answer.Status, got.Mismatch)
				}
				return
			}
			assertRefused(t, answer.Status, got.Mismatch, string(answer.Body), tt.want)
		})
	}
}

// assertRefused checks that a mismatched request was refused with the distinctive
// status, and that the reason reached both the record and the caller.
func assertRefused(t *testing.T, status int, mismatch, body, want string) {
	t.Helper()

	if status != StatusExpectationFailed {
		t.Errorf("status = %d, want %d", status, StatusExpectationFailed)
	}
	if !strings.Contains(mismatch, want) {
		t.Errorf("recorded mismatch = %q, want it to contain %q", mismatch, want)
	}
	if !strings.Contains(body, want) {
		t.Errorf("response body = %q, want it to contain %q", body, want)
	}
}

func TestRequestWithNoScriptLoadedIsRefused(t *testing.T) {
	server, _ := start(t, "", nil)

	answer := post(t, server, PathResponses, `{}`, nil)
	if answer.Status != StatusExpectationFailed {
		t.Fatalf("status = %d, want %d", answer.Status, StatusExpectationFailed)
	}

	observed := server.Observed()
	if len(observed) != 1 || observed[0].Mismatch != "no replay script is loaded" {
		t.Fatalf("observed = %+v", observed)
	}
}

func TestControlEndpointsDriveTheLifecycle(t *testing.T) {
	server, dir := start(t, "", map[string]string{"provider-response.json": `{"ok":true}`})

	// reset loads a script the same way an in-process caller does.
	script := bufferedScript(PathChatCompletions)
	if code := controlReset(t, server, dir, script); code != http.StatusNoContent {
		t.Fatalf("reset status = %d, want 204", code)
	}

	const body = `{"model":"m"}`
	if answer := post(t, server, PathChatCompletions, body, nil); answer.Status != http.StatusOK {
		t.Fatalf("provider status = %d, want 200", answer.Status)
	}

	// observed reports the same bytes the in-process reader sees.
	observed := controlObserved(t, server)
	if len(observed) != 1 || string(observed[0].Body) != body {
		t.Fatalf("observed = %+v", observed)
	}

	// A second reset clears the recorded requests.
	if code := controlReset(t, server, dir, script); code != http.StatusNoContent {
		t.Fatalf("second reset status = %d, want 204", code)
	}
	if len(controlObserved(t, server)) != 0 {
		t.Errorf("observed survived a reset")
	}
}

func TestControlResetRejectsAnInvalidScript(t *testing.T) {
	server, dir := start(t, "", nil)

	code := controlReset(t, server, dir, scriptHeader+"expect: {method: POST}\nsteps: [{kind: status, status: 200}]\n")
	if code != http.StatusBadRequest {
		t.Fatalf("reset status = %d, want 400", code)
	}
}

func TestConcurrentRequestsAndObservedPolling(t *testing.T) {
	server, _ := start(t, bufferedScript(PathResponses), map[string]string{
		"provider-response.json": `{"ok":true}`,
	})

	const requests = 16
	var wg sync.WaitGroup
	for i := range requests {
		wg.Add(1)
		go func() {
			defer wg.Done()
			post(t, server, PathResponses, fmt.Sprintf(`{"n":%d}`, i), nil)
		}()
	}

	polling := make(chan struct{})
	go func() {
		defer close(polling)
		for range 100 {
			_ = server.Observed()
		}
	}()

	wg.Wait()
	<-polling

	if got := len(server.Observed()); got != requests {
		t.Fatalf("observed %d requests, want %d", got, requests)
	}
}

func controlReset(t *testing.T, server *Server, dir, script string) int {
	t.Helper()

	resp, err := http.Post(server.URL()+PathReset+"?dir="+dir, "application/yaml", strings.NewReader(script))
	if err != nil {
		t.Fatalf("reset: %v", err)
	}
	defer resp.Body.Close()
	return resp.StatusCode
}

func controlObserved(t *testing.T, server *Server) []ObservedRequest {
	t.Helper()

	resp, err := http.Get(server.URL() + PathObserved)
	if err != nil {
		t.Fatalf("observed: %v", err)
	}
	defer resp.Body.Close()

	var observed []ObservedRequest
	if err := json.NewDecoder(resp.Body).Decode(&observed); err != nil {
		t.Fatalf("decode observed: %v", err)
	}
	return observed
}
