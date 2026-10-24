package execution

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestGenerationAttemptReservationsAreProcessWideAndAtomic(t *testing.T) {
	server := &Server{config: Config{MaxGenerationAttempts: 7}}
	var admitted atomic.Int64
	var group sync.WaitGroup
	for range 100 {
		group.Add(1)
		go func() {
			defer group.Done()
			if server.reserveGenerationAttempt() {
				admitted.Add(1)
			}
		}()
	}
	group.Wait()
	if admitted.Load() != 7 || server.generationAttempts.Load() != 7 || server.reserveGenerationAttempt() {
		t.Fatalf("limit exceeded: %d", admitted.Load())
	}
	unlimited := &Server{}
	for range 100 {
		if !unlimited.reserveGenerationAttempt() {
			t.Fatal("disabled cap rejected an attempt")
		}
	}
}

func TestFailedGenerationConsumesTheAttemptLimit(t *testing.T) {
	var calls atomic.Int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		writeJSON(w, 503, map[string]any{"error": map[string]string{"message": "unavailable", "type": "server_error"}})
	}))
	defer upstream.Close()
	config := testConfig(t, upstream.URL, false)
	config.MaxGenerationAttempts = 1
	_, server := executorWithHTTPDecisions(t, config, &decisionCapture{})
	body := `{"model":"arc-1.0","messages":[{"role":"user","content":"test"}]}`
	status, _ := executeHTTP(t, server.URL, "/v1/chat/completions", body, "first")
	if status != 503 {
		t.Fatalf("first failure: %d", status)
	}
	for _, session := range []string{"first", "another-session"} {
		status, response := executeHTTP(t, server.URL, "/v1/chat/completions", body, session)
		if status != 429 || !strings.Contains(string(response), "generation_limit_reached") {
			t.Fatalf("exhausted cap: %d %s", status, response)
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("failed call or new session bypassed cap: %d", calls.Load())
	}
}
