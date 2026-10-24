package execution

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// Pins v1 behaviour (as at 66c84d4d): a prefix rewrite such as a compaction is
// a model-switch boundary, so the next Decisions request carries no hold and
// no evidence from the invalidated transcript. Only execution_context_v2 keeps
// the served action, as last_served (TestV2ExplicitEventIsStrippedAndPrecedesMarkers).
func TestV1PrefixRewriteClearsHoldAndOldContextEvidence(t *testing.T) {
	for _, change := range []string{"instruction", "tools", "history"} {
		t.Run(change, func(t *testing.T) {
			var calls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if calls.Add(1) == 3 {
					w.WriteHeader(http.StatusServiceUnavailable)
					_, _ = io.WriteString(w, `{"type":"error","error":{"type":"overloaded_error","message":"fixture unavailable"}}`)
					return
				}
				_, _ = io.WriteString(w, `{"id":"msg_fixture","type":"message","role":"assistant","model":"qwen-fixture","content":[{"type":"text","text":"ready"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":30,"output_tokens":1}}`)
			}))
			defer upstream.Close()
			config := testConfig(t, upstream.URL, true)
			capture := &decisionCapture{}
			server, httpServer := executorWithHTTPDecisions(t, config, capture)
			body := map[string]any{
				"model": "arc-1.0", "max_tokens": 64, "system": "Original instructions",
				"messages": []map[string]any{{"role": "user", "content": "first"}},
			}
			execute := func(want int) {
				t.Helper()
				raw, err := json.Marshal(body)
				if err != nil {
					t.Fatal(err)
				}
				if status, response := executeHTTP(t, httpServer.URL, "/v1/messages", string(raw), "prefix-rewrite"); status != want {
					t.Fatalf("execution status %d, want %d: %s", status, want, response)
				}
			}
			execute(http.StatusOK)
			body["messages"] = append(body["messages"].([]map[string]any),
				map[string]any{"role": "assistant", "content": "ready"},
				map[string]any{"role": "user", "content": "second"})
			execute(http.StatusOK)
			state, err := server.states.acquire(t.Context(), "prefix-rewrite", "main")
			if err != nil {
				t.Fatal(err)
			}
			populated := len(state.placements) > 0 && len(state.workers) == 1 && len(state.ledger) == 2 && state.completed == 2
			state.release()
			if !populated {
				t.Fatal("completed private-append executions did not populate the original context")
			}
			body["messages"] = append(body["messages"].([]map[string]any),
				map[string]any{"role": "assistant", "content": "ready"},
				map[string]any{"role": "user", "content": "third"})
			switch change {
			case "instruction":
				body["system"] = "Changed instructions"
			case "tools":
				body["tools"] = []map[string]any{{"name": "read", "input_schema": map[string]any{"type": "object", "properties": map[string]any{}}}}
			case "history":
				body["messages"].([]map[string]any)[0]["content"] = "corrected first message"
			}
			execute(http.StatusServiceUnavailable)
			capture.mu.Lock()
			contexts := append([]externalContext(nil), capture.contexts...)
			capture.mu.Unlock()
			if len(contexts) != 3 {
				t.Fatalf("got %d Decisions lookups, want 3", len(contexts))
			}
			if contexts[0].Hold != nil || len(contexts[1].Attribution) != 1 || len(contexts[1].Workers) != 1 {
				t.Fatal("initial and extending requests did not carry the expected execution history")
			}
			rewritten := contexts[2]
			if rewritten.Hold != nil {
				t.Fatalf("v1 prefix rewrite kept a model hold: %#v", rewritten.Hold)
			}
			if rewritten.ContextEpoch == contexts[1].ContextEpoch || len(rewritten.Attribution) != 0 || len(rewritten.Workers) != 0 {
				t.Fatal("prefix rewrite retained old-epoch attribution or cache evidence")
			}
			state, err = server.states.acquire(t.Context(), "prefix-rewrite", "main")
			if err != nil {
				t.Fatal(err)
			}
			uncommitted := state.completed == 2 && state.held == "" && state.epoch == 1 &&
				len(state.placements) == 0 && len(state.workers) == 0 && len(state.ledger) == 0 &&
				len(state.issuers) == 0 && len(state.provenance) == 0 && state.pending != nil
			state.release()
			if !uncommitted {
				t.Fatal("failed rewritten request kept the v1 hold or committed context evidence")
			}
			execute(http.StatusOK)
			capture.mu.Lock()
			lookups := len(capture.contexts)
			capture.mu.Unlock()
			if lookups != 3 || calls.Load() != 4 {
				t.Fatal("identical retry did not reuse the retained decision")
			}
			state, err = server.states.acquire(t.Context(), "prefix-rewrite", "main")
			if err != nil {
				t.Fatal(err)
			}
			defer state.release()
			if state.completed != 3 || state.epoch != 1 || len(state.ledger) != 1 || len(state.workers) != 1 {
				t.Fatal("successful retry did not commit exactly once in the new context epoch")
			}
		})
	}
}

// v1 ignores harness events (the Pi extension ships with every hybrid launch):
// an explicit sidecall still advances state and an unlabelled follow-up is not
// reclassified, exactly as before v2. The body member never reaches the provider.
func TestV1IgnoresExplicitEventsButStripsThem(t *testing.T) {
	var bodies []map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		bodies = append(bodies, body)
		_, _ = io.WriteString(w, `{"id":"msg_fixture","type":"message","role":"assistant","model":"qwen-fixture","content":[{"type":"text","text":"ready"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":30,"output_tokens":1}}`)
	}))
	defer upstream.Close()
	config := testConfig(t, upstream.URL, true)
	server, httpServer := executorWithHTTPDecisions(t, config, &decisionCapture{})
	for _, body := range []string{
		`{"model":"arc-1.0","max_tokens":64,"rayline_event":{"kind":"task_start"},"messages":[{"role":"user","content":"first"}]}`,
		`{"model":"arc-1.0","max_tokens":64,"rayline_event":{"kind":"sidecall"},"messages":[{"role":"user","content":"title this"}]}`,
	} {
		if status, response := executeHTTP(t, httpServer.URL, "/v1/messages", body, "v1-events"); status != http.StatusOK {
			t.Fatalf("status %d: %s", status, response)
		}
	}
	for _, body := range bodies {
		if _, leaked := body["rayline_event"]; leaked {
			t.Fatal("event member reached the provider")
		}
	}
	state, err := server.states.acquire(t.Context(), "v1-events", "main")
	if err != nil {
		t.Fatal(err)
	}
	defer state.release()
	if state.completed != 2 || state.epoch != 1 || state.explicitEvents {
		t.Fatalf("v1 applied v2 event semantics: completed=%d epoch=%d explicit=%v", state.completed, state.epoch, state.explicitEvents)
	}
}
