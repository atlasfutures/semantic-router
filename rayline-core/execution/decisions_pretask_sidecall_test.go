package execution

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"testing"
)

const titleSidecall = `{"model":"arc-1.0","max_tokens":64,"rayline_event":{"kind":"sidecall"},"messages":[{"role":"user","content":"Title this conversation: private sidecall sentinel"}]}`

// refusalLogs returns the "decisions lookup refused" records logged while run.
func refusalLogs(t *testing.T, run func()) []map[string]any {
	t.Helper()
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	defer slog.SetDefault(previous)
	run()
	if strings.Contains(logs.String(), "private sidecall sentinel") || strings.Contains(logs.String(), "fixture-hosted-key-never-for-local") {
		t.Fatal("request material or a credential entered the log")
	}
	var records []map[string]any
	for _, line := range bytes.Split(bytes.TrimSpace(logs.Bytes()), []byte("\n")) {
		var record map[string]any
		if json.Unmarshal(line, &record) == nil && record["msg"] == "decisions lookup refused" {
			records = append(records, record)
		}
	}
	return records
}

// A side call before the branch's first task start has no last_served for
// Decisions to project from. The core answers the service's refusal itself,
// with no lookup, in the exact shape of that refusal, marked final.
func TestV2PreTaskSidecallSkipsDecisionsAndAnswersItsRefusal(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	_, httpServer := v2Server(t, config, fixture)
	var (
		status   int
		header   http.Header
		response []byte
	)
	records := refusalLogs(t, func() {
		status, header, response = executeWithHeaders(t, httpServer.URL, titleSidecall, "pretask")
	})
	if status != http.StatusBadGateway || !strings.Contains(string(response), `"decision_failed"`) || !strings.Contains(string(response), "Decisions request refused (HTTP 422, projection_refused)") {
		t.Fatalf("status %d: %s", status, response)
	}
	if got := header.Get("x-should-retry"); got != "false" {
		t.Errorf("x-should-retry %q want false", got)
	}
	if got := header.Get("Retry-After"); got != "" {
		t.Errorf("Retry-After %q want none", got)
	}
	if lookups := fixture.snapshot(); len(lookups) != 0 {
		t.Fatalf("pre-task sidecall consulted Decisions %d times", len(lookups))
	}
	if local.count() != 0 || cloud.count() != 0 {
		t.Fatal("pre-task sidecall was dispatched")
	}
	if len(records) != 1 || records[0]["source"] != "local" || records[0]["status"] != float64(422) || records[0]["code"] != "projection_refused" || records[0]["route_id"] != nil {
		t.Fatalf("refusal log %v", records)
	}
}

// Once a task start has served a model, a side call is decided as before:
// Decisions is consulted with last_served and the call is served.
func TestV2SidecallAfterTaskStartStillConsultsDecisions(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	_, httpServer := v2Server(t, config, fixture)
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "posttask"); status != 200 {
		t.Fatalf("task start status %d: %s", status, response)
	}
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", titleSidecall, "posttask"); status != 200 {
		t.Fatalf("sidecall status %d: %s", status, response)
	}
	lookups := fixture.snapshot()
	if len(lookups) != 2 || lookups[1].Execution.Event.kindName() != EventSidecall || lookups[1].Execution.LastServed == nil || lookups[1].Execution.LastServed.ActionID != localActionID {
		t.Fatalf("sidecall lookup %+v", lookups)
	}
}

// Every HTTP refusal from the service logs one line: status, contract code,
// and the route id its header names, never request material.
func TestDecisionsRefusalLogsStatusCodeAndRouteID(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{answer: func(w http.ResponseWriter) bool {
		w.Header().Set("X-Rayline-Route-Id", "route-refused-1")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"error":{"code":"projection_refused"}}`))
		return true
	}}
	_, httpServer := v2Server(t, config, fixture)
	var status int
	records := refusalLogs(t, func() {
		status, _ = executeHTTP(t, httpServer.URL, "/v1/chat/completions", chatTurn(userTask), "refused")
	})
	if status != http.StatusBadGateway {
		t.Fatalf("status %d", status)
	}
	if len(records) != 1 || records[0]["source"] != "service" || records[0]["status"] != float64(422) || records[0]["code"] != "projection_refused" || records[0]["route_id"] != "route-refused-1" {
		t.Fatalf("refusal log %v", records)
	}
}
