package execution

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
)

// refusalLogs returns the "decisions lookup refused" records logged while run.
func refusalLogs(t *testing.T, run func()) []map[string]any {
	t.Helper()
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	defer slog.SetDefault(previous)
	run()
	if strings.Contains(logs.String(), "private refusal sentinel") || strings.Contains(logs.String(), "fixture-hosted-key-never-for-local") {
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

// Every HTTP refusal from the service logs one line: status, contract code,
// and the route id its header names, never request material. The core's 502
// carries that route id, as does one for a response it could not accept, and
// omits the header when the service named none.
func TestDecisionsRefusalLogsStatusCodeAndRouteID(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	var unreadable, unnamed atomic.Bool
	fixture := &v2Decisions{answer: func(w http.ResponseWriter) bool {
		if !unnamed.Load() {
			w.Header().Set("X-Rayline-Route-Id", "route-refused-1")
		}
		if unreadable.Load() {
			w.Header().Set("X-Rayline-Route-Id", "route-unreadable-1")
			_, _ = w.Write([]byte(`not json`))
			return true
		}
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"error":{"code":"projection_refused"}}`))
		return true
	}}
	_, httpServer := v2Server(t, config, fixture)
	var (
		status int
		header http.Header
	)
	records := refusalLogs(t, func() {
		status, header, _ = executeWithHeaders(t, httpServer.URL, chatTurn(`{"role":"user","content":"private refusal sentinel"}`), "refused")
	})
	if status != http.StatusBadGateway || header.Get("X-Rayline-Route-Id") != "route-refused-1" {
		t.Fatalf("status %d route id %q", status, header.Get("X-Rayline-Route-Id"))
	}
	if len(records) != 1 || records[0]["source"] != "service" || records[0]["status"] != float64(422) || records[0]["code"] != "projection_refused" || records[0]["route_id"] != "route-refused-1" {
		t.Fatalf("refusal log %v", records)
	}
	unreadable.Store(true)
	status, header, response := executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "unreadable")
	if status != http.StatusBadGateway || !strings.Contains(string(response), "invalid Decisions response") || header.Get("X-Rayline-Route-Id") != "route-unreadable-1" {
		t.Fatalf("status %d route id %q: %s", status, header.Get("X-Rayline-Route-Id"), response)
	}
	// A refusal that names no route id leaves the header off the 502.
	unreadable.Store(false)
	unnamed.Store(true)
	status, header, _ = executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "unnamed")
	if _, present := header["X-Rayline-Route-Id"]; status != http.StatusBadGateway || present {
		t.Fatalf("status %d route id %q", status, header.Get("X-Rayline-Route-Id"))
	}
}
