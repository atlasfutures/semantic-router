package execution

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
)

// jinjaFailure is llama.cpp's answer when the chat template raises.
const jinjaFailure = `{"error":{"code":500,"message":"\n------------\nWhile executing CallExpression at line 110, column 28 in source:\n...eveloper\" %}↵        {{- raise_exception('System message must be at the beginnin...\n                                           ^\nError: Jinja Exception: System message must be at the beginning.","type":"server_error"}}`

func captureLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return &logs
}

func logRecords(t *testing.T, logs *bytes.Buffer, message string) []map[string]any {
	t.Helper()
	var records []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(logs.String()), "\n") {
		var record map[string]any
		if json.Unmarshal([]byte(line), &record) == nil && record["msg"] == message {
			records = append(records, record)
		}
	}
	return records
}

// failingLocal answers every request with status and body.
func failingLocal(t *testing.T, status int, body string) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)
	return server, &calls
}

// unavailableFirst selects the local action until the core reports it
// unavailable, then the cloud action.
func unavailableFirst(sent contextV2) string {
	for _, target := range sent.Targets {
		if target.ActionID == localActionID && target.Capacity != nil && !target.Capacity.Fits {
			return cloudActionID
		}
	}
	return localActionID
}

func TestLocalServerErrorBeforeOutputReselectsCloud(t *testing.T) {
	logs := captureLogs(t)
	local, localCalls := failingLocal(t, http.StatusInternalServerError, jinjaFailure)
	cloud := &upstreamFixture{}
	config := hybridV2Config(t, local.URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: unavailableFirst}
	_, httpServer := v2Server(t, config, fixture)
	status, header, response := executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "local-500")
	if status != 200 || header.Get("X-Rayline-Executed-Action-Id") != cloudActionID {
		t.Fatalf("status %d executed %q: %s", status, header.Get("X-Rayline-Executed-Action-Id"), response)
	}
	lookups := fixture.snapshot()
	if len(lookups) != 2 {
		t.Fatalf("got %d lookups, want the decision and one re-consult", len(lookups))
	}
	turn := lookups[1].Execution.Turn
	if turn.Reconsult == nil || *turn.Reconsult != "capacity" || turn.RetryOf == nil || *turn.RetryOf != "route-0" || turn.ID != lookups[0].Execution.Turn.ID {
		t.Fatalf("re-consult turn %+v", turn)
	}
	fact := lookups[1].Execution.Targets[0].Capacity
	if fact == nil || fact.Fits || fact.Reason == nil || *fact.Reason != CapacityRuntimeUnavailable {
		t.Fatalf("local fact %s", mustJSON(fact))
	}
	if localCalls.Load() != 1 || cloud.count() != 1 {
		t.Fatalf("local=%d cloud=%d", localCalls.Load(), cloud.count())
	}

	failures := logRecords(t, logs, "local upstream failed")
	if len(failures) != 1 {
		t.Fatalf("local failure logged %d times: %s", len(failures), logs)
	}
	failure := failures[0]
	detail, _ := failure["error"].(string)
	if failure["route_id"] != "route-0" || failure["deployment"] != config.Deployments[0].ID || failure["status"] != float64(500) ||
		!strings.Contains(detail, "raise_exception('System message must be at the beginnin") || len([]rune(detail)) > maxUpstreamErrorDetail {
		t.Fatalf("failure record %v", failure)
	}
	if strings.Contains(logs.String(), "Read the file.") {
		t.Fatal("the prompt reached the log")
	}
	completed := logRecords(t, logs, "execution completed")
	if len(completed) != 1 || completed[0]["deployment"] != "cloud-kimi" || completed[0]["route_id"] != "route-1" {
		t.Fatalf("completion %v", completed)
	}

	// The identical client retry keeps the cloud decision; the local runtime
	// that failed this turn is not asked again.
	if status, _, _ := executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "local-500"); status != http.StatusConflict {
		t.Fatalf("observed retry status %d", status)
	}
	if localCalls.Load() != 1 {
		t.Fatalf("local asked again: %d", localCalls.Load())
	}
}

func TestUnreachableLocalRuntimeReselectsCloud(t *testing.T) {
	logs := captureLogs(t)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	closed := "http://" + listener.Addr().String() + "/v1/chat/completions"
	listener.Close()
	cloud := &upstreamFixture{}
	config := hybridV2Config(t, closed, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: unavailableFirst}
	_, httpServer := v2Server(t, config, fixture)
	status, header, response := executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "local-down")
	if status != 200 || header.Get("X-Rayline-Executed-Action-Id") != cloudActionID || cloud.count() != 1 {
		t.Fatalf("status %d executed %q: %s", status, header.Get("X-Rayline-Executed-Action-Id"), response)
	}
	failures := logRecords(t, logs, "local upstream failed")
	if len(failures) != 1 || failures[0]["status"] != float64(0) || failures[0]["error"] != "connection refused" || strings.Contains(logs.String(), closed) {
		t.Fatalf("failure records %v", failures)
	}
}

// Only one local failure per turn is re-consulted: a second local runtime
// that also fails is forwarded.
func TestLocalFallbackRunsOncePerTurn(t *testing.T) {
	first, firstCalls := failingLocal(t, http.StatusInternalServerError, jinjaFailure)
	second, secondCalls := failingLocal(t, http.StatusServiceUnavailable, `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`)
	cloud := &upstreamFixture{}
	config := hybridV2Config(t, first.URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	otherLocal := strings.Repeat("e", 64)
	deployment := config.Deployments[0]
	deployment.ID, deployment.URL = "fixture-worker-2", second.URL
	action := config.Actions[0]
	action.ID, action.DeploymentID = otherLocal, deployment.ID
	config.Deployments = append(config.Deployments, deployment)
	config.Actions = append(config.Actions, action)
	fixture := &v2Decisions{choose: func(sent contextV2) string {
		refused := map[string]bool{}
		for _, target := range sent.Targets {
			refused[target.ActionID] = target.Capacity != nil && !target.Capacity.Fits
		}
		switch {
		case !refused[localActionID]:
			return localActionID
		case !refused[otherLocal]:
			return otherLocal
		}
		return cloudActionID
	}}
	_, httpServer := v2Server(t, config, fixture)
	status, _, response := executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "twice")
	if status != http.StatusServiceUnavailable {
		t.Fatalf("status %d: %s", status, response)
	}
	if firstCalls.Load() != 1 || secondCalls.Load() != 1 || cloud.count() != 0 || len(fixture.snapshot()) != 2 {
		t.Fatalf("first=%d second=%d cloud=%d lookups=%d", firstCalls.Load(), secondCalls.Load(), cloud.count(), len(fixture.snapshot()))
	}
}

func TestLocalClientErrorIsForwardedNotReselected(t *testing.T) {
	logs := captureLogs(t)
	local, calls := failingLocal(t, http.StatusBadRequest, `{"error":{"code":400,"message":"invalid tool schema","type":"invalid_request_error"}}`)
	cloud := &upstreamFixture{}
	config := hybridV2Config(t, local.URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{choose: unavailableFirst}
	_, httpServer := v2Server(t, config, fixture)
	if status, _, response := executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "local-400"); status != http.StatusBadRequest {
		t.Fatalf("status %d: %s", status, response)
	}
	if calls.Load() != 1 || cloud.count() != 0 || len(fixture.snapshot()) != 1 {
		t.Fatalf("local=%d cloud=%d lookups=%d", calls.Load(), cloud.count(), len(fixture.snapshot()))
	}
	if failures := logRecords(t, logs, "local upstream failed"); len(failures) != 1 || failures[0]["status"] != float64(400) {
		t.Fatalf("failure records %v", failures)
	}
}

func TestUpstreamErrorDetailKeepsOnlyTheMessage(t *testing.T) {
	for body, want := range map[string]string{
		`{"error":{"message":"boom","input":"secret prompt"}}`: "boom",
		`{"error":"boom"}`: "boom",
		`{"message":"boom","detail":"secret prompt"}`: "boom",
		`secret prompt in plain text`:                 "",
	} {
		if got := upstreamErrorDetail([]byte(body)); got != want {
			t.Fatalf("%s: got %q", body, got)
		}
	}
	if got := truncateDetail(strings.Repeat("é", 300)); len([]rune(got)) != maxUpstreamErrorDetail {
		t.Fatalf("truncated to %d runes", len([]rune(got)))
	}
}

// Every action bound to the failed deployment is out for the turn, so a
// sibling action is never dispatched to the same failing runtime.
func TestLocalFailureExcludesEverySiblingOnThatDeployment(t *testing.T) {
	local, calls := failingLocal(t, http.StatusInternalServerError, jinjaFailure)
	cloud := &upstreamFixture{}
	config := hybridV2Config(t, local.URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	sibling := config.Actions[0]
	sibling.ID = strings.Repeat("f", 64)
	config.Actions = append(config.Actions, sibling)
	fixture := &v2Decisions{choose: func(sent contextV2) string {
		refused := map[string]bool{}
		for _, target := range sent.Targets {
			refused[target.ActionID] = target.Capacity != nil && !target.Capacity.Fits
		}
		switch {
		case !refused[localActionID]:
			return localActionID
		case !refused[sibling.ID]:
			return sibling.ID
		}
		return cloudActionID
	}}
	_, httpServer := v2Server(t, config, fixture)
	if status, _, response := executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "sibling"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	if calls.Load() != 1 || cloud.count() != 1 || len(fixture.snapshot()) != 2 {
		t.Fatalf("local=%d cloud=%d lookups=%d", calls.Load(), cloud.count(), len(fixture.snapshot()))
	}
}

// downCounter is a qualified counter whose runtime can go away.
type downCounter struct {
	identity cachecontext.PromptIdentity
	down     atomic.Bool
}

func (counter *downCounter) Identity() cachecontext.PromptIdentity { return counter.identity }

func (counter *downCounter) Count(context.Context, cachecontext.PreparedPrompt) (cachecontext.PromptMeasurement, error) {
	if counter.down.Load() {
		return cachecontext.PromptMeasurement{}, errors.New("counter runtime unreachable")
	}
	return cachecontext.PromptMeasurement{PromptTokens: 300, Provenance: "provider_reported"}, nil
}

// When the fallback consult itself fails, the identical client retry still
// holds the failed local decision. Its recorded fact re-consults before the
// request is prepared or counted, so a counter on the same failed runtime
// cannot fail the retry.
func TestRetryAfterFailedFallbackConsultSkipsTheFailedRuntime(t *testing.T) {
	counter := &downCounter{}
	var calls atomic.Int64
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		counter.down.Store(true)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, jinjaFailure)
	}))
	t.Cleanup(local.Close)
	cloud := &upstreamFixture{}
	config := hybridV2Config(t, local.URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	fixture := &v2Decisions{}
	fixture.choose = func(sent contextV2) string {
		if sent.Turn.Reconsult == nil {
			fixture.mu.Lock()
			fixture.failNext = 1
			fixture.mu.Unlock()
		}
		return unavailableFirst(sent)
	}
	server, httpServer := v2Server(t, config, fixture)
	counter.identity = promptIdentity(config.Deployments[0])
	if err := server.RegisterPromptCounter(config.Deployments[0].ID, counter); err != nil {
		t.Fatal(err)
	}
	if status, _, response := executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "retry-down"); status != http.StatusBadGateway {
		t.Fatalf("first attempt status %d: %s", status, response)
	}
	status, header, response := executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "retry-down")
	if status != 200 || header.Get("X-Rayline-Executed-Action-Id") != cloudActionID {
		t.Fatalf("retry status %d executed %q: %s", status, header.Get("X-Rayline-Executed-Action-Id"), response)
	}
	if calls.Load() != 1 || cloud.count() != 1 {
		t.Fatalf("local=%d cloud=%d", calls.Load(), cloud.count())
	}
}

// A runtime that sends a 5xx status and then stalls on its body still falls
// back once the bounded read gives up.
func TestStalledLocalErrorBodyStillFallsBack(t *testing.T) {
	previous := localErrorBodyTimeout
	localErrorBodyTimeout = 50 * time.Millisecond
	t.Cleanup(func() { localErrorBodyTimeout = previous })
	release := make(chan struct{})
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":`)
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(local.Close)
	cloud := &upstreamFixture{}
	config := hybridV2Config(t, local.URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	_, httpServer := v2Server(t, config, &v2Decisions{choose: unavailableFirst})
	// Registered last so it runs first: servers close only after the stall ends.
	t.Cleanup(func() { close(release) })
	done := make(chan int, 1)
	go func() {
		status, _, _ := executeWithHeaders(t, httpServer.URL, chatTurn(userTask), "stalled")
		done <- status
	}()
	select {
	case status := <-done:
		if status != 200 || cloud.count() != 1 {
			t.Fatalf("status %d cloud=%d", status, cloud.count())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a stalled local error body held the turn")
	}
}
