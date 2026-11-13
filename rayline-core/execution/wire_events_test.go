package execution

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

type replayRequest struct {
	name     string
	format   llmprotocol.WireFormat
	body     []byte
	affinity bool
}

func fixtureFile(t *testing.T, name string) replayRequest {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "wire_events", name))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Body json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	format := llmprotocol.AnthropicMessagesV1
	var probe map[string]json.RawMessage
	_ = json.Unmarshal(fixture.Body, &probe)
	if _, responses := probe["input"]; responses {
		format = llmprotocol.OpenAIResponsesV1
	}
	return replayRequest{name: name, format: format, body: fixture.Body, affinity: true}
}

func fixtureLines(t *testing.T, name string) []replayRequest {
	t.Helper()
	file, err := os.Open(filepath.Join("testdata", "wire_events", name))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var result []replayRequest
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1<<20), 8<<20)
	for scanner.Scan() {
		var line struct {
			SessionAffinity bool            `json:"session_affinity"`
			Body            json.RawMessage `json:"body"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &line); err != nil {
			t.Fatal(err)
		}
		result = append(result, replayRequest{name: fmt.Sprintf("%s#%d", name, len(result)), format: llmprotocol.AnthropicMessagesV1, body: line.Body, affinity: line.SessionAffinity})
	}
	return result
}

func isWirePrefix(previous, current []string) bool {
	if len(previous) > len(current) {
		return false
	}
	for index := range previous {
		if previous[index] != current[index] {
			return false
		}
	}
	return true
}

// replayEvents applies the core's v2 inference to a real request sequence,
// comparing each request with the last non-sidecall request, as the core does.
func replayEvents(requests []replayRequest) []ExecutionEvent {
	var previous []string
	var events []ExecutionEvent
	for _, request := range requests {
		wire := wireMessageFingerprints(request.format, request.body)
		rewrite := previous != nil && !isWirePrefix(previous, wire)
		event := inferV2Event(parseWireRequest(request.format, request.body), previous, wire, rewrite, false)
		events = append(events, event)
		if !event.is(EventSidecall) {
			previous = wire
		}
	}
	return events
}

type wantEvent struct {
	kind       string
	compaction *CompactionEvent
}

func retained(index int) *int { return &index }

func assertEvents(t *testing.T, got []ExecutionEvent, want []wantEvent) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d events, want %d", len(got), len(want))
	}
	for index, event := range got {
		if event.kindName() != want[index].kind || event.Basis != BasisMarkerInferred {
			t.Errorf("request %d: event %s/%s, want %s/marker_inferred", index, event.kindName(), event.Basis, want[index].kind)
		}
		if mustJSON(event.Compaction) != mustJSON(want[index].compaction) {
			t.Errorf("request %d: compaction %s, want %s", index, mustJSON(event.Compaction), mustJSON(want[index].compaction))
		}
	}
}

func TestWireEventsReplayRealHarnessSequences(t *testing.T) {
	newCompaction := func(summary int, from *int) *CompactionEvent {
		return &CompactionEvent{SummaryMessage: summary, RetainedFrom: from, New: true}
	}
	for name, test := range map[string]struct {
		requests []replayRequest
		want     []wantEvent
	}{
		// Stock Pi 0.80.2: a tool loop, an overflow, Pi's own summarization call,
		// then the compacted transcript with the retained last tool exchange.
		"pi retained tail": {fixtureLines(t, "pi_0.80.2_overflow_retained_tail.jsonl"), []wantEvent{
			{EventTaskStart, nil}, {EventToolResult, nil}, {EventToolResult, nil}, {EventSidecall, nil},
			{EventCompaction, newCompaction(0, retained(1))}, {EventToolResult, nil}, {EventToolResult, nil}, {EventUserMessage, nil}}},
		"pi no tail": {fixtureLines(t, "pi_0.80.2_overflow_no_tail.jsonl"), []wantEvent{
			{EventTaskStart, nil}, {EventToolResult, nil}, {EventToolResult, nil}, {EventSidecall, nil},
			{EventCompaction, newCompaction(0, nil)}, {EventToolResult, nil}, {EventToolResult, nil}, {EventToolResult, nil}}},
		"claude code 2.1.280": {[]replayRequest{fixtureFile(t, "claude_code_2.1.280_main_turn.json"), fixtureFile(t, "claude_code_2.1.280_compaction_request.json"),
			fixtureFile(t, "claude_code_2.1.280_post_compaction.json"), fixtureFile(t, "claude_code_2.1.280_title.json")}, []wantEvent{
			{EventToolResult, nil}, {EventSidecall, nil}, {EventCompaction, newCompaction(0, nil)}, {EventSidecall, nil}}},
		"claude code 2.1.309": {[]replayRequest{fixtureFile(t, "claude_code_2.1.309_main_turn_before_suggestion.json"), fixtureFile(t, "claude_code_2.1.309_prompt_suggestion.json"),
			fixtureFile(t, "claude_code_2.1.309_main_turn_after_suggestion.json"), fixtureFile(t, "claude_code_2.1.309_main_turn_before_compaction.json"),
			fixtureFile(t, "claude_code_2.1.309_compaction_request.json"), fixtureFile(t, "claude_code_2.1.309_post_compaction.json")}, []wantEvent{
			{EventToolResult, nil}, {EventSidecall, nil}, {EventUserMessage, nil}, {EventToolResult, nil}, {EventSidecall, nil}, {EventCompaction, newCompaction(0, nil)}}},
		"codex 0.154.0": {[]replayRequest{fixtureFile(t, "codex_0.154.0_main_turn.json"), fixtureFile(t, "codex_0.154.0_compaction_request.json"),
			fixtureFile(t, "codex_0.154.0_post_compaction.json"), fixtureFile(t, "codex_0.154.0_post_compaction_next.json")}, []wantEvent{
			{EventToolResult, nil}, {EventSidecall, nil}, {EventCompaction, newCompaction(3, nil)}, {EventToolResult, nil}}},
	} {
		t.Run(name, func(t *testing.T) { assertEvents(t, replayEvents(test.requests), test.want) })
	}
}

// Without a known summary literal, a compaction is still recognized from its
// structure: a new first user message replacing at least two earlier messages,
// followed by a retained run that ends where the previous request ended.
func TestWireEventsStructuralCompactionAndPlainRewrites(t *testing.T) {
	pi := fixtureLines(t, "pi_0.80.2_overflow_retained_tail.jsonl")
	unmarked := pi[4]
	unmarked.body = bytes.Replace(unmarked.body, []byte(piCompactionPrefix), []byte("Summary of earlier work:"), 1)
	events := replayEvents([]replayRequest{pi[0], pi[1], pi[2], unmarked})
	if !events[3].is(EventCompaction) || mustJSON(events[3].Compaction) != `{"summary_message":0,"retained_from":1,"new":true}` {
		t.Fatalf("structural compaction %+v %s", events[3], mustJSON(events[3].Compaction))
	}
	// Editing the first message keeps everything after it: a prefix rewrite
	// (new epoch), never a switch boundary.
	edited := pi[2]
	edited.body = bytes.Replace(edited.body, []byte("Read big.txt three times"), []byte("Read big.txt four times"), 1)
	events = replayEvents([]replayRequest{pi[0], pi[1], pi[2], edited})
	if !events[3].is(EventToolResult) || events[3].Compaction != nil {
		t.Fatalf("plain rewrite became %+v", events[3])
	}
}

func TestWireEventsTypedTurnsSubagentsAndUnknown(t *testing.T) {
	messages := func(raw string) wireRequest {
		return parseWireRequest(llmprotocol.AnthropicMessagesV1, []byte(`{"messages":`+raw+`}`))
	}
	for name, test := range map[string]struct {
		request wireRequest
		lineage bool
		want    string
	}{
		"reminder only is not typed": {messages(`[{"role":"user","content":"go"},{"role":"assistant","content":"ok"},{"role":"user","content":[{"type":"text","text":"<system-reminder>todo list changed</system-reminder>"}]}]`), false, "unknown"},
		"typed beside reminder":      {messages(`[{"role":"user","content":"go"},{"role":"assistant","content":"ok"},{"role":"user","content":[{"type":"text","text":"<system-reminder>x</system-reminder>"},{"type":"text","text":"now fix it"}]}]`), false, EventUserMessage},
		"command expansion":          {messages(`[{"role":"user","content":"go"},{"role":"assistant","content":"ok"},{"role":"user","content":"<command-name>/review</command-name>"}]`), false, "unknown"},
		"subagent report as text":    {messages(`[{"role":"user","content":"go"},{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"Agent","input":{}}]},{"role":"user","content":"the subagent found two files"}]`), false, EventToolResult},
		"claude code subagent claim": {parseWireRequest(llmprotocol.AnthropicMessagesV1, []byte(`{"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.309; cc_is_subagent=true"}],"messages":[{"role":"user","content":"explore"}]}`)), false, EventSubagentStart},
		"child session lineage":      {messages(`[{"role":"user","content":"explore"}]`), true, EventSubagentStart},
		"opening prompt":             {messages(`[{"role":"user","content":"explore"}]`), false, EventTaskStart},
	} {
		t.Run(name, func(t *testing.T) {
			if got := inferWireEvent(test.request, test.lineage).kindName(); got != test.want {
				t.Fatalf("event %s, want %s", got, test.want)
			}
		})
	}
}

// End to end through the core with stock Pi's real compaction sequence: the
// failed overflow request, the session-less summarization call (stateless side
// call) and the compacted transcript, whose retained reply keeps its
// attribution at its new position.
func TestV2CoreReplaysStockPiCompaction(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Stream bool `json:"stream"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		calls++
		call := calls
		mu.Unlock()
		if call == 3 { // Pi's third request overflowed at the provider.
			w.WriteHeader(400)
			_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long"}}`)
			return
		}
		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, messagesStreamReady)
			return
		}
		_, _ = io.WriteString(w, messagesReady)
	}))
	defer upstream.Close()
	cloud := &upstreamFixture{messages: true}
	config := onProtocol(t, hybridV2Config(t, upstream.URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions"), llmprotocol.AnthropicMessagesV1)
	config.Deployments[0].ContextWindow, config.Deployments[0].MaxOutputTokens = 65536, 4096 // Pi sends max_tokens 2048
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	server, httpServer := v2Server(t, config, fixture)
	statuses := []int{}
	for _, request := range fixtureLines(t, "pi_0.80.2_overflow_retained_tail.jsonl") {
		httpRequest, _ := http.NewRequest(http.MethodPost, httpServer.URL+"/v1/messages", bytes.NewReader(request.body))
		httpRequest.Header.Set("Authorization", "Bearer "+testCoreToken)
		if request.affinity {
			httpRequest.Header.Set("X-Session-Affinity", "pi-session")
		}
		response, err := http.DefaultClient.Do(httpRequest)
		if err != nil {
			t.Fatal(err)
		}
		reply, _ := io.ReadAll(response.Body)
		response.Body.Close()
		if response.StatusCode != 200 {
			t.Logf("%s: %d %s", request.name, response.StatusCode, reply)
		}
		statuses = append(statuses, response.StatusCode)
	}
	if fmt.Sprint(statuses) != "[200 200 400 200 200 200 200 200]" {
		t.Fatalf("statuses %v", statuses)
	}
	lookups := fixture.snapshot()
	kinds := []string{}
	for _, lookup := range lookups {
		kinds = append(kinds, lookup.Execution.Event.kindName())
	}
	if strings.Join(kinds, ",") != "task_start,tool_result,tool_result,sidecall,compaction,tool_result,tool_result,user_message" {
		t.Fatalf("events %v", kinds)
	}
	compacted := lookups[4].Execution
	if compacted.Turn.Epoch == lookups[1].Execution.Turn.Epoch || compacted.LastServed == nil || compacted.LastServed.Message == nil || *compacted.LastServed.Message != 1 {
		t.Fatalf("compacted epoch/last_served %+v", compacted.LastServed)
	}
	if len(compacted.History) != 1 || compacted.History[0].Message != 1 || compacted.History[0].ActionID != localActionID {
		t.Fatalf("retained history %+v", compacted.History)
	}
	if lookups[3].Execution.LastServed != nil || len(lookups[3].Execution.History) != 0 {
		t.Fatal("the session-less summarization call saw main-session state")
	}
	state, err := server.states.acquire(t.Context(), "pi-session", "main")
	if err != nil {
		t.Fatal(err)
	}
	defer state.release()
	if state.completed != 6 {
		t.Fatalf("completed %d, want 6 (overflow failed, summary call stateless)", state.completed)
	}
}

// After a real compaction every request carries the summary; a later plain
// rewrite is a new epoch, never a second compaction.
func TestWireEventsCarriedSummaryIsNotANewCompaction(t *testing.T) {
	pi := fixtureLines(t, "pi_0.80.2_overflow_retained_tail.jsonl")
	edited := pi[6]
	var body map[string]any
	if err := json.Unmarshal(edited.body, &body); err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]any)
	messages[2] = map[string]any{"role": "user", "content": []any{map[string]any{"type": "tool_result", "tool_use_id": "toolu_x", "content": "cleared"}}}
	edited.body, _ = json.Marshal(body)
	events := replayEvents([]replayRequest{pi[4], pi[5], edited})
	if events[2].is(EventCompaction) {
		t.Fatalf("carried summary read as a new compaction: %s", mustJSON(events[2].Compaction))
	}
}

// A compaction whose first consult fails is still a compaction on the retry.
func TestV2InferredCompactionSurvivesAFailedConsult(t *testing.T) {
	var mu sync.Mutex
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Stream bool `json:"stream"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, messagesStreamReady)
	}))
	defer upstream.Close()
	cloud := &upstreamFixture{messages: true}
	config := onProtocol(t, hybridV2Config(t, upstream.URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions"), llmprotocol.AnthropicMessagesV1)
	config.Deployments[0].ContextWindow, config.Deployments[0].MaxOutputTokens = 65536, 4096
	fixture := &v2Decisions{choose: func(contextV2) string { return localActionID }}
	_, httpServer := v2Server(t, config, fixture)
	send := func(request replayRequest) int {
		httpRequest, _ := http.NewRequest(http.MethodPost, httpServer.URL+"/v1/messages", bytes.NewReader(request.body))
		httpRequest.Header.Set("Authorization", "Bearer "+testCoreToken)
		httpRequest.Header.Set("X-Session-Affinity", "pi-retry")
		response, err := http.DefaultClient.Do(httpRequest)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, response.Body)
		response.Body.Close()
		return response.StatusCode
	}
	pi := fixtureLines(t, "pi_0.80.2_overflow_retained_tail.jsonl")
	for _, index := range []int{0, 1, 2} {
		if status := send(pi[index]); status != 200 {
			t.Fatalf("request %d status %d", index, status)
		}
	}
	fixture.mu.Lock()
	fixture.failNext = 1
	fixture.mu.Unlock()
	if status := send(pi[4]); status == 200 {
		t.Fatal("consult failure was not surfaced")
	}
	if status := send(pi[4]); status != 200 {
		t.Fatalf("retry status %d", status)
	}
	lookups := fixture.snapshot()
	last := lookups[len(lookups)-1].Execution
	if !last.Event.is(EventCompaction) || len(last.History) != 1 || last.History[0].Message != 1 {
		t.Fatalf("retry event %s history %+v", last.Event.kindName(), last.History)
	}
}

func TestWireEventsResponsesStringInputIsNotIndexed(t *testing.T) {
	body := []byte(`{"model":"m","input":"` + codexSummaryPrefix[:60] + `"}`)
	request := parseWireRequest(llmprotocol.OpenAIResponsesV1, body)
	if got := detectCompaction(request, []string{"a", "b", "c"}, wireMessageFingerprints(llmprotocol.OpenAIResponsesV1, body)); got != nil {
		t.Fatalf("compaction %+v", got)
	}
	body = []byte(`{"model":"m","input":"` + piCompactionPrefix + ` summary"}`)
	_ = inferV2Event(parseWireRequest(llmprotocol.OpenAIResponsesV1, body), []string{"a", "b", "c"}, wireMessageFingerprints(llmprotocol.OpenAIResponsesV1, body), true, false)
}

func TestWireEventsResponsesStringInputSidecall(t *testing.T) {
	body, _ := json.Marshal(map[string]any{"model": "m", "input": codexCompactionPrompt})
	if !parseWireRequest(llmprotocol.OpenAIResponsesV1, body).isSidecall() {
		t.Fatal("Codex compaction prompt as string input was not a side call")
	}
}

// Hermes asks for a final summary when its tool loop hits max_iterations. On
// Messages that request carries no session header (on Chat and Responses it
// does), so the literal must be a side call on every wire; an ordinary user
// turn with the same history is not, and neither is a user message that
// merely starts with the literal.
func TestWireEventsHermesIterationSummaryIsSidecall(t *testing.T) {
	history := []map[string]any{
		{"role": "user", "content": "List the files, then summarise them."},
		{"role": "assistant", "content": []map[string]any{{"type": "tool_use", "id": "toolu_1", "name": "terminal", "input": map[string]any{"command": "ls"}}}},
		{"role": "user", "content": []map[string]any{{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"}}},
	}
	summary := append(append([]map[string]any{}, history...), map[string]any{"role": "user", "content": hermesIterationSummary})
	body, _ := json.Marshal(map[string]any{"model": "m", "max_tokens": 1024, "messages": summary})
	if !parseWireRequest(llmprotocol.AnthropicMessagesV1, body).isSidecall() {
		t.Fatal("Hermes iteration-cap summary on Messages was not a side call")
	}
	body, _ = json.Marshal(map[string]any{"model": "m", "input": []map[string]any{
		{"role": "user", "content": "List the files, then summarise them."},
		{"role": "user", "content": hermesIterationSummary},
	}})
	if !parseWireRequest(llmprotocol.OpenAIResponsesV1, body).isSidecall() {
		t.Fatal("Hermes iteration-cap summary on Responses was not a side call")
	}
	body, _ = json.Marshal(map[string]any{"model": "m", "input": []map[string]any{
		{"role": "user", "content": []map[string]any{{"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}, {"type": "input_text", "text": hermesIterationSummary}}},
	}})
	if parseWireRequest(llmprotocol.OpenAIResponsesV1, body).isSidecall() {
		t.Fatal("an image with the literal as its only text on Responses was a side call")
	}
	body, _ = json.Marshal(map[string]any{"model": "m", "messages": []map[string]any{
		{"role": "user", "content": "List the files, then summarise them."},
		{"role": "user", "content": hermesIterationSummary},
	}})
	if !parseWireRequest(llmprotocol.OpenAIChatV1, body).isSidecall() {
		t.Fatal("Hermes iteration-cap summary on Chat was not a side call")
	}
	for _, last := range []any{
		"Now read a.txt.",
		hermesIterationSummary + " Also list the directory again.",
		// The literal as the last block of a longer message is still a turn.
		[]map[string]any{{"type": "text", "text": "Now read a.txt."}, {"type": "text", "text": hermesIterationSummary}},
		// An image with the literal as its only text is a turn, not the nudge.
		[]map[string]any{{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}, {"type": "text", "text": hermesIterationSummary}},
	} {
		turn := append(append([]map[string]any{}, history...), map[string]any{"role": "user", "content": last})
		body, _ = json.Marshal(map[string]any{"model": "m", "max_tokens": 1024, "messages": turn})
		if parseWireRequest(llmprotocol.AnthropicMessagesV1, body).isSidecall() {
			t.Fatalf("user turn %v after the same history was a side call", last)
		}
	}
	// The literal as the sole block of a block-array message is the summary.
	block := append(append([]map[string]any{}, history...), map[string]any{"role": "user", "content": []map[string]any{{"type": "text", "text": hermesIterationSummary}}})
	body, _ = json.Marshal(map[string]any{"model": "m", "max_tokens": 1024, "messages": block})
	if !parseWireRequest(llmprotocol.AnthropicMessagesV1, body).isSidecall() {
		t.Fatal("Hermes iteration-cap summary as a single text block was not a side call")
	}
	// The shape Hermes really sends when the loop exhausts after a tool call:
	// its Anthropic converter merges the tool row and the nudge that follows
	// into one user message, so the marker arrives beside tool results.
	merged := []map[string]any{
		{"role": "user", "content": "List the files, then summarise them."},
		{"role": "assistant", "content": []map[string]any{{"type": "tool_use", "id": "toolu_1", "name": "terminal", "input": map[string]any{"command": "ls"}}}},
		{"role": "user", "content": []map[string]any{
			{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"},
			{"type": "text", "text": hermesIterationSummary},
		}},
	}
	body, _ = json.Marshal(map[string]any{"model": "m", "max_tokens": 1024, "messages": merged})
	if !parseWireRequest(llmprotocol.AnthropicMessagesV1, body).isSidecall() {
		t.Fatal("Hermes summary merged with a tool result was not a side call")
	}
	// Two tool results and the marker is the same shape.
	two := append([]map[string]any{}, merged[:2]...)
	two = append(two, map[string]any{"role": "user", "content": []map[string]any{
		{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"},
		{"type": "tool_result", "tool_use_id": "toolu_2", "content": "b.txt"},
		{"type": "text", "text": hermesIterationSummary},
	}})
	body, _ = json.Marshal(map[string]any{"model": "m", "max_tokens": 1024, "messages": two})
	if !parseWireRequest(llmprotocol.AnthropicMessagesV1, body).isSidecall() {
		t.Fatal("Hermes summary merged with two tool results was not a side call")
	}
	// A tool result, an image and the marker is a multimodal turn.
	withImage := append([]map[string]any{}, merged[:2]...)
	withImage = append(withImage, map[string]any{"role": "user", "content": []map[string]any{
		{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"},
		{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
		{"type": "text", "text": hermesIterationSummary},
	}})
	body, _ = json.Marshal(map[string]any{"model": "m", "max_tokens": 1024, "messages": withImage})
	if parseWireRequest(llmprotocol.AnthropicMessagesV1, body).isSidecall() {
		t.Fatal("a tool result with an image and the literal was a side call")
	}
}
