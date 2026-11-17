package execution

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// tokenCounterFixture is a same-host counter answering in fixed units with a
// fixed count, recording what it was asked to count.
type tokenCounterFixture struct {
	mu     sync.Mutex
	units  string
	tokens int
	bodies [][]byte
	auth   []string
}

func (fixture *tokenCounterFixture) serve(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fixture.mu.Lock()
		defer fixture.mu.Unlock()
		fixture.auth = append(fixture.auth, r.Header.Get("Authorization"))
		if r.Method == http.MethodGet {
			writeJSON(w, 200, map[string]any{"units": fixture.units})
			return
		}
		body, _ := io.ReadAll(r.Body)
		fixture.bodies = append(fixture.bodies, body)
		writeJSON(w, 200, map[string]any{"prompt_tokens": fixture.tokens, "units": fixture.units})
	}))
	t.Cleanup(server.Close)
	return server
}

func (fixture *tokenCounterFixture) counted() [][]byte {
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	return append([][]byte(nil), fixture.bodies...)
}

const counterUnits = "hf-tokenizer:0123456789abcdef"

func countedLocalConfig(t *testing.T, counterURL string) (Config, *upstreamFixture) {
	t.Helper()
	local := &upstreamFixture{}
	cloud := &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL)
	config.Deployments[0].TokenUnits = counterUnits
	config.Deployments[0].InputModalities = []string{"text", "image"}
	config.Deployments[0].TokenCounter = &TokenCounterConfig{URL: counterURL, AuthEnv: "TEST_TOKEN_COUNTER_KEY"}
	t.Setenv("TEST_TOKEN_COUNTER_KEY", "counter-secret")
	t.Setenv("TEST_COUNTER_ALIAS", "fixture-hosted-key-never-for-local")
	return config, local
}

func TestTokenCounterMeasuresTheLocalTargetWithItsImages(t *testing.T) {
	counter := &tokenCounterFixture{units: counterUnits, tokens: 700}
	config, _ := countedLocalConfig(t, counter.serve(t).URL+"/count")
	fixture := &v2Decisions{choose: fitPolicy}
	server, httpServer := v2Server(t, config, fixture)
	if err := server.RegisterConfiguredPromptCounters(t.Context()); err != nil {
		t.Fatal(err)
	}
	png := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	body := `{"model":"arc-1.0","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"what is this"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + png + `"}}]}]}`
	status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", body, "counted-image")
	if status != http.StatusOK {
		t.Fatalf("status %d: %.300s", status, response)
	}
	lookups := fixture.snapshot()
	if len(lookups) == 0 {
		t.Fatal("no decision was asked")
	}
	var local *CapacityFact
	for _, target := range lookups[0].Execution.Targets {
		if target.Placement == "local" {
			local = target.Capacity
		}
	}
	if local == nil || !local.Fits || local.Provenance != capacityProvenanceMeasured || local.Units == nil || *local.Units != counterUnits || local.Required == nil || *local.Required != 700+100 {
		t.Fatalf("local capacity %+v", local)
	}
	bodies := counter.counted()
	if len(bodies) == 0 || !strings.Contains(string(bodies[0]), `"image_url"`) || !strings.Contains(string(bodies[0]), png) {
		t.Fatalf("the counter did not get the prepared Chat request with its image: %.300s", bodies)
	}
	for _, header := range counter.auth {
		if header != "Bearer counter-secret" {
			t.Fatalf("counter credential %q", header)
		}
	}
}

func TestTokenCounterOverTheWindowExcludesLocalAsMeasured(t *testing.T) {
	counter := &tokenCounterFixture{units: counterUnits, tokens: 950}
	config, local := countedLocalConfig(t, counter.serve(t).URL)
	fixture := &v2Decisions{choose: fitPolicy}
	server, httpServer := v2Server(t, config, fixture)
	if err := server.RegisterConfiguredPromptCounters(t.Context()); err != nil {
		t.Fatal(err)
	}
	status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", `{"model":"arc-1.0","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`, "counted-over")
	if status != http.StatusOK {
		t.Fatalf("status %d: %.300s", status, response)
	}
	for _, target := range fixture.snapshot()[0].Execution.Targets {
		if target.Placement == "local" && (target.Capacity == nil || target.Capacity.Fits || target.Capacity.Provenance != capacityProvenanceMeasured) {
			t.Fatalf("local capacity %+v", target.Capacity)
		}
	}
	if local.count() != 0 {
		t.Fatal("an over-window prompt reached the local runtime")
	}
}

func TestTokenCounterQualification(t *testing.T) {
	counter := &tokenCounterFixture{units: counterUnits, tokens: 1}
	url := counter.serve(t).URL
	base, _ := countedLocalConfig(t, url)
	cases := map[string]func(*Deployment){
		"units differ":       func(d *Deployment) { d.TokenUnits = "hf-tokenizer:other" },
		"remote host":        func(d *Deployment) { d.TokenCounter.URL = "http://counter.example:9/count" },
		"https":              func(d *Deployment) { d.TokenCounter.URL = strings.Replace(url, "http:", "https:", 1) },
		"credential unset":   func(d *Deployment) { d.TokenCounter.AuthEnv = "TEST_TOKEN_COUNTER_UNSET" },
		"two counters":       func(d *Deployment) { d.PromptCounter = &LlamaCPPConfig{} },
		"no units":           func(d *Deployment) { d.TokenUnits = "" },
		"credentials in url": func(d *Deployment) { d.TokenCounter.URL = strings.Replace(url, "http://", "http://user:pass@", 1) },
		"decisions key":      func(d *Deployment) { d.TokenCounter.AuthEnv = "TEST_DECISIONS_KEY" },
		"hosted key alias":   func(d *Deployment) { d.TokenCounter.AuthEnv = "TEST_COUNTER_ALIAS" },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			config := base
			config.Deployments = append([]Deployment(nil), base.Deployments...)
			deployment := config.Deployments[0]
			tokenCounter := *deployment.TokenCounter
			deployment.TokenCounter = &tokenCounter
			mutate(&deployment)
			config.Deployments[0] = deployment
			server, _ := v2Server(t, config, &v2Decisions{choose: fitPolicy})
			if err := server.RegisterConfiguredPromptCounters(t.Context()); err == nil {
				t.Fatal("qualified")
			}
		})
	}
}

func TestTokenCounterRefusesAChangedUnitOrAMissingCount(t *testing.T) {
	answers := []string{`{"units":"` + counterUnits + `"}`, `{"prompt_tokens":5,"units":"hf-tokenizer:other"}`, `{"units":"` + counterUnits + `"}`}
	var mu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		answer := answers[0]
		answers = answers[1:]
		_, _ = io.WriteString(w, answer)
	}))
	t.Cleanup(server.Close)
	deployment := Deployment{ID: "local", Kind: "local", Protocol: "openai.chat.v1", TokenUnits: counterUnits, RuntimeIdentity: "r", WireModel: "m", TokenCounter: &TokenCounterConfig{URL: server.URL}}
	counter, err := newHTTPCounter(t.Context(), deployment, Config{Deployments: []Deployment{deployment}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(map[string]any{"messages": []any{}})
	if _, err := counter.Count(t.Context(), cachecontext.PreparedPrompt{Identity: counter.Identity(), ContextEpoch: "e1", Body: body}); err == nil || !strings.Contains(err.Error(), "units") {
		t.Fatalf("changed units: %v", err)
	}
	if _, err := counter.Count(t.Context(), cachecontext.PreparedPrompt{Identity: counter.Identity(), ContextEpoch: "e1", Body: body}); err == nil || !strings.Contains(err.Error(), "prompt_tokens") {
		t.Fatalf("missing count: %v", err)
	}
}

func TestTokenCounterURLErrorsNeverQuoteTheURL(t *testing.T) {
	deployment := Deployment{ID: "local", Kind: "local", Protocol: "openai.chat.v1", TokenUnits: counterUnits, TokenCounter: &TokenCounterConfig{URL: "http://user:synthetic-secret@127.0.0.1:bad/count"}}
	_, err := newHTTPCounter(t.Context(), deployment, Config{Deployments: []Deployment{deployment}}, nil)
	if err == nil || strings.Contains(err.Error(), "synthetic-secret") {
		t.Fatalf("error %v", err)
	}
}

// A counter that fails after a local turn leaves the prefix unknown and the
// local target unavailable; the turn is decided again, not refused.
func TestTokenCounterFailureAfterALocalTurnHandsOff(t *testing.T) {
	counter := &tokenCounterFixture{units: counterUnits, tokens: 100}
	config, _ := countedLocalConfig(t, counter.serve(t).URL)
	fixture := &v2Decisions{choose: fitPolicy}
	server, httpServer := v2Server(t, config, fixture)
	if err := server.RegisterConfiguredPromptCounters(t.Context()); err != nil {
		t.Fatal(err)
	}
	first := `{"model":"arc-1.0","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", first, "counter-fails"); status != http.StatusOK {
		t.Fatalf("first turn %d: %.300s", status, response)
	}
	counter.mu.Lock()
	counter.units = "hf-tokenizer:changed"
	counter.mu.Unlock()
	next := `{"model":"arc-1.0","max_tokens":100,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"again"}]}`
	status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", next, "counter-fails")
	if status != http.StatusOK {
		t.Fatalf("second turn %d: %.300s", status, response)
	}
	lookups := fixture.snapshot()
	for _, target := range lookups[len(lookups)-1].Execution.Targets {
		if target.Placement == "local" && (target.Capacity == nil || target.Capacity.Fits || target.Capacity.Reason == nil || *target.Capacity.Reason != CapacityRuntimeUnavailable) {
			t.Fatalf("local capacity %+v", target.Capacity)
		}
	}
}

// After a local turn, a request the local worker cannot carry (an output
// allowance over its limit) is decided again rather than refused.
func TestTokenCounterWorkerThatCannotPrepareLeavesThePrefixUnknown(t *testing.T) {
	counter := &tokenCounterFixture{units: counterUnits, tokens: 100}
	config, _ := countedLocalConfig(t, counter.serve(t).URL)
	// The service prefers local whenever it fits; only the core's facts can
	// steer the second turn to the cloud.
	fixture := &v2Decisions{choose: fitPolicy}
	server, httpServer := v2Server(t, config, fixture)
	if err := server.RegisterConfiguredPromptCounters(t.Context()); err != nil {
		t.Fatal(err)
	}
	first := `{"model":"arc-1.0","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", first, "counter-prepare"); status != http.StatusOK {
		t.Fatalf("first turn %d: %.300s", status, response)
	}
	next := `{"model":"arc-1.0","max_tokens":400,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"again"}]}`
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", next, "counter-prepare"); status != http.StatusOK {
		t.Fatalf("second turn %d: %.300s", status, response)
	}
	if lookups := fixture.snapshot(); len(lookups) != 2 {
		t.Fatalf("the second turn was not decided: %d lookups", len(lookups))
	}
}

func TestTokenCounterTransportErrorsNeverQuoteTheURL(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	listener.Close()
	counter := &httpCounter{url: "http://" + address + "/v1/token-count?private-marker", client: &http.Client{}}
	err = counter.exchange(t.Context(), http.MethodGet, nil, &struct{}{})
	if err == nil || strings.Contains(err.Error(), "private-marker") || strings.Contains(err.Error(), "/v1/token-count") {
		t.Fatalf("error %v", err)
	}
}

// protocolUpstream is a local runtime for one protocol that records what it
// is sent and answers in that protocol.
func protocolUpstream(t *testing.T, protocol llmprotocol.WireFormat, bodies *[]map[string]any, mu *sync.Mutex) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		*bodies = append(*bodies, body)
		mu.Unlock()
		if protocol == llmprotocol.AnthropicMessagesV1 {
			_, _ = io.WriteString(w, messagesReady)
			return
		}
		_, _ = io.WriteString(w, `{"id":"resp1","object":"response","created_at":1,"status":"completed","model":"fixture","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ready","annotations":[]}]}],"usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21}}`)
	}))
	t.Cleanup(server.Close)
	return server
}

// A Messages or Responses turn is counted in its Chat form and sized as the
// count plus translatedCountAllowance: a local target the byte bound shuts
// out is admitted as measured, and the runtime still gets its own protocol.
func TestTokenCounterCountsMessagesAndResponsesInTheChatForm(t *testing.T) {
	prose := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 400)
	billing := "x-anthropic-billing-header: cc_version=2.1.325.382; cc_entrypoint=sdk-cli;"
	for _, tc := range []struct {
		protocol   llmprotocol.WireFormat
		path, body string
		native     string
		// charged is the dropped text the size adds by its bytes.
		charged int
	}{
		{llmprotocol.AnthropicMessagesV1, "/v1/messages", `{"model":"arc-1.0","max_tokens":100,"system":[{"type":"text","text":"` + billing + `"},{"type":"text","text":"Be brief.","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":[{"type":"text","text":"` + prose + `"}]}]}`, "max_tokens", worstCaseText(billing)},
		{llmprotocol.OpenAIResponsesV1, "/v1/responses", `{"model":"arc-1.0","max_output_tokens":100,"instructions":"Be brief.","prompt_cache_key":"k","input":[{"role":"user","content":[{"type":"input_text","text":"` + prose + `"}]}]}`, "input", 0},
	} {
		for _, counted := range []bool{true, false} {
			t.Run(string(tc.protocol)+map[bool]string{true: " counted", false: " byte bound"}[counted], func(t *testing.T) {
				counter := &tokenCounterFixture{units: counterUnits, tokens: 3000}
				config, _ := countedLocalConfig(t, counter.serve(t).URL)
				var (
					mu     sync.Mutex
					bodies []map[string]any
				)
				config.Deployments, config.Actions = config.Deployments[:1], config.Actions[:1]
				config = onProtocol(t, config, tc.protocol)
				config.Deployments[0].URL = protocolUpstream(t, tc.protocol, &bodies, &mu).URL
				config.Deployments[0].ContextWindow = 8000
				if !counted {
					config.Deployments[0].TokenCounter = nil
				}
				fixture := &v2Decisions{choose: fitPolicy}
				server, httpServer := v2Server(t, config, fixture)
				if err := server.RegisterConfiguredPromptCounters(t.Context()); err != nil {
					t.Fatal(err)
				}
				status, response := executeHTTP(t, httpServer.URL, tc.path, tc.body, "translated")
				lookups := fixture.snapshot()
				if !counted {
					// About 19k bytes of prose: the bound exceeds the window, so
					// the turn is refused before Decisions.
					if status != http.StatusBadRequest || len(lookups) != 0 || len(bodies) != 0 {
						t.Fatalf("status %d lookups %d dispatched %d: %.300s", status, len(lookups), len(bodies), response)
					}
					return
				}
				if status != http.StatusOK || len(lookups) != 1 {
					t.Fatalf("status %d lookups %d: %.300s", status, len(lookups), response)
				}
				fact := lookups[0].Execution.Targets[0].Capacity
				// 3000 counted, +150 (5%) +512 allowance, +100 output.
				if want := int64(3762 + tc.charged); fact == nil || !fact.Fits || fact.Provenance != capacityProvenanceMeasured || *fact.Required != want {
					t.Fatalf("fact %s, want required %d", mustJSON(fact), want)
				}
				sent := counter.counted()
				var chat map[string]any
				if len(sent) == 0 || json.Unmarshal(sent[0], &chat) != nil || chat["messages"] == nil || chat["input"] != nil || !strings.Contains(string(sent[0]), "Be brief.") {
					t.Fatalf("the counter did not get the Chat form: %.400s", sent)
				}
				if len(bodies) != 1 || bodies[0][tc.native] == nil || bodies[0]["messages"] != nil && tc.protocol == llmprotocol.OpenAIResponsesV1 {
					t.Fatalf("the runtime did not get its own protocol: %.300s", mustJSON(bodies))
				}
			})
		}
	}
}

// A request whose Chat form would leave out something the runtime may render
// is sized by its bytes, logged with the reason, and still dispatched.
func TestTokenCounterSkipsARequestTheChatFormCannotCarry(t *testing.T) {
	counter := &tokenCounterFixture{units: counterUnits, tokens: 10}
	config, _ := countedLocalConfig(t, counter.serve(t).URL)
	var (
		mu     sync.Mutex
		bodies []map[string]any
	)
	config.Deployments, config.Actions = config.Deployments[:1], config.Actions[:1]
	config = onProtocol(t, config, llmprotocol.AnthropicMessagesV1)
	config.Deployments[0].URL = protocolUpstream(t, llmprotocol.AnthropicMessagesV1, &bodies, &mu).URL
	config.Deployments[0].ContextWindow = 8000
	fixture := &v2Decisions{choose: fitPolicy}
	server, httpServer := v2Server(t, config, fixture)
	if err := server.RegisterConfiguredPromptCounters(t.Context()); err != nil {
		t.Fatal(err)
	}
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	defer slog.SetDefault(previous)
	body := `{"model":"arc-1.0","max_tokens":100,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","media_type":"text/plain","data":"a document"}},{"type":"text","text":"Summarise it."}]}]}`
	status, response := executeHTTP(t, httpServer.URL, "/v1/messages", body, "uncounted")
	if status != http.StatusOK || len(bodies) != 1 {
		t.Fatalf("status %d dispatched %d: %.300s", status, len(bodies), response)
	}
	fact := fixture.snapshot()[0].Execution.Targets[0].Capacity
	if fact == nil || fact.Provenance != capacityProvenanceEstimated {
		t.Fatalf("fact %s", mustJSON(fact))
	}
	if len(counter.counted()) != 0 || !strings.Contains(logs.String(), `"msg":"token counter skipped"`) || !strings.Contains(logs.String(), `the request carries a document block whole`) || strings.Contains(logs.String(), "token counter failed") {
		t.Fatalf("counted %d, logs %s", len(counter.counted()), logs.String())
	}
}

// worstCaseText is what droppedTextBytes charges for text.
func worstCaseText(text string) int {
	quoted, _ := json.Marshal(text)
	return int(worstCaseBytes(quoted))
}

func TestChatFormForCountingRefusesWhatItCannotCarry(t *testing.T) {
	engine, err := newCountingEngine()
	if err != nil {
		t.Fatal(err)
	}
	const (
		messages  = llmprotocol.AnthropicMessagesV1
		responses = llmprotocol.OpenAIResponsesV1
	)
	for _, tc := range []struct {
		name     string
		protocol llmprotocol.WireFormat
		body     string
		// uncounted is the reason the request is not counted; "" counts it.
		uncounted string
		// charged is the dropped text charged by its bytes.
		charged int
	}{
		{"signed thinking and cache markers", messages, `{"model":"m","max_tokens":64,"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral"}}],"context_management":{"edits":[]},"tools":[{"name":"Read","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"thinking","thinking":"t","signature":"x"},{"type":"tool_use","id":"u","name":"Read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"u","content":"ok"}]}]}`, "", 0},
		{"readable reasoning and routing members", responses, `{"model":"m","prompt_cache_key":"k","client_metadata":{"a":"b"},"include":["reasoning.encrypted_content"],"reasoning":{"effort":"high","summary":"auto"},"input":[{"role":"user","content":"hi"},{"type":"reasoning","id":"rs","summary":[{"type":"summary_text","text":"summary-sentinel"}]},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"ok"}],"tools":[{"type":"function","name":"f","parameters":{}}]}`, "", 0},
		{"claude code billing line and compaction instructions", messages, `{"model":"m","max_tokens":64,"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.325.382; cc_entrypoint=sdk-cli;"},{"type":"text","text":"s"}],"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"},{"type":"compact_20260112","instructions":"Summarise the work so far."}]},"messages":[{"role":"user","content":"hi"}]}`, "", worstCaseText("x-anthropic-billing-header: cc_version=2.1.325.382; cc_entrypoint=sdk-cli;") + worstCaseText("all") + worstCaseText("Summarise the work so far.")},
		{"billing line as system and expanding instructions", messages, `{"model":"m","max_tokens":64,"system":"x-anthropic-billing-header: cc_version=2.1.325.382;","context_management":{"edits":[{"type":"compact_20260112","instructions":"` + strings.Repeat("\ufdfa", 100) + `"}]},"messages":[{"role":"user","content":"hi"}]}`, "", worstCaseText("x-anthropic-billing-header: cc_version=2.1.325.382;") + worstCaseText(strings.Repeat(string(rune(0xFDFA)), 100))},
		{"claude code system message with effort", messages, `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"},{"role":"system","content":[{"type":"text","text":"env"}],"output_config":{"effort":"medium"}}]}`, "", 0},
		{"number past float64 beside dropped text", messages, `{"model":"m","max_tokens":64,"context_management":{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":1e400},"instructions":"Summarise."}]},"messages":[{"role":"user","content":"hi"}]}`, "", worstCaseText("Summarise.")},
		{"prose after the billing prefix", messages, `{"model":"m","max_tokens":64,"system":[{"type":"text","text":"x-anthropic-billing-header: is how the next line starts, and this is prose."}],"messages":[{"role":"user","content":"hi"}]}`, "", 0},
		{"output format on a system message", messages, `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"},{"role":"system","content":"env","output_config":{"effort":"low","format":{"type":"json_schema","schema":{"description":"rendered?"}}}}]}`, "a message carries output_config other than an effort", 0},
		{"deferred tool", messages, `{"model":"m","max_tokens":64,"tools":[{"name":"Deferred","input_schema":{"type":"object"},"defer_loading":true}],"messages":[{"role":"user","content":"hi"}]}`, "", 0},
		{"carried system block", messages, `{"model":"m","max_tokens":64,"system":[{"type":"text","text":"s"},{"type":"document","source":{"type":"content","content":[{"type":"text","text":"a long system document"}]}}],"messages":[{"role":"user","content":"hi"}]}`, "the request carries a document block whole", 0},
		{"redacted thinking", messages, `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"redacted_thinking","data":"zzz"},{"type":"text","text":"a"}]},{"role":"user","content":"more"}]}`, "the Chat form drops reasoning", 0},
		{"encrypted reasoning", responses, `{"model":"m","input":[{"role":"user","content":"hi"},{"type":"reasoning","id":"rs","summary":[],"encrypted_content":"x"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"ok"}],"tools":[{"type":"function","name":"f","parameters":{}}]}`, "the request carries a reasoning block whole", 0},
		{"server tool", messages, `{"model":"m","max_tokens":64,"tools":[{"type":"web_search_20250305","name":"web_search"}],"messages":[{"role":"user","content":"hi"}]}`, "the Chat form dropped tools.type", 0},
		{"document", messages, `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","media_type":"text/plain","data":"d"}}]}]}`, "the request carries a document block whole", 0},
		{"no Chat form", messages, `not json`, "no Chat form", 0},
	} {
		chat, charged, err := chatFormForCounting(engine, tc.protocol, []byte(tc.body))
		if strings.Contains(tc.body, "summary-sentinel") && !strings.Contains(string(chat), "summary-sentinel") {
			t.Errorf("%s: the reasoning summary is not in the Chat form: %s", tc.name, chat)
		}
		var uncounted *uncountedRequest
		switch {
		case tc.uncounted == "" && (err != nil || len(chat) == 0 || charged != tc.charged):
			t.Errorf("%s: not counted as expected: charged %d, want %d: %v", tc.name, charged, tc.charged, err)
		case tc.uncounted != "" && (!errors.As(err, &uncounted) || !strings.HasPrefix(uncounted.reason, tc.uncounted)):
			t.Errorf("%s: %v, want %q", tc.name, err, tc.uncounted)
		}
	}
}

// preparedFor is body as the core prepares it for the fixture's local
// deployment on protocol: what a token counter is asked to count.
func preparedFor(t *testing.T, protocol llmprotocol.WireFormat, body []byte) []byte {
	t.Helper()
	config := capacityEstimateConfig(t, "http://127.0.0.1:8/v1/chat/completions", "http://127.0.0.1:9/v1/chat/completions")
	config.Deployments, config.Actions = config.Deployments[:1], config.Actions[:1]
	config = onProtocol(t, config, protocol)
	config.Deployments[0].MaxOutputTokens, config.Deployments[0].ContextWindow = 128000, 1_000_000
	server, _ := v2Server(t, config, &v2Decisions{choose: fitPolicy})
	request, envelope, _, err := server.engine.DecodeRequestForMutation(protocol, body)
	if err != nil {
		t.Fatal(err)
	}
	roles, err := wireMessageRoles(protocol, body)
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := server.prepare(request, envelope, server.actions[protocol][localActionID], roles, &branchState{})
	if err != nil {
		t.Fatal(err)
	}
	return prepared.body
}

// Real Codex and Claude Code turns, prepared for a local deployment as the
// core prepares them, are counted. Codex declares web_search, which the Chat
// form cannot carry, but the core drops it for a local target before the
// counter sees the request.
func TestTokenCounterCountsRealHarnessTurns(t *testing.T) {
	engine, err := newCountingEngine()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		file     string
		protocol llmprotocol.WireFormat
	}{
		{"../protocolcodec/testdata/codex/turn1-request.json", llmprotocol.OpenAIResponsesV1},
		{"../protocolcodec/testdata/codex/turn2-request.json", llmprotocol.OpenAIResponsesV1},
		{"testdata/capacity/codex_responses_turn.json", llmprotocol.OpenAIResponsesV1},
		{"../protocolcodec/testdata/claude-code/tool-turn-request.json", llmprotocol.AnthropicMessagesV1},
		{"testdata/capacity/claude_code_messages_turn.json", llmprotocol.AnthropicMessagesV1},
	} {
		body, err := os.ReadFile(tc.file)
		if err != nil {
			t.Fatal(err)
		}
		prepared := preparedFor(t, tc.protocol, body)
		if strings.Contains(string(body), `"web_search"`) && strings.Contains(string(prepared), `"web_search"`) {
			t.Errorf("%s: the prepared request still declares web_search", tc.file)
		}
		if _, _, err := chatFormForCounting(engine, tc.protocol, prepared); err != nil {
			t.Errorf("%s: %v", tc.file, err)
		}
	}
}

// More deferred tools than the default diagnostics limit are still counted;
// a list that does reach the counting limit is not, since a drop past it
// may go unrecorded.
func TestChatFormForCountingNeedsACompleteDropList(t *testing.T) {
	tools := make([]any, 70)
	for index := range tools {
		tools[index] = map[string]any{"name": fmt.Sprintf("mcp_tool_%d", index), "input_schema": map[string]any{"type": "object"}, "defer_loading": true}
	}
	body := mustJSON(map[string]any{"model": "m", "max_tokens": 64, "tools": tools, "messages": []any{map[string]any{"role": "user", "content": "hi"}}})
	engine, err := newCountingEngine()
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = chatFormForCounting(engine, llmprotocol.AnthropicMessagesV1, []byte(body)); err != nil {
		t.Fatalf("70 deferred tools: %v", err)
	}
	previous := countDiagnostics
	countDiagnostics = 64
	defer func() { countDiagnostics = previous }()
	small, err := newCountingEngine()
	if err != nil {
		t.Fatal(err)
	}
	// 64 neutral drops fill the list; the server tool's drop is then not
	// recorded at all, which only the full list gives away.
	full := map[string]any{"model": "m", "max_tokens": 64, "tools": append(append([]any(nil), tools[:64]...), map[string]any{"type": "web_search_20250305", "name": "web_search"}), "messages": []any{map[string]any{"role": "user", "content": "hi"}}}
	var uncounted *uncountedRequest
	if _, _, err := chatFormForCounting(small, llmprotocol.AnthropicMessagesV1, []byte(mustJSON(full))); !errors.As(err, &uncounted) || !strings.Contains(uncounted.reason, "too many") {
		t.Fatalf("a full drop list was counted: %v", err)
	}
}

// A count that would overflow the allowance is refused, and the reason for
// an uncounted request never quotes the request.
func TestTokenCounterRefusesAnImplausibleCountAndLogsNoRequestText(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			_, _ = io.WriteString(w, `{"units":"`+counterUnits+`"}`)
			return
		}
		_, _ = io.WriteString(w, `{"prompt_tokens":9000000000000000000,"units":"`+counterUnits+`"}`)
	}))
	t.Cleanup(server.Close)
	deployment := Deployment{ID: "local", Kind: "local", Protocol: llmprotocol.AnthropicMessagesV1, TokenUnits: counterUnits, RuntimeIdentity: "r", WireModel: "m", TokenCounter: &TokenCounterConfig{URL: server.URL}}
	if _, err := newHTTPCounter(t.Context(), deployment, Config{Deployments: []Deployment{deployment}}, nil); err == nil {
		t.Fatal("a Messages counter without a codec qualified")
	}
	engine, err := newCountingEngine()
	if err != nil {
		t.Fatal(err)
	}
	counter, err := newHTTPCounter(t.Context(), deployment, Config{Deployments: []Deployment{deployment}}, engine)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	if _, err := counter.Count(t.Context(), cachecontext.PreparedPrompt{Identity: counter.Identity(), ContextEpoch: "e1", Body: body}); err == nil || !strings.Contains(err.Error(), "implausible") {
		t.Fatalf("huge count: %v", err)
	}
	secret := "secret-request-text-" + strings.Repeat("x", 80)
	for _, request := range []string{
		`{"model":"m","max_tokens":64,"` + secret + `":1,"messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","max_tokens":64,"tools":[{"name":"` + secret + `","input_schema":"not an object"}],"messages":[{"role":"user","content":"hi"}]}`,
	} {
		_, _, err := chatFormForCounting(engine, llmprotocol.AnthropicMessagesV1, []byte(request))
		var uncounted *uncountedRequest
		if errors.As(err, &uncounted) && strings.Contains(uncounted.reason, "secret-request-text") {
			t.Errorf("the reason quotes the request: %q", uncounted.reason)
		}
	}
}
