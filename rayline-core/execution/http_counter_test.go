package execution

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
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
	body := `{"model":"arc-1.0","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"what is this"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + png + `"}}]}]}`
	status, response := executeHTTP(t, httpServer.URL, "/v1/messages", body, "counted-image")
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
	status, response := executeHTTP(t, httpServer.URL, "/v1/messages", `{"model":"arc-1.0","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`, "counted-over")
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
	counter, err := newHTTPCounter(t.Context(), deployment, Config{Deployments: []Deployment{deployment}})
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
	_, err := newHTTPCounter(t.Context(), deployment, Config{Deployments: []Deployment{deployment}})
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
	if status, response := executeHTTP(t, httpServer.URL, "/v1/messages", first, "counter-fails"); status != http.StatusOK {
		t.Fatalf("first turn %d: %.300s", status, response)
	}
	counter.mu.Lock()
	counter.units = "hf-tokenizer:changed"
	counter.mu.Unlock()
	next := `{"model":"arc-1.0","max_tokens":100,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"again"}]}`
	status, response := executeHTTP(t, httpServer.URL, "/v1/messages", next, "counter-fails")
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
