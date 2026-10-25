package execution

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

const signalsBlock = `{"topic.operation":{"schema":"rayline.signal.topic.operation.v1","scopes":{"latest_user":{"top":"code_change","unknown":false}}}}`

func executeWithHeaders(t *testing.T, url, body, session string) (int, http.Header, []byte) {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, url+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+testCoreToken)
	request.Header.Set("X-Rayline-Session-Id", session)
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	raw, _ := io.ReadAll(response.Body)
	return response.StatusCode, response.Header, raw
}

// signalsSetup runs a derived v2 profile whose fixture service answers a
// signals block when (and only when) routing.signals was requested.
func signalsSetup(t *testing.T) (*v2Decisions, *upstreamFixture, string) {
	t.Helper()
	local := &upstreamFixture{}
	localURL := chatUpstream(t, local, false).URL + "/v1/chat/completions"
	config, _, err := LoadConfig(writeDerivedConfig(t, func(config map[string]any) {
		config["deployments"].([]any)[0].(map[string]any)["url"] = localURL
	}))
	if err != nil {
		t.Fatal(err)
	}
	qwen := "d6936db7073d3f30c663d504b7acbb124e8f752a9a766cfb5e3129050e26e0a6"
	fixture := &v2Decisions{choose: func(contextV2) string { return qwen }}
	fixture.mutate = func(response map[string]any) {
		lookups := fixture.snapshot()
		var routing map[string]json.RawMessage
		_ = json.Unmarshal(lookups[len(lookups)-1].Raw["routing"], &routing)
		if _, requested := routing["signals"]; requested {
			response["signals"] = json.RawMessage(signalsBlock)
		}
	}
	_, httpServer := v2Server(t, config, fixture)
	return fixture, local, httpServer.URL
}

func TestRoutingSignalsPassThroughToDecisionsAndReturnInHeader(t *testing.T) {
	fixture, local, url := signalsSetup(t)
	body := `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"hi"}],"routing":{"signals": ["topic.operation", "topic.domain"]}}`
	status, header, response := executeWithHeaders(t, url, body, "signals")
	if status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	var routing map[string]json.RawMessage
	_ = json.Unmarshal(fixture.snapshot()[0].Raw["routing"], &routing)
	// The caller's names reach Decisions unchanged (as compact JSON), beside the core's routing.
	if string(routing["signals"]) != `["topic.operation","topic.domain"]` || routing["policy"] == nil || routing["execution"] == nil {
		t.Fatalf("Decisions routing %s", fixture.snapshot()[0].Raw["routing"])
	}
	if got := header.Get(signalsHeader); got != signalsBlock {
		t.Fatalf("x-vsr-signals %q", got)
	}
	// Signals are Decisions metadata: never sent to the generation endpoint.
	if local.count() != 1 || local.bodies[0]["routing"] != nil {
		t.Fatalf("upstream received routing: %v", local.bodies)
	}
}

func TestRequestsWithoutSignalsAreUnchanged(t *testing.T) {
	body := []byte(`{"model":"arc-1.0", "messages":[{"role":"user","content":"routing"}]}`)
	stripped, signals, err := extractRoutingSignals(body)
	if err != nil || signals != nil || !bytes.Equal(stripped, body) || &stripped[0] != &body[0] {
		t.Fatal("a body without routing.signals was rewritten")
	}
	fixture, local, url := signalsSetup(t)
	status, header, response := executeWithHeaders(t, url, string(body), "plain")
	if status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	var routing map[string]json.RawMessage
	_ = json.Unmarshal(fixture.snapshot()[0].Raw["routing"], &routing)
	if _, exists := routing["signals"]; exists || header.Get(signalsHeader) != "" || local.count() != 1 {
		t.Fatalf("signals appeared without a request: routing %s header %q", fixture.snapshot()[0].Raw["routing"], header.Get(signalsHeader))
	}
}

func TestRoutingSignalsValidation(t *testing.T) {
	for name, routing := range map[string]string{
		"not an array": `{"signals":"topic.operation"}`,
		"null":         `{"signals":null}`,
		"empty name":   `{"signals":[""]}`,
		"non-string":   `{"signals":[1]}`,
	} {
		t.Run(name, func(t *testing.T) {
			_, _, url := signalsSetup(t)
			status, _, response := executeWithHeaders(t, url, `{"model":"arc-1.0","messages":[{"role":"user","content":"hi"}],"routing":`+routing+`}`, "bad")
			if status != 400 || !strings.Contains(string(response), "invalid_routing_signals") {
				t.Fatalf("status %d: %s", status, response)
			}
		})
	}
	// Any other per-request routing preference is still refused as before.
	fixture, local, url := signalsSetup(t)
	status, _, _ := executeWithHeaders(t, url, `{"model":"arc-1.0","messages":[{"role":"user","content":"hi"}],"routing":{"signals":[],"value_usd":2}}`, "other")
	if status != http.StatusBadGateway || len(fixture.snapshot()) != 0 || local.count() != 0 {
		t.Fatalf("per-request routing preference accepted: %d", status)
	}
	// An empty list is passed through unchanged too (the service answers none).
	_, signals, err := extractRoutingSignals([]byte(`{"routing":{"signals":[]}}`))
	if err != nil || string(signals) != `[]` {
		t.Fatalf("empty signals %q %v", signals, err)
	}
}

func TestConfiguredRoutingSignalsAreAccepted(t *testing.T) {
	for _, mode := range []string{"execution_context_v2", "current"} {
		config, _, err := LoadConfig(writeDerivedConfig(t, nil))
		if err != nil {
			t.Fatal(err)
		}
		config.Decisions.Routing = json.RawMessage(`{"policy":{"preset":"arc-policy.hybrid-default.v1"},"signals":["topic.domain"]}`)
		if mode == "current" {
			config.Decisions.APIMode = mode
			config.Decisions.Routing = json.RawMessage(`{"signals":["topic.domain"]}`)
		}
		t.Setenv("TEST_DECISIONS_KEY", "fixture")
		if _, err := NewHTTPDecisions(config); err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		config.Decisions.Routing = json.RawMessage(`{"policy":{"preset":"arc-policy.hybrid-default.v1"},"signals":"topic.domain"}`)
		if _, err := NewHTTPDecisions(config); err == nil {
			t.Fatalf("%s: non-array signals accepted", mode)
		}
	}
}

// A retry naming other signals is a different request: Decisions is consulted
// again rather than the pending turn's signals reused.
func TestRetryWithOtherSignalsConsultsAgain(t *testing.T) {
	fixture, local, url := signalsSetup(t)
	local.fail = 1
	base := `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"hi"}],"routing":{"signals":`
	if status, _, _ := executeWithHeaders(t, url, base+`["topic.domain"]}}`, "retry"); status == 200 {
		t.Fatal("the failed upstream attempt succeeded")
	}
	status, header, response := executeWithHeaders(t, url, base+`["topic.operation"]}}`, "retry")
	if status != 200 || header.Get(signalsHeader) == "" {
		t.Fatalf("status %d: %s", status, response)
	}
	lookups := fixture.snapshot()
	var routing map[string]json.RawMessage
	_ = json.Unmarshal(lookups[len(lookups)-1].Raw["routing"], &routing)
	if len(lookups) != 2 || string(routing["signals"]) != `["topic.operation"]` {
		t.Fatalf("lookups %d, last signals %s", len(lookups), routing["signals"])
	}
}
