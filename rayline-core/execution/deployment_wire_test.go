package execution

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

func TestApprovedCloudWireSettings(t *testing.T) {
	deployment := Deployment{Kind: "cloud", RegistryProvider: "openrouter", Protocol: llmprotocol.OpenAIResponsesV1,
		ProviderPin: &ProviderPin{Only: []string{"together"}}}
	if err := validateProviderPin(deployment); err != nil {
		t.Fatal(err)
	}
	body, err := applyDeploymentWireBody([]byte(`{"model":"approved","input":"hi","store":true,"session_id":"private","prompt_cache_key":"private","reasoning":{"effort":"high"}}`), deployment)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire["store"] != false || wire["session_id"] != nil || wire["prompt_cache_key"] != nil || wire["reasoning"] == nil {
		t.Fatal("cloud transport policy was not applied without disturbing the control")
	}
	pin := wire["provider"].(map[string]any)
	if pin["allow_fallbacks"] != false || pin["only"].([]any)[0] != "together" {
		t.Fatal("provider selection changed")
	}
	for _, bad := range []*ProviderPin{nil, {Only: []string{"one", "two"}}, {Only: []string{"one"}, AllowFallbacks: true}} {
		deployment.ProviderPin = bad
		if validateProviderPin(deployment) == nil {
			t.Fatal("ambiguous provider admitted")
		}
	}
}

func TestLocalCredentialAliasCannotExposeHostedKey(t *testing.T) {
	t.Setenv("TEST_DECISIONS_KEY", "synthetic-hosted-secret")
	t.Setenv("TEST_LOCAL_ALIAS", "synthetic-hosted-secret")
	config := testConfig(t, "http://127.0.0.1:1/v1/chat/completions", false)
	config.Deployments[0].AuthEnv = "TEST_LOCAL_ALIAS"
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	_, err = NewServer(config, registry, &HTTPDecisions{}, testCoreToken)
	if err == nil || !strings.Contains(err.Error(), "hosted Decisions credential") {
		t.Fatalf("hosted key alias was admitted for local inference: %v", err)
	}
}

func TestUnknownHistoricalSessionCannotLosePrivateLedger(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("untracked history reached inference") }))
	defer upstream.Close()
	capture := &decisionCapture{}
	_, server := executorWithHTTPDecisions(t, testConfig(t, upstream.URL, false), capture)
	for _, path := range []string{"/v1/messages", "/v1/chat/completions", "/v1/responses"} {
		body := `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"old"},{"role":"assistant","content":"old reply"},{"role":"user","content":"continue"}]}`
		if path == "/v1/responses" {
			body = strings.ReplaceAll(strings.ReplaceAll(body, "max_tokens", "max_output_tokens"), "messages", "input")
		}
		status, result := executeHTTP(t, server.URL, path, body, path)
		if status != 409 || !strings.Contains(string(result), "no execution ledger") {
			t.Fatalf("history resume returned %d: %s", status, result)
		}
	}
	if len(capture.contexts) != 0 {
		t.Fatal("untracked history spent a Decisions call")
	}
}
