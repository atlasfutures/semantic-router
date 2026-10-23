package execution

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

const testCoreToken = "execution-test-token-00000000000000000000"

func publicControl(t *testing.T, id string) json.RawMessage {
	t.Helper()
	raw, err := os.ReadFile("../thinkingcontrol/thinking_controls.compiled.json")
	if err != nil {
		t.Fatal(err)
	}
	var registry struct {
		Controls map[string]json.RawMessage `json:"controls"`
	}
	if err := json.Unmarshal(raw, &registry); err != nil {
		t.Fatal(err)
	}
	control, ok := registry.Controls[id]
	if !ok {
		t.Fatal("test control is absent from the pinned public registry")
	}
	return control
}

func testConfig(t *testing.T, endpoint string, privateAppend bool) Config {
	t.Helper()
	controlID := "628d285537ca7cdb52c4bd3433cea7a1ec36005a14dbc67ffa1c40487d7b8ad0"
	provider := "openai"
	protocol := llmprotocol.OpenAIChatV1
	if privateAppend {
		controlID = "605a61ada934f704248cb99cc84179bf15747302120d11e67ba723db0d8d955b"
		provider = "openrouter"
		protocol = llmprotocol.AnthropicMessagesV1
	}
	return Config{ContractVersion: 1, PackageSHA256: strings.Repeat("a", 64), RegistrySHA256: thinkingcontrol.EmbeddedSHA256, AllowExperimentalControls: privateAppend,
		Decisions:   DecisionsConfig{AuthEnv: "TEST_DECISIONS_KEY", Model: "arc-1.0", Result: "top", Routing: json.RawMessage(`{}`), OfferSHA256: strings.Repeat("b", 64)},
		Deployments: []Deployment{{ID: "fixture-worker", Kind: "local", URL: endpoint, Protocol: protocol, WireModel: "qwen-fixture", RegistryModel: "qwen/qwen3.8-27b", RegistryProvider: provider, RuntimeIdentity: "external-http-fixture-v1", MaxOutputTokens: 8192, TokenUnits: "provider:fixture-worker"}},
		Actions:     []Action{{ID: strings.Repeat("c", 64), Worker: "fixture-qwen", Model: "qwen3.8-27b", DeploymentID: "fixture-worker", ControlID: controlID, Control: publicControl(t, controlID)}}}
}

type decisionCapture struct {
	mu       sync.Mutex
	contexts []externalContext
	step     int
}

func executorWithHTTPDecisions(t *testing.T, config Config, capture *decisionCapture) (*Server, *httptest.Server) {
	t.Helper()
	t.Setenv("TEST_DECISIONS_KEY", "fixture-hosted-key-never-for-local")
	decisions := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer fixture-hosted-key-never-for-local" {
			t.Error("Decisions credential missing")
		}
		var body struct {
			Routing struct {
				Execution externalContext `json:"execution"`
			} `json:"routing"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			http.Error(w, "bad request", 400)
			return
		}
		capture.mu.Lock()
		capture.contexts = append(capture.contexts, body.Routing.Execution)
		index := capture.step
		if index >= len(config.Actions) {
			index = len(config.Actions) - 1
		}
		capture.step++
		capture.mu.Unlock()
		action := config.Actions[index]
		// The external fixture reports conservative policy defaults, never measured warmth.
		heldModel := ""
		if hold, ok := body.Routing.Execution.Hold.(map[string]any); ok {
			for _, candidate := range config.Actions {
				if candidate.ID == hold["action_id"] {
					heldModel = candidate.Model
				}
			}
		}
		workerEchoes := make([]workerEcho, 0, len(body.Routing.Execution.Workers))
		for _, worker := range body.Routing.Execution.Workers {
			echo := defaultWorkerEcho()
			echo.ActionID, echo.Worker = worker.ActionID, worker.Worker
			for _, candidate := range config.Actions {
				if candidate.ID == worker.ActionID && candidate.Model == heldModel {
					echo.Held, echo.Pricing, echo.PWarm = echoPtr(true), "default_held_warm", echoPtr(1.0)
				}
			}
			admission, prefixAdmitted, _ := expectedAdmission(worker)
			for field, reason := range admission {
				echo.NotAdmitted = append(echo.NotAdmitted, notAdmittedEcho{Field: field, Reason: reason})
			}
			if prefixAdmitted {
				echo.PrefixTokens, echo.PrefixBasis = echoPtr(worker.CurrentPrefix.Tokens), echoPtr("worker_count")
			}
			workerEchoes = append(workerEchoes, echo)
		}
		writeJSON(w, 200, map[string]any{"route_id": fmt.Sprintf("route-%d", index), "model": action.Model, "worker": action.Worker, "policy_action_id": action.ID,
			"identity": map[string]any{"package_sha256": config.PackageSHA256, "action_id": action.ID, "policy_action_id": action.ID, "control_id": action.ControlID, "arm_id": nil, "thinking_controls_sha256": config.RegistrySHA256, "offer_sha256": config.Decisions.OfferSHA256, "prices": map[string]any{"basis": "catalog", "price_sha256": nil, "exclusive": false}},
			"routing":  map[string]any{"execution": map[string]any{"status": "applied", "turn": body.Routing.Execution.Turn, "pinned": map[string]string{"package_sha256": "matched", "thinking_controls_sha256": "matched", "offer_sha256": "matched"}, "boundary": "user_message", "user_turn_cache": "repartition_multiwarm", "user_turn_cache_source": "service_config", "episode_identity": "per_lookup_random", "multiwarm": true, "workers": workerEchoes}, "unapplied_keys": []string{}}})
	}))
	t.Cleanup(decisions.Close)
	config.Decisions.URL = decisions.URL
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	server, err := NewServer(config, registry, provider, testCoreToken)
	if err != nil {
		t.Fatal(err)
	}
	httpServer := httptest.NewServer(server)
	t.Cleanup(httpServer.Close)
	return server, httpServer
}

func executeHTTP(t *testing.T, url, path, body, session string) (int, []byte) {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, url+path, strings.NewReader(body))
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
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	return response.StatusCode, raw
}

func TestExecutionUsesSharedCodecAcrossThreeIngressFormats(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, path := range []string{"/v1/messages", "/v1/chat/completions", "/v1/responses"} {
			t.Run(fmt.Sprintf("%s/stream=%v", path, stream), func(t *testing.T) {
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Header.Get("Authorization") != "" {
						t.Error("hosted credential reached external inference")
					}
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					if body["model"] != "qwen-fixture" {
						t.Error("worker alias was not applied")
					}
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = io.WriteString(w, "data: {\"id\":\"chat1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen-fixture\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ready\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chat1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen-fixture\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":1,\"total_tokens\":21}}\n\ndata: [DONE]\n\n")
					} else {
						_, _ = io.WriteString(w, `{"id":"chat1","object":"chat.completion","created":1,"model":"qwen-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}`)
					}
				}))
				defer upstream.Close()
				capture := &decisionCapture{}
				server, httpServer := executorWithHTTPDecisions(t, testConfig(t, upstream.URL, false), capture)
				body := fmt.Sprintf(`{"model":"arc-1.0","max_tokens":64,"stream":%v,"messages":[{"role":"user","content":"hello"}]}`, stream)
				if path == "/v1/responses" {
					body = fmt.Sprintf(`{"model":"arc-1.0","max_output_tokens":64,"stream":%v,"input":[{"role":"user","content":"hello"}]}`, stream)
				}
				status, response := executeHTTP(t, httpServer.URL, path, body, "session-matrix")
				if status != 200 || !bytes.Contains(response, []byte("ready")) {
					t.Fatalf("translation failed: status %d", status)
				}
				state, err := server.states.acquire(t.Context(), "session-matrix", "main")
				if err != nil {
					t.Fatal(err)
				}
				defer state.release()
				if state.completed != 1 || state.workers["fixture-worker"].Call.PromptTokens == nil {
					t.Error("complete response did not commit provider cache evidence")
				}
			})
		}
	}
}

func TestPrivateAppendRetryAndReturnToNeutralCommitOnce(t *testing.T) {
	var mu sync.Mutex
	var sent [][]byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		sent = append(sent, body)
		index := len(sent)
		mu.Unlock()
		if index == 1 {
			w.WriteHeader(503)
			_, _ = io.WriteString(w, `{"type":"error","error":{"type":"overloaded_error","message":"fixture unavailable"}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"msg_fixture","type":"message","role":"assistant","model":"qwen-fixture","content":[{"type":"text","text":"ready"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":30,"output_tokens":1}}`)
	}))
	defer upstream.Close()
	config := testConfig(t, upstream.URL, true)
	neutral := config.Actions[0]
	neutral.ID = strings.Repeat("d", 64)
	neutral.ControlID = "17ce2fd14e762777a95db605228cc8f731d2d3e200c4c5d07a735f85fca2eae1"
	neutral.Control = publicControl(t, neutral.ControlID)
	config.Actions = append(config.Actions, neutral)
	capture := &decisionCapture{}
	server, httpServer := executorWithHTTPDecisions(t, config, capture)
	now := time.Unix(1700000000, 0)
	server.now = func() time.Time { return now }
	server.provider.(*HTTPDecisions).now = func() time.Time { return now.Add(7 * time.Second) }
	body := `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"first"}]}`
	if status, _ := executeHTTP(t, httpServer.URL, "/v1/messages", body, "private-append"); status != 503 {
		t.Fatalf("expected visible upstream error, got %d", status)
	}
	if status, _ := executeHTTP(t, httpServer.URL, "/v1/messages", body, "private-append"); status != 200 {
		t.Fatalf("retry failed: %d", status)
	}
	second := `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"first"},{"role":"assistant","content":"ready"},{"role":"user","content":"second"}]}`
	if status, _ := executeHTTP(t, httpServer.URL, "/v1/messages", second, "private-append"); status != 200 {
		t.Fatalf("neutral turn failed: %d", status)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(sent) != 3 || !bytes.Equal(sent[0], sent[1]) {
		t.Fatal("retry did not reproduce exactly the staged provider request")
	}
	control, err := thinkingcontrol.ParseControl(config.Actions[0].Control)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(sent[2]), control.Instruction.Text) != 1 || !bytes.Contains(sent[2], []byte(control.Instruction.NeutralText)) {
		t.Error("committed private append history or neutral restoration was lost")
	}
	capture.mu.Lock()
	defer capture.mu.Unlock()
	if len(capture.contexts) != 2 {
		t.Fatalf("retry consulted Decisions again: %d decisions", len(capture.contexts))
	}
	workers := capture.contexts[1].Workers
	if len(workers) != 1 || workers[0].Gap == nil || *workers[0].Gap != 7 {
		t.Error("Decisions did not receive the last successful completion clock")
	}
	if workers[0].TTL != nil {
		t.Error("non-Anthropic runtime inherited an Anthropic cache lifetime")
	}
}

func TestTruncatedStreamCannotCommitOrRegenerateObservedOutput(t *testing.T) {
	var calls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"chat1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen-fixture\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\n")
	}))
	defer upstream.Close()
	server, httpServer := executorWithHTTPDecisions(t, testConfig(t, upstream.URL, false), &decisionCapture{})
	body := `{"model":"arc-1.0","stream":true,"messages":[{"role":"user","content":"hello"}]}`
	_, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", body, "truncated")
	if !bytes.Contains(response, []byte("partial")) {
		t.Fatal("fixture did not exercise observed output")
	}
	status, _ := executeHTTP(t, httpServer.URL, "/v1/chat/completions", body, "truncated")
	if status != 409 || calls != 1 {
		t.Errorf("partial generation was retried: status %d calls %d", status, calls)
	}
	state, err := server.states.acquire(t.Context(), "truncated", "main")
	if err != nil {
		t.Fatal(err)
	}
	defer state.release()
	if state.completed != 0 || len(state.workers) != 0 || len(state.placements) != 0 {
		t.Error("failed stream committed success state")
	}
}
