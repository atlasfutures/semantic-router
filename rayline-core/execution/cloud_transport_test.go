package execution

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

func modelPinConfig(t *testing.T, endpoint string) Config {
	t.Helper()
	t.Setenv("TEST_MODEL_PIN_KEY", "synthetic-router-key")
	config := testConfig(t, endpoint+"/v1/chat/completions", true)
	deployment := &config.Deployments[0]
	deployment.Kind, deployment.CloudTransport, deployment.AuthEnv = "cloud", cloudModelPin, "TEST_MODEL_PIN_KEY"
	deployment.Protocol = llmprotocol.OpenAIChatV1
	deployment.WireModel, deployment.RegistryModel = "moonshotai/kimi-k3", "moonshotai/kimi-k3"
	config.Actions[0].Model = "kimi-k3"
	return config
}

func modelPinHeaders(header http.Header) {
	header.Set("X-Rayline-Selected-Model", "moonshotai/kimi-k3")
	header.Set("X-Rayline-Decider", "rules")
	header.Set("X-Rayline-Decider-Reason", "pin")
}

func TestModelPinTransportsPrivateAppend(t *testing.T) {
	calls := 0
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer synthetic-router-key" || r.Header.Get("X-Workshop-Required-Model") != "moonshotai/kimi-k3" {
			t.Error("model pin transport identity changed")
		}
		for _, header := range []string{"X-Rayline-Dispatch-Profile", "X-Rayline-Action-Id", "X-Rayline-Worker"} {
			if r.Header.Get(header) != "" {
				t.Errorf("model pin claimed exact receipt header %s", header)
			}
		}
		body, _ := io.ReadAll(r.Body)
		var wire map[string]json.RawMessage
		if err := json.Unmarshal(body, &wire); err != nil {
			t.Fatal(err)
		}
		if string(wire["model"]) != `"moonshotai/kimi-k3"` || wire["provider"] != nil || wire["reasoning"] != nil || wire["reasoning_effort"] != nil {
			t.Errorf("default control or catalog provider ownership changed: %s", body)
		}
		if bytes.Count(body, []byte("reason more thoroughly before acting")) != 1 {
			t.Errorf("private append was absent or duplicated: %s", body)
		}
		modelPinHeaders(w.Header())
		_, _ = io.WriteString(w, `{"id":"chat1","object":"chat.completion","created":1,"model":"moonshotai/kimi-k3","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":40,"completion_tokens":8,"total_tokens":48}}`)
	}))
	defer gateway.Close()
	_, server := executorWithHTTPDecisions(t, modelPinConfig(t, gateway.URL), &decisionCapture{})
	status, answer := executeHTTP(t, server.URL, "/v1/chat/completions", `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}`, "model-pin")
	if status != 200 || !bytes.Contains(answer, []byte("ready")) || bytes.Contains(answer, []byte("reason more thoroughly")) || calls != 1 {
		t.Fatalf("model pin failed: %d %s (calls=%d)", status, answer, calls)
	}
}

func TestModelPinRejectsMissingOrContradictoryReceiptWithoutRetry(t *testing.T) {
	for _, reason := range []string{"", "fallback", "arc_model"} {
		t.Run(fmt.Sprintf("reason=%s", reason), func(t *testing.T) {
			calls := 0
			gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				modelPinHeaders(w.Header())
				w.Header().Set("X-Rayline-Decider-Reason", reason)
				_, _ = io.WriteString(w, `{"choices":[]}`)
			}))
			defer gateway.Close()
			_, server := executorWithHTTPDecisions(t, modelPinConfig(t, gateway.URL), &decisionCapture{})
			status, _ := executeHTTP(t, server.URL, "/v1/chat/completions", `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}`, "receipt")
			if status != 502 || calls != 1 {
				t.Fatalf("invalid receipt accepted or retried: status=%d calls=%d", status, calls)
			}
		})
	}
}

func TestModelPinScopeAndNativeControlAreExplicit(t *testing.T) {
	deployment := modelPinConfig(t, "https://gateway.example").Deployments[0]
	if err := validateCloudTransport(deployment); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*Deployment){
		"transport": func(d *Deployment) { d.CloudTransport = "try_exact_then_pin" },
		"profile":   func(d *Deployment) { d.DispatchProfileSHA256 = strings.Repeat("a", 64) },
		"issuer":    func(d *Deployment) { d.IssuerSHA256 = strings.Repeat("a", 64) },
		"provider":  func(d *Deployment) { d.ProviderPin = &ProviderPin{Only: []string{"together"}} },
		"wire":      func(d *Deployment) { d.WireModel = "" },
		"spaced":    func(d *Deployment) { d.WireModel = "kimi k3" },
		"other":     func(d *Deployment) { d.WireModel = "claude-opus-5-5" },
		"registry":  func(d *Deployment) { d.RegistryModel = "kimi-k3" },
		"responses": func(d *Deployment) { d.Protocol = llmprotocol.OpenAIResponsesV1 },
	} {
		t.Run(name, func(t *testing.T) {
			invalid := deployment
			change(&invalid)
			if validateCloudTransport(invalid) == nil {
				t.Fatal("unqualified model pin admitted")
			}
		})
	}
	if validateModelPinControl(deployment, thinkingcontrol.Control{Native: "xhigh"}) == nil {
		t.Fatal("gateway effort normalization silently accepted")
	}
	if encryptedDeploymentIssuer(deployment) != reasoning.ReasoningIssuerUnknown || opaqueDeploymentIssuer(deployment) != reasoning.ReasoningIssuerUnknown {
		t.Fatal("model receipt was used as provider/account attestation")
	}
	header := make(http.Header)
	modelPinHeaders(header)
	deployment.CloudTransport = cloudExactDispatch
	if dispatchEchoMatches(header, admittedAction{deployment: deployment}, Decision{}) {
		t.Fatal("exact dispatch silently accepted the model-pin receipt")
	}
}

func TestModelPinDropsUnverifiedReasoningDetailsKeepsToolsAndText(t *testing.T) {
	body := []byte(`{"model":"auto","messages":[{"role":"assistant","content":"visible","reasoning":"readable","reasoning_details":[{"type":"reasoning.encrypted","data":"opaque-secret","format":"unknown"}],"tool_calls":[{"id":"call1","type":"function","function":{"name":"Read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call1","content":"result"}]}`)
	engine := protocolcodec.NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	deployment := modelPinConfig(t, "https://gateway.example").Deployments[0]
	prepareReasoning(&request, deployment, &branchState{})
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	for _, preserved := range []string{"visible", "readable", "call1", "result"} {
		if !bytes.Contains(encoded.Body, []byte(preserved)) {
			t.Fatalf("lost %s: %s", preserved, encoded.Body)
		}
	}
	if bytes.Contains(encoded.Body, []byte("opaque-secret")) || bytes.Contains(encoded.Body, []byte("reasoning_details")) {
		t.Fatalf("unknown issuer replayed: %s", encoded.Body)
	}
}

func TestModelPinMessagesControlAndTranscriptBoundaries(t *testing.T) {
	deployment := modelPinConfig(t, "https://gateway.example").Deployments[0]
	deployment.Protocol = llmprotocol.AnthropicMessagesV1
	deployment.URL = "https://gateway.example/v1/messages"
	if err := validateCloudTransport(deployment); err != nil {
		t.Fatal(err)
	}
	for _, native := range []string{"default", "low", "medium", "high", "max"} {
		if err := validateModelPinControl(deployment, thinkingcontrol.Control{Native: native}); err != nil {
			t.Fatal(err)
		}
	}
	// A literal xhigh is forwarded wherever it appears. The control fields are
	// owned by placer.Render upstream of this and never carry a client value at
	// dispatch; elsewhere the string is the client's own data.
	for name, body := range map[string]string{
		"effort":   `{"output_config":{"effort":"xhigh"}}`,
		"thinking": `{"thinking":{"type":"enabled","effort":"xhigh"}}`,
		"schema":   `{"tools":[{"name":"ReportFindings","input_schema":{"properties":{"level":{"enum":["low","high","xhigh","max"]}}}}]}`,
		"prose":    `{"messages":[{"role":"user","content":"the literal xhigh inside prose survives"}]}`,
	} {
		t.Run(name+"Forwarded", func(t *testing.T) {
			kept, err := applyDeploymentWireBody([]byte(body), nil, deployment)
			if err != nil {
				t.Fatalf("client data refused: %v", err)
			}
			if !bytes.Contains(kept, []byte("xhigh")) {
				t.Fatalf("xhigh was not forwarded verbatim: %s", kept)
			}
		})
	}
	// An in-band system row reaches the gateway as the client sent it: the
	// core never folds one, so there is no transcript mutation to refuse.
	forwarded := []byte(`{"messages":[{"role":"user","content":"Read the file."},{"role":"system","content":"later reminder"}]}`)
	body, err := applyDeploymentWireBody(forwarded, nil, deployment)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(body, []byte(`"role":"system"`)) || !bytes.Contains(body, []byte("later reminder")) {
		t.Fatalf("in-band system row was not forwarded verbatim: %s", body)
	}
	// A developer row is refused from the client's own roles. The encoded body
	// cannot carry that role: the decoder canonicalizes it to system, so a
	// body-only check would never fire on a real request.
	canonicalized := []byte(`{"messages":[{"role":"user","content":"hi"},{"role":"system","content":"was developer"}]}`)
	if _, err := applyDeploymentWireBody(canonicalized, []string{"user", "developer"}, deployment); err == nil {
		t.Fatal("developer authority was admitted after canonicalization")
	}
	if _, err := applyDeploymentWireBody(canonicalized, []string{"user", "system"}, deployment); err != nil {
		t.Fatalf("a system row the client actually sent was refused: %v", err)
	}
}

func TestModelPinMessagesAdmitsActualHaikuSteeringAction(t *testing.T) {
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" || r.Header.Get("X-Workshop-Required-Model") != "anthropic/claude-haiku-5.5" {
			t.Error("Messages model pin was not explicit")
		}
		body, _ := io.ReadAll(r.Body)
		if !bytes.Contains(body, []byte("reason more thoroughly before acting")) {
			t.Error("Messages private append absent")
		}
		modelPinHeaders(w.Header())
		w.Header().Set("X-Rayline-Selected-Model", "anthropic/claude-haiku-5.5")
		_, _ = io.WriteString(w, `{"id":"msg_fixture","type":"message","role":"assistant","model":"anthropic/claude-haiku-5.5","content":[{"type":"text","text":"ready"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":30,"output_tokens":1}}`)
	}))
	defer gateway.Close()
	config := modelPinConfig(t, gateway.URL)
	deployment := &config.Deployments[0]
	deployment.URL = gateway.URL + "/v1/messages"
	deployment.Protocol = llmprotocol.AnthropicMessagesV1
	deployment.WireModel, deployment.RegistryModel = "anthropic/claude-haiku-5.5", "anthropic/claude-haiku-5.5"
	config.Actions[0].Model = "claude-haiku-5.5"
	_, server := executorWithHTTPDecisions(t, config, &decisionCapture{})
	status, answer := executeHTTP(t, server.URL, "/v1/messages", `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}`, "haiku")
	if status != 200 || !bytes.Contains(answer, []byte("ready")) {
		t.Fatalf("Messages action failed: %d %s", status, answer)
	}
}

// The gateway may serve the measured model under its own catalog id, here
// Anthropic's claude-sonnet-5-5 for the registry's anthropic/claude-sonnet-5.5:
// the controls come from the registry cell, the pin and its echo from the
// catalog id, and an echo of the registry id is not that pin.
func TestModelPinWireModelMustNameTheRegistryModel(t *testing.T) {
	for _, pin := range []struct {
		wire, registry string
		names          bool
	}{
		{"anthropic/claude-sonnet-5.5", "anthropic/claude-sonnet-5.5", true},
		{"claude-sonnet-5-5", "anthropic/claude-sonnet-5.5", true},
		{"claude-opus-5-5", "anthropic/claude-sonnet-5.5", false},
		{"claude-sonnet-5.5", "anthropic/claude-sonnet-5.5", false},
		{"sonnet", "anthropic/claude-sonnet-5.5", false},
		{"", "anthropic/", false},
	} {
		if pinWireNamesRegistryModel(pin.wire, pin.registry) != pin.names {
			t.Errorf("wire %q for registry %q: names = %v", pin.wire, pin.registry, !pin.names)
		}
	}
}

func TestModelPinMessagesPinsGatewayCatalogId(t *testing.T) {
	for _, echo := range []string{"claude-sonnet-5-5", "anthropic/claude-sonnet-5.5"} {
		t.Run(echo, func(t *testing.T) {
			gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				var wire map[string]json.RawMessage
				if err := json.Unmarshal(body, &wire); err != nil {
					t.Fatal(err)
				}
				if r.Header.Get("X-Workshop-Required-Model") != "claude-sonnet-5-5" || string(wire["model"]) != `"claude-sonnet-5-5"` {
					t.Errorf("catalog id was not pinned: %s %s", r.Header.Get("X-Workshop-Required-Model"), wire["model"])
				}
				if !bytes.Contains(body, []byte("reason more thoroughly before acting")) {
					t.Error("registry cell's private append absent")
				}
				modelPinHeaders(w.Header())
				w.Header().Set("X-Rayline-Selected-Model", echo)
				_, _ = io.WriteString(w, `{"id":"msg_fixture","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[{"type":"text","text":"ready"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":30,"output_tokens":1}}`)
			}))
			defer gateway.Close()
			config := modelPinConfig(t, gateway.URL)
			deployment := &config.Deployments[0]
			deployment.URL = gateway.URL + "/v1/messages"
			deployment.Protocol = llmprotocol.AnthropicMessagesV1
			deployment.WireModel, deployment.RegistryModel = "claude-sonnet-5-5", "anthropic/claude-sonnet-5.5"
			config.Actions[0].Model = "claude-sonnet-5.5"
			_, server := executorWithHTTPDecisions(t, config, &decisionCapture{})
			status, answer := executeHTTP(t, server.URL, "/v1/messages", `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}`, "sonnet")
			if accepted := status == 200 && bytes.Contains(answer, []byte("ready")); accepted != (echo == "claude-sonnet-5-5") {
				t.Fatalf("echo %s: %d %s", echo, status, answer)
			}
		})
	}
}

func TestModelPinSolMessagesPreservesEveryPackageControl(t *testing.T) {
	controls := []struct {
		name, actionID, controlID, effort, instruction string
	}{
		{"medium", "216520e83b7c0a2c1b583320717dc67488b6b2ee76b28f4984c6f9a5b0a3b38d", "e63d9d8f3c66e9181cbdedfc4eb0f58b329595447b4ce11e693944b0b4374fdb", "medium", ""},
		{"none", "4bf076770a2ba20749b0517d35ef991c8a2abfcd23a79337b1051d6240427e2f", "17ce2fd14e762777a95db605228cc8f731d2d3e200c4c5d07a735f85fca2eae1", "", ""},
		{"up", "cd7c76971ab176b35e0aeac6f772d9b320bf0e952307982896a3411aa6d677fb", "605a61ada934f704248cb99cc84179bf15747302120d11e67ba723db0d8d955b", "", "reason more thoroughly before acting"},
		{"down", "d751ccaaabfc7419ff2dcee3ee1fbf5cd5e5dc35dff40bd4c19e48525e2755f5", "77f4674fea6c3b4f3f53ab0337b61fc725ef66e045f317767efdbf94b6979d48", "", "use minimal deliberation"},
		{"default", "fef78dc1956117b24a08e8d9cd5774f41f2154f258d0cccec59e113f5793792c", "628d285537ca7cdb52c4bd3433cea7a1ec36005a14dbc67ffa1c40487d7b8ad0", "", ""},
	}
	for _, control := range controls {
		t.Run(control.name, func(t *testing.T) {
			calls := 0
			gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if r.URL.Path != "/v1/messages" || r.Header.Get("X-Workshop-Required-Model") != "openai/gpt-6.1-sol" {
					t.Error("Sol control reached the wrong concrete model or protocol")
				}
				body, _ := io.ReadAll(r.Body)
				var wire struct {
					Model        string `json:"model"`
					OutputConfig struct {
						Effort string `json:"effort"`
					} `json:"output_config"`
				}
				if err := json.Unmarshal(body, &wire); err != nil {
					t.Fatal(err)
				}
				if wire.Model != "openai/gpt-6.1-sol" || wire.OutputConfig.Effort != control.effort {
					t.Errorf("native control changed: %s", body)
				}
				if control.instruction != "" && bytes.Count(body, []byte(control.instruction)) != 1 {
					t.Errorf("private append absent or duplicated: %s", body)
				}
				modelPinHeaders(w.Header())
				w.Header().Set("X-Rayline-Selected-Model", "openai/gpt-6.1-sol")
				_, _ = io.WriteString(w, `{"id":"msg_sol","type":"message","role":"assistant","model":"openai/gpt-6.1-sol","content":[{"type":"text","text":"ready"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":30,"output_tokens":1}}`)
			}))
			defer gateway.Close()
			config := modelPinConfig(t, gateway.URL)
			deployment := &config.Deployments[0]
			deployment.URL, deployment.Protocol = gateway.URL+"/v1/messages", llmprotocol.AnthropicMessagesV1
			deployment.WireModel, deployment.RegistryModel = "openai/gpt-6.1-sol", "openai/gpt-6.1-sol"
			action := &config.Actions[0]
			action.ID, action.Worker, action.Model = control.actionID, "gpt-6.1-sol", "gpt-6.1-sol"
			action.ControlID, action.Control = control.controlID, publicControl(t, control.controlID)
			_, server := executorWithHTTPDecisions(t, config, &decisionCapture{})
			status, answer := executeHTTP(t, server.URL, "/v1/messages", `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"run scheduler"}]}`, "sol-"+control.name)
			if status != 200 || !bytes.Contains(answer, []byte("ready")) || calls != 1 {
				t.Fatalf("Sol action failed: %d %s calls=%d", status, answer, calls)
			}
		})
	}
}

// A developer row must be refused on the real dispatch path, where the decoder
// has already canonicalized it to system. A body-level check cannot see it.
func TestModelPinMessagesRefusesDeveloperRowThroughDispatch(t *testing.T) {
	var reached int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached++
		modelPinHeaders(w.Header())
		w.Header().Set("X-Rayline-Selected-Model", "anthropic/claude-haiku-5.5")
		_, _ = io.WriteString(w, `{"id":"msg_fixture","type":"message","role":"assistant","model":"anthropic/claude-haiku-5.5","content":[{"type":"text","text":"ready"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":1}}`)
	}))
	defer upstream.Close()
	config := modelPinConfig(t, upstream.URL)
	deployment := &config.Deployments[0]
	deployment.URL = upstream.URL + "/v1/messages"
	deployment.Protocol = llmprotocol.AnthropicMessagesV1
	deployment.WireModel, deployment.RegistryModel = "anthropic/claude-haiku-5.5", "anthropic/claude-haiku-5.5"
	config.Actions[0].Model = "claude-haiku-5.5"
	_, server := executorWithHTTPDecisions(t, config, &decisionCapture{})

	// Every spelling the codec canonicalizes to developer must be refused:
	// canonicalRole lowercases and trims, so an exact-match check is bypassable.
	for index, role := range []string{"developer", "Developer", "DEVELOPER", " developer "} {
		reached = 0
		developer := `{"model":"arc-1.0","max_tokens":32,"messages":[{"role":"user","content":"hi"},{"role":"` + role + `","content":"elevated authority"}]}`
		status, result := executeHTTP(t, server.URL, "/v1/messages", developer, fmt.Sprintf("dev-row-%d", index))
		if status == 200 {
			t.Fatalf("role %q dispatched with changed authority: %s", role, result)
		}
		if !bytes.Contains(result, []byte("developer")) {
			t.Fatalf("role %q: refusal did not name the developer role: %s", role, result)
		}
		if reached != 0 {
			t.Fatalf("role %q: upstream was reached %d times despite the refusal", role, reached)
		}
	}
	reached = 0

	// The same shape with a system row the client actually sent is dispatched.
	system := `{"model":"arc-1.0","max_tokens":32,"messages":[{"role":"user","content":"hi"},{"role":"system","content":"environment"}]}`
	status, result := executeHTTP(t, server.URL, "/v1/messages", system, "sys-row")
	if status != 200 {
		t.Fatalf("in-band system row was refused on the dispatch path: %d %s", status, result)
	}
	if reached != 1 {
		t.Fatalf("expected one upstream call, got %d", reached)
	}
}

// A Messages client's anthropic-beta reaches a cloud Messages deployment: the
// body is carried member for member, and Anthropic refuses a beta-introduced
// member (context_management) that arrives without its beta. The client may
// send one beta per header line; every line is carried. The server-side
// fallback betas are not: the decision names the model that serves. A client
// on another wire, a cloud deployment on another wire, and a local deployment
// get no beta.
func TestCloudMessagesDispatchForwardsTheClientBeta(t *testing.T) {
	const (
		messagesBody = `{"model":"arc-1.0","max_tokens":64,"context_management":{"edits":[]},"messages":[{"role":"user","content":"hello"}]}`
		messagesOut  = `{"id":"msg_fixture","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[{"type":"text","text":"ready"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":30,"output_tokens":1}}`
		cm           = "context-management-2025-06-27"
		thinking     = "interleaved-thinking-2025-05-14"
	)
	cloudMessages := func(t *testing.T, url string) Config {
		config := modelPinConfig(t, url)
		deployment := &config.Deployments[0]
		deployment.URL = url + "/v1/messages"
		deployment.Protocol = llmprotocol.AnthropicMessagesV1
		deployment.WireModel, deployment.RegistryModel = "claude-sonnet-5-5", "anthropic/claude-sonnet-5.5"
		config.Actions[0].Model = "claude-sonnet-5.5"
		return config
	}
	for _, c := range []struct {
		name, path, body, answer, selected, want string
		betas                                    []string
		config                                   func(*testing.T, string) Config
	}{
		{"messages to cloud messages", "/v1/messages", messagesBody, messagesOut, "claude-sonnet-5-5", cm + "," + thinking, []string{cm + "," + thinking}, cloudMessages},
		{"one beta per line", "/v1/messages", messagesBody, messagesOut, "claude-sonnet-5-5", cm + "," + thinking, []string{cm, thinking}, cloudMessages},
		{"fallback beta removed", "/v1/messages", messagesBody, messagesOut, "claude-sonnet-5-5", cm, []string{cm + ", Server-Side-Fallback-2026-06-01"}, cloudMessages},
		{"only a fallback beta", "/v1/messages", messagesBody, messagesOut, "claude-sonnet-5-5", "", []string{"server-side-fallback-2026-06-01"}, cloudMessages},
		{"messages to local messages", "/v1/messages", messagesBody, messagesOut, "", "", []string{cm}, func(t *testing.T, url string) Config {
			return testConfig(t, url+"/v1/messages", true)
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, calls := "unset", 0
			gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				got = strings.Join(r.Header.Values("Anthropic-Beta"), "|")
				if c.selected != "" {
					modelPinHeaders(w.Header())
					w.Header().Set("X-Rayline-Selected-Model", c.selected)
				}
				_, _ = io.WriteString(w, c.answer)
			}))
			defer gateway.Close()
			_, server := executorWithHTTPDecisions(t, c.config(t, gateway.URL), &decisionCapture{})
			request, err := http.NewRequest(http.MethodPost, server.URL+c.path, strings.NewReader(c.body))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Set("Authorization", "Bearer "+testCoreToken)
			request.Header.Set("X-Rayline-Session-Id", "beta")
			for _, beta := range c.betas {
				request.Header.Add("Anthropic-Beta", beta)
			}
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			answer, _ := io.ReadAll(response.Body)
			response.Body.Close()
			if response.StatusCode != 200 || calls != 1 {
				t.Fatalf("%d after %d upstream calls: %s", response.StatusCode, calls, answer)
			}
			if got != c.want {
				t.Fatalf("upstream anthropic-beta = %q, want %q", got, c.want)
			}
		})
	}
}
