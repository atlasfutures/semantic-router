package execution

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

func writeDerivedConfig(t *testing.T, mutate func(map[string]any)) string {
	t.Helper()
	config := map[string]any{
		"contract_version":            1,
		"allow_experimental_controls": true,
		"decisions": map[string]any{"api_mode": "execution_context_v2", "url": "https://router.example/v1/routes", "auth_env": "TEST_DECISIONS_KEY",
			"model": "rayline-router", "result": "top", "routing": map[string]any{"policy": map[string]any{"preset": "arc-policy.hybrid-default.v1"}}},
		"deployments": []any{
			map[string]any{"id": "local", "kind": "local", "url": "http://127.0.0.1:8080/v1/chat/completions", "protocol": "openai.chat.v1", "wire_model": "qwen3.8:27b",
				"registry_model": "qwen/qwen3.8-27b", "registry_provider": "openai", "runtime_identity": "local_model:qwen3.8:27b", "max_output_tokens": 4096,
				"context_window": 16384, "input_modalities": []string{"text"}, "token_units": ""},
			map[string]any{"id": "cloud-sonnet", "kind": "cloud", "cloud_transport": "model_pin", "url": "https://router.example/v1/messages", "protocol": "anthropic.messages.v1",
				"wire_model": "anthropic/claude-sonnet-5.5", "registry_model": "anthropic/claude-sonnet-5.5", "registry_provider": "openrouter",
				"runtime_identity": "rayline-catalog-model-pin-v1", "max_output_tokens": 4096, "context_window": 200000, "input_modalities": []string{"text"},
				"token_units": "", "auth_env": "TEST_DECISIONS_KEY"},
		},
		"derived_actions": []any{
			map[string]any{"model": "qwen3.8-27b", "worker": "qwen3.8-27b", "deployment_id": "local"},
			map[string]any{"model": "claude-sonnet-5.5", "worker": "claude-sonnet-5.5", "deployment_id": "cloud-sonnet", "bases": []string{"default"}},
		},
	}
	if mutate != nil {
		mutate(config)
	}
	raw, _ := json.Marshal(config)
	path := filepath.Join(t.TempDir(), "hybrid.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// Derived action identities equal ARC's v5 ids (sha256 of JCS
// {"control_id","model"}): these are the ids package 5505cbf8 binds for
// local Qwen and Sonnet.
func TestDerivedActionsUseARCActionIdentity(t *testing.T) {
	config, registry, err := LoadConfig(writeDerivedConfig(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	if config.RegistrySHA256 != registry.SHA256 || config.PackageSHA256 != "" {
		t.Fatalf("pins %q %q", config.RegistrySHA256, config.PackageSHA256)
	}
	ids := map[string][]string{}
	for _, action := range config.Actions {
		ids[action.Model] = append(ids[action.Model], action.ID)
	}
	for model := range ids {
		sort.Strings(ids[model])
	}
	if strings.Join(ids["qwen3.8-27b"], ",") != "d6936db7073d3f30c663d504b7acbb124e8f752a9a766cfb5e3129050e26e0a6" {
		t.Fatalf("qwen actions %v", ids["qwen3.8-27b"])
	}
	want := "231d62afdd0cc65c726614798985f31baa45536533fdbee6cec3abc511c4443b,78c901608efbed63739a5e4e44380b4b9a98326b39ac6ed8f3be7db574b38cfe,8a3787b121f03eb61c77381ad0a024f5bfa03db323e32076c9906840984bc0b4,b399be26ceb6193121d05eeda1119d2bea8f224874d019bdc0e31baa99432b75"
	if strings.Join(ids["claude-sonnet-5.5"], ",") != want {
		t.Fatalf("sonnet actions %v", ids["claude-sonnet-5.5"])
	}
	t.Setenv("TEST_DECISIONS_KEY", "fixture")
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewServer(config, registry, provider, testCoreToken); err != nil {
		t.Fatal(err)
	}
	// The embedded registry may be newer than the serving one: no pin.
	if expect := provider.projectV2(DecisionInput{State: ExecutionContext{ContextEpoch: "e:0"}}).Expect; expect != nil {
		t.Fatalf("a package-less profile pinned %+v", expect)
	}
}

// An explicit registry_path executes that registry with no serving-side pin.
func TestDerivedProfileTakesAnExplicitRegistryPath(t *testing.T) {
	wd, _ := os.Getwd()
	config, registry, err := LoadConfig(writeDerivedConfig(t, func(config map[string]any) {
		config["registry_path"] = filepath.Join(wd, "..", "thinkingcontrol", "thinking_controls.compiled.json")
	}))
	if err != nil {
		t.Fatal(err)
	}
	if config.RegistrySHA256 != registry.SHA256 || len(config.Actions) == 0 {
		t.Fatalf("registry %q actions %d", config.RegistrySHA256, len(config.Actions))
	}
	t.Setenv("TEST_DECISIONS_KEY", "fixture")
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	if expect := provider.projectV2(DecisionInput{State: ExecutionContext{ContextEpoch: "e:0"}}).Expect; expect != nil {
		t.Fatalf("a package-less profile pinned %+v", expect)
	}
}

func TestDerivedActionsAreV2OnlyAndExclusiveOfAPackage(t *testing.T) {
	for name, mutate := range map[string]func(map[string]any){
		"current mode":      func(config map[string]any) { config["decisions"].(map[string]any)["api_mode"] = "current" },
		"with package":      func(config map[string]any) { config["package_sha256"] = strings.Repeat("a", 64) },
		"with registry pin": func(config map[string]any) { config["registry_sha256"] = strings.Repeat("3", 64) },
		"unknown deployment": func(config map[string]any) {
			config["derived_actions"] = []any{map[string]any{"model": "qwen3.8-27b", "worker": "q", "deployment_id": "nowhere"}}
		},
		"no admitted control": func(config map[string]any) {
			config["derived_actions"] = []any{map[string]any{"model": "qwen3.8-27b", "worker": "q", "deployment_id": "local", "bases": []string{"xhigh"}}}
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, err := LoadConfig(writeDerivedConfig(t, mutate)); err == nil {
				t.Fatal("accepted")
			}
		})
	}
}

// A package-less profile executes end to end: the service names its serving
// package, which the core records without a local pin.
func TestDerivedProfileExecutesARequest(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	localURL := chatUpstream(t, local, false).URL + "/v1/chat/completions"
	cloudURL := chatUpstream(t, cloud, true).URL + "/v1/chat/completions"
	path := writeDerivedConfig(t, func(config map[string]any) {
		deployments := config["deployments"].([]any)
		deployments[0].(map[string]any)["url"] = localURL
		sonnet := deployments[1].(map[string]any)
		sonnet["url"], sonnet["protocol"] = cloudURL, "openai.chat.v1"
		sonnet["wire_model"], sonnet["registry_model"] = "moonshotai/kimi-k3", "moonshotai/kimi-k3"
		config["derived_actions"] = []any{
			map[string]any{"model": "qwen3.8-27b", "worker": "qwen3.8-27b", "deployment_id": "local"},
			map[string]any{"model": "kimi-k3", "worker": "kimi-k3", "deployment_id": "cloud-sonnet", "bases": []string{"default"}},
		}
	})
	config, _, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	qwen := "d6936db7073d3f30c663d504b7acbb124e8f752a9a766cfb5e3129050e26e0a6"
	// The service serves another package and a different (older) registry.
	fixture := &v2Decisions{choose: func(contextV2) string { return qwen }, mutate: func(response map[string]any) {
		response["identity"].(map[string]any)["package_sha256"] = strings.Repeat("5", 64)
		response["identity"].(map[string]any)["thinking_controls_sha256"] = strings.Repeat("3", 64)
	}}
	_, httpServer := v2Server(t, config, fixture)
	if status, response := executeHTTP(t, httpServer.URL, "/v1/chat/completions", `{"model":"arc-1.0","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`, "derived"); status != 200 {
		t.Fatalf("status %d: %s", status, response)
	}
	if local.count() != 1 || fixture.snapshot()[0].Execution.Expect != nil {
		t.Fatal("derived profile did not execute locally without a package pin")
	}
}

func TestV2RefusalNamesTheContractCode(t *testing.T) {
	local, cloud := &upstreamFixture{}, &upstreamFixture{}
	config := hybridV2Config(t, chatUpstream(t, local, false).URL, chatUpstream(t, cloud, true).URL+"/v1/chat/completions")
	refusing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusConflict)
		_, _ = io.WriteString(w, `{"error":{"code":"execution_identity_mismatch","message":"secret detail","type":"invalid_request_error"},"type":"error"}`)
	}))
	defer refusing.Close()
	t.Setenv("TEST_DECISIONS_KEY", "fixture")
	config.Decisions.URL = refusing.URL
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	_, err = provider.Decide(t.Context(), DecisionInput{Body: []byte(`{"model":"arc-1.0","messages":[]}`), Format: "openai.chat.v1", SessionID: "s", BranchID: "main", RequestID: "r"})
	if err == nil || !strings.Contains(err.Error(), "HTTP 409, execution_identity_mismatch") || strings.Contains(err.Error(), "secret") {
		t.Fatalf("error %v", err)
	}
}

// glmProtocolsConfig serves z-ai/glm-5.3 through model_pin on Messages and on
// Chat, beside local Qwen on Chat. OpenRouter admits more glm controls on
// Messages than on Chat, so the two protocols offer different glm actions.
func glmProtocolsConfig(t *testing.T, mutate func(map[string]any)) string {
	t.Helper()
	return writeDerivedConfig(t, func(config map[string]any) {
		deployments := config["deployments"].([]any)
		glm := func(id, protocol, path string) map[string]any {
			deployment := map[string]any{}
			for key, value := range deployments[1].(map[string]any) {
				deployment[key] = value
			}
			deployment["id"], deployment["protocol"], deployment["url"] = id, protocol, "https://router.example"+path
			deployment["wire_model"], deployment["registry_model"] = "z-ai/glm-5.3", "z-ai/glm-5.3"
			return deployment
		}
		config["deployments"] = []any{deployments[0], glm("glm-messages", "anthropic.messages.v1", "/v1/messages"), glm("glm-chat", "openai.chat.v1", "/v1/chat/completions")}
		config["derived_actions"] = []any{
			map[string]any{"model": "qwen3.8-27b", "worker": "qwen3.8-27b", "deployment_id": "local"},
			map[string]any{"model": "glm-5.3", "worker": "glm-5.3", "deployment_id": "glm-messages"},
			map[string]any{"model": "glm-5.3", "worker": "glm-5.3", "deployment_id": "glm-chat"},
		}
		if mutate != nil {
			mutate(config)
		}
	})
}

func loadServer(t *testing.T, path string) (Config, *HTTPDecisions, *Server, error) {
	t.Helper()
	config, registry, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("TEST_DECISIONS_KEY", "fixture")
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		return config, nil, nil, err
	}
	server, err := NewServer(config, registry, provider, testCoreToken)
	return config, provider, server, err
}

// Each protocol is offered only the actions bound to one of its deployments;
// a control both glm deployments admit is one action in both views.
func TestEachProtocolIsOfferedOnlyItsBoundActions(t *testing.T) {
	config, provider, server, err := loadServer(t, glmProtocolsConfig(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	deploymentOf := map[string][]string{}
	for _, action := range config.Actions {
		deploymentOf[action.ID] = append(deploymentOf[action.ID], action.DeploymentID)
	}
	offered := func(format llmprotocol.WireFormat) map[string]bool {
		ids := map[string]bool{}
		for _, target := range provider.projectV2(DecisionInput{Format: format, State: ExecutionContext{ContextEpoch: "e:0"}}).Targets {
			ids[target.ActionID] = true
		}
		return ids
	}
	chat, messages := offered(llmprotocol.OpenAIChatV1), offered(llmprotocol.AnthropicMessagesV1)
	shared := 0
	for id, deployments := range deploymentOf {
		bound := map[string]bool{}
		for _, deployment := range deployments {
			bound[deployment] = true
		}
		if chat[id] != (bound["local"] || bound["glm-chat"]) || messages[id] != bound["glm-messages"] {
			t.Fatalf("action %s on %v: chat=%v messages=%v", id, deployments, chat[id], messages[id])
		}
		if bound["glm-chat"] && bound["glm-messages"] {
			shared++
		}
	}
	// The default control is admitted on both cells; the Messages cell admits
	// more, which Chat is never offered.
	if shared == 0 || len(messages) <= shared {
		t.Fatalf("shared=%d messages=%d chat=%d", shared, len(messages), len(chat))
	}
	if message := server.unboundProtocol(llmprotocol.OpenAIResponsesV1); message != "no deployment serves openai.responses.v1 requests for models glm-5.3, qwen3.8-27b; the core does not translate between protocols" {
		t.Fatalf("unbound protocol message %q", message)
	}
	if message := server.unboundProtocol(llmprotocol.AnthropicMessagesV1); message != "" {
		t.Fatalf("bound protocol refused: %q", message)
	}
}

func TestProtocolBindingErrorsFailAtLoad(t *testing.T) {
	for name, tc := range map[string]struct {
		mutate func(map[string]any)
		want   string
	}{
		"two deployments of one protocol": {func(config map[string]any) {
			deployments := config["deployments"].([]any)
			second := map[string]any{}
			for key, value := range deployments[2].(map[string]any) {
				second[key] = value
			}
			second["id"] = "glm-chat-2"
			config["deployments"] = append(deployments, second)
			config["derived_actions"] = append(config["derived_actions"].([]any), map[string]any{"model": "glm-5.3", "worker": "glm-5.3", "deployment_id": "glm-chat-2"})
		}, "binds more than one openai.chat.v1 deployment"},
		"another worker on another protocol": {func(config map[string]any) {
			config["derived_actions"].([]any)[2].(map[string]any)["worker"] = "glm-5.3-chat"
		}, "names a different worker or reported_model on another deployment"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, _, err := loadServer(t, glmProtocolsConfig(t, tc.mutate)); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %v, want %q", err, tc.want)
			}
		})
	}
}

// Only execution_context_v2 offers ARC the request's own targets, so the
// other modes refuse at load an action some served protocol cannot execute.
func TestOfferWideModesRefuseProtocolSpecificActions(t *testing.T) {
	t.Setenv("TEST_DECISIONS_KEY", "fixture")
	config, _, err := LoadConfig(writeDerivedConfig(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	config.Decisions.APIMode, config.Decisions.Routing = DecisionsCurrent, json.RawMessage(`{}`)
	if _, err := NewHTTPDecisions(config); err == nil || !strings.Contains(err.Error(), "current Decisions offers every action to every protocol: action ") {
		t.Fatalf("protocol-specific actions under current: %v", err)
	}
}
