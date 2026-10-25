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
