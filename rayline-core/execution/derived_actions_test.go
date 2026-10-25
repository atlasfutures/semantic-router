package execution

import (
	"encoding/json"
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
	if expect := provider.projectV2(DecisionInput{State: ExecutionContext{ContextEpoch: "e:0"}}).Expect; expect != nil {
		t.Fatalf("a package-less profile pinned %+v", expect)
	}
}

func TestDerivedActionsAreV2OnlyAndExclusiveOfAPackage(t *testing.T) {
	for name, mutate := range map[string]func(map[string]any){
		"current mode": func(config map[string]any) { config["decisions"].(map[string]any)["api_mode"] = "current" },
		"with package": func(config map[string]any) { config["package_sha256"] = strings.Repeat("a", 64) },
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
