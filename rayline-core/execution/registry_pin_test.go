package execution

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A packaged profile whose package names another (serving) registry than the
// binary embeds: v2 without registry_path executes with the embedded registry
// and pins the serving one; an explicit registry_path or v1 stays exact.
func writePackagedProfile(t *testing.T, mode, registrySHA string, withPath bool) string {
	t.Helper()
	dir := t.TempDir()
	controlID := "628d285537ca7cdb52c4bd3433cea7a1ec36005a14dbc67ffa1c40487d7b8ad0"
	serving := strings.Repeat("3", 64)
	manifest := map[string]any{"schema_version": "rayline.arc-policy-package.v6", "package_id": strings.Repeat("9", 64), "thinking_controls_sha256": serving,
		"actions": []any{map[string]any{"action_id": "d6936db7073d3f30c663d504b7acbb124e8f752a9a766cfb5e3129050e26e0a6", "model": "qwen3.8-27b", "control_id": controlID, "control": json.RawMessage(publicControl(t, controlID))}}}
	raw, _ := json.Marshal(manifest)
	if err := os.WriteFile(filepath.Join(dir, "package.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	routing := map[string]any{}
	if mode == DecisionsExecutionContextV2 {
		routing["policy"] = map[string]any{"preset": "arc-policy.hybrid-default.v1"}
	}
	config := map[string]any{"contract_version": 1, "package_path": "package.json", "package_sha256": fingerprint(raw),
		"decisions": map[string]any{"api_mode": mode, "url": "https://router.example/v1/routes", "auth_env": "TEST_DECISIONS_KEY", "model": "rayline-router", "result": "top", "routing": routing, "offer_sha256": strings.Repeat("b", 64)},
		"deployments": []any{map[string]any{"id": "local", "kind": "local", "url": "http://127.0.0.1:8080/v1/chat/completions", "protocol": "openai.chat.v1", "wire_model": "q",
			"registry_model": "qwen/qwen3.8-27b", "registry_provider": "openai", "runtime_identity": "r", "max_output_tokens": 4096, "context_window": 16384, "input_modalities": []string{"text"}, "token_units": ""}},
		"bindings": []any{map[string]any{"action_id": "d6936db7073d3f30c663d504b7acbb124e8f752a9a766cfb5e3129050e26e0a6", "worker": "qwen3.8-27b", "deployment_id": "local"}}}
	if registrySHA != "" {
		config["registry_sha256"] = registrySHA
	}
	if withPath {
		wd, _ := os.Getwd()
		config["registry_path"] = filepath.Join(wd, "..", "thinkingcontrol", "thinking_controls.compiled.json")
	}
	raw, _ = json.Marshal(config)
	path := filepath.Join(dir, "hybrid.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestV2PackagedProfileNeedsNoRegistryPath(t *testing.T) {
	serving := strings.Repeat("3", 64)
	for name, test := range map[string]struct {
		mode     string
		registry string
		withPath bool
		ok       bool
	}{
		"v2 pins the serving registry":    {DecisionsExecutionContextV2, serving, false, true},
		"v2 takes the package registry":   {DecisionsExecutionContextV2, "", false, true},
		"explicit registry_path is exact": {DecisionsExecutionContextV2, serving, true, false},
		"v1 stays exact":                  {DecisionsExecutionContextV1, serving, false, false},
	} {
		t.Run(name, func(t *testing.T) {
			config, registry, err := LoadConfig(writePackagedProfile(t, test.mode, test.registry, test.withPath))
			if (err == nil) != test.ok {
				t.Fatalf("ok=%v err=%v", test.ok, err)
			}
			if err != nil {
				return
			}
			if config.RegistrySHA256 != serving || registry.SHA256 == serving {
				t.Fatalf("serving pin %s executor %s", config.RegistrySHA256, registry.SHA256)
			}
			t.Setenv("TEST_DECISIONS_KEY", "fixture")
			provider, err := NewHTTPDecisions(config)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := NewServer(config, registry, provider, testCoreToken); err != nil {
				t.Fatal(err)
			}
			if expect := provider.projectV2(DecisionInput{State: ExecutionContext{ContextEpoch: "e:0"}}).Expect; expect == nil || expect.ThinkingControlsSHA256 == nil || *expect.ThinkingControlsSHA256 != serving {
				t.Fatalf("expect %+v", expect)
			}
		})
	}
}
