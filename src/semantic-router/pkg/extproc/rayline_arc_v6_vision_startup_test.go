//go:build !windows && cgo

package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// v6CellConfig renders the v5 e2e cell's canonical YAML serving
// pathfinder's v6 fixture (same actions), with each modelCards entry's
// vision line as given ("" leaves the card unmarked), and returns its path.
func v6CellConfig(t *testing.T, glmVision, opusVision string) string {
	t.Helper()
	manifest, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v6.canonical_v2.json")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "package.json")
	if err := os.WriteFile(manifestPath, manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(manifest)
	rendered := strings.NewReplacer(
		"{{POLICY_URL}}", "http://127.0.0.1:1", "{{ALIAS}}", policyTestAlias, "{{PACKAGE}}", hex.EncodeToString(sum[:]),
		"{{MANIFEST}}", manifestPath, "{{GLM_FORMAT}}", "api_format: openai",
		"{{OPUS}}", v5OpusAction, "{{GLM_NONE}}", v5GLMNone, "{{GLM_UP}}", v5GLMUp,
	).Replace(v5ConfigTemplate)
	for card, vision := range map[string]string{"glm": glmVision, "opus": opusVision} {
		if vision == "" {
			continue
		}
		anchor := "    - name: " + card + "\n      modality: text\n"
		if !strings.Contains(rendered, anchor) {
			t.Fatalf("the cell config no longer carries %q", anchor)
		}
		rendered = strings.Replace(rendered, anchor, anchor+"      vision: "+vision+"\n", 1)
	}
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// Through the public load path -- config.Parse, which startup
// (parseRouterConfigFile) and every file reload (parseReloadConfig) call --
// a v6 cell whose canonical routing.modelCards leave an arm's vision unset is
// refused, naming the arm.
func TestV6CellWithAnUnmarkedCardIsRefusedAtLoad(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	for name, cards := range map[string][2]string{
		"both unmarked": {"", ""},
		"glm unmarked":  {"", "true"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := config.Parse(v6CellConfig(t, cards[0], cards[1]))
			if err == nil || !strings.Contains(err.Error(), "unmarked: glm") {
				t.Fatalf("err = %v, want glm named unmarked", err)
			}
			if cards[1] != "" && strings.Contains(err.Error(), "opus") {
				t.Fatalf("err names the marked opus: %v", err)
			}
		})
	}
}

// With every card marked, the canonical config loads, routing.modelCards'
// vision materializes on the model params (false as *false, not unset), and
// the router builds from it as at startup.
func TestV6CellWithEveryCardMarkedLoads(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	path := v6CellConfig(t, "false", "true")
	cfg, err := config.Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	glm, opus := cfg.ModelConfig["glm"].Vision, cfg.ModelConfig["opus"].Vision
	if glm == nil || *glm || opus == nil || !*opus {
		t.Fatalf("vision materialized as glm=%v opus=%v, want *false and *true", glm, opus)
	}
	if _, err := NewOpenAIRouter(path); err != nil {
		t.Fatalf("build router: %v", err)
	}
}
