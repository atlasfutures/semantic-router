package extproc

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	maskedTestManifest = "../selection/raylinearc/testdata/policy_service/package_manifest.v5.json"
	maskedTestOpus     = "864cc7a8eab47911b61a2b3b5d79dcb8ed81c8ae9f8a90e941aea5bf38fc21d8"
	maskedTestGLMNone  = "20b6cddd7ae7cedc3ba8321cac7d962dca24b3c98dcf0408076adf40549e4a80"
	maskedTestGLMUp    = "cf5bc08ecfab91cdc1e26c1c2b98e7256e4a9e64183210ceb619e9c4745ff055"
)

// maskedTestDecision serves the shared v5 fixture with its GLM "up" control
// as a second Opus action, and Opus on the given endpoint: Opus has the
// plain native-only action and a steered one.
func maskedTestDecision(t *testing.T, vertex bool) (*config.RouterConfig, *config.Decision) {
	t.Helper()
	raw, err := os.ReadFile(maskedTestManifest)
	if err != nil {
		t.Fatal(err)
	}
	manifest := bytes.Replace(raw, []byte(`"model": "glm-5.3-flash",
      "control": {
        "base": {
          "native": "default"
        },
        "budget_tokens": null,
        "instruction": {
          "level": "up"`), []byte(`"model": "claude-opus-5",
      "control": {
        "base": {
          "native": "default"
        },
        "budget_tokens": null,
        "instruction": {
          "level": "up"`), 1)
	if bytes.Equal(manifest, raw) {
		t.Fatal("the fixture no longer holds the GLM up action")
	}
	path := filepath.Join(t.TempDir(), "package.json")
	if err := os.WriteFile(path, manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(manifest)
	opus := config.ModelParams{
		PreferredEndpoints: []string{"openrouter"}, APIFormat: config.APIFormatOpenAI,
		ExternalModelIDs: map[string]string{"vllm": "anthropic/claude-opus-5"},
	}
	if vertex {
		opus = config.ModelParams{
			PreferredEndpoints: []string{"vertex"}, APIFormat: config.APIFormatOpenAI,
			ExternalModelIDs: map[string]string{"vertex-ai": "publisher/claude-opus-5", "default": "anthropic/claude-opus-5"},
		}
	}
	cfg := &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"arm-opus": opus,
			"arm-glm": {
				PreferredEndpoints: []string{"openrouter"}, APIFormat: config.APIFormatOpenAI,
				ExternalModelIDs: map[string]string{"vllm": "z-ai/glm-5.3-flash"},
			},
		},
		VLLMEndpoints: []config.VLLMEndpoint{
			{Name: "openrouter", Address: "openrouter.ai", Port: 443, ProviderProfileName: "openrouter"},
			{Name: "vertex", Address: "aiplatform.googleapis.com", Port: 443, Type: "vertex-ai", ProviderProfileName: "vertex"},
		},
		ProviderProfiles: map[string]config.ProviderProfile{
			"openrouter": {Type: "openai", BaseURL: "https://openrouter.ai/api/v1"},
			"vertex":     {Type: "vertex-ai", BaseURL: "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi"},
		},
	}}
	on := true
	decision := &config.Decision{
		Name: "masked",
		ModelRefs: []config.ModelRef{
			{Model: "arm-opus", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &on}},
			{Model: "arm-glm", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &on}},
		},
		Algorithm: &config.AlgorithmConfig{RaylineARC: &config.RaylineARCAlgorithmConfig{
			PolicyService: &config.RaylineARCPolicyServiceConfig{
				BaseURL: "http://policy-service:8000", TotalTimeoutSeconds: 60,
				PackageSHA256: hex.EncodeToString(sum[:]), PackageManifest: path, AllowExperimentalControls: true,
				Bindings: []config.RaylineARCPolicyBinding{
					{ActionID: maskedTestOpus, Worker: "arm-opus"},
					{ActionID: maskedTestGLMNone, Worker: "arm-glm"},
					{ActionID: maskedTestGLMUp, Worker: "arm-opus"},
				},
			},
		}},
	}
	return cfg, decision
}

// The offer never holds a masked binding: on Vertex AI the steer Opus's chat
// cell refuses is left out and Opus keeps its plain action. Off Vertex AI
// nothing is masked here (such a config fails to load instead).
func TestPolicyScorerLeavesMaskedBindingsOut(t *testing.T) {
	cfg, decision := maskedTestDecision(t, true)
	scorer := newPolicyServiceScorer(cfg, decision)
	// Still the package's (a response scoring it is not unbound), never offered.
	if _, bound := scorer.bindings[maskedTestGLMUp]; !bound {
		t.Fatal("the masked steer left the catalog")
	}
	for _, actionID := range scorer.actionOrder {
		if actionID == maskedTestGLMUp {
			t.Fatal("the masked steer is in the offer")
		}
	}
	if len(scorer.actionOrder) != 2 || scorer.bindings[maskedTestOpus].model != "claude-opus-5" {
		t.Fatalf("offer = %v", scorer.actionOrder)
	}

	cfg, decision = maskedTestDecision(t, false)
	if masked := config.RaylineARCMaskedBindings(cfg, decision); len(masked) != 0 {
		t.Fatalf("an OpenRouter worker's binding was masked: %+v", masked)
	}
	if scorer := newPolicyServiceScorer(cfg, decision); len(scorer.actionOrder) != 3 {
		t.Fatalf("offer = %v", scorer.actionOrder)
	}
}
