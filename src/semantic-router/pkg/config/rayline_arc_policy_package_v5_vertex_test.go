package config

import (
	"bytes"
	"strings"
	"testing"
)

// policyV5VertexOpusSteer is the v5 fixture with the GLM "up" control as a
// second Opus action (action ids are opaque to VSR): Opus has the plain
// native-only action and a steered one.
func policyV5VertexOpusSteer(t *testing.T) (*RouterConfig, Decision) {
	t.Helper()
	manifest := bytes.Replace(readPolicyV5Fixture(t),
		[]byte(`"model": "glm-5.3-flash",
      "control": {
        "base": {
          "native": "default"
        },
        "budget_tokens": null,
        "instruction": {
          "level": "up"`),
		[]byte(`"model": "claude-opus-5",
      "control": {
        "base": {
          "native": "default"
        },
        "budget_tokens": null,
        "instruction": {
          "level": "up"`), 1)
	if bytes.Equal(manifest, readPolicyV5Fixture(t)) {
		t.Fatal("the fixture no longer holds the GLM up action")
	}
	cfg, decision := policyV5Decision(t, manifest)
	decision.Algorithm.RaylineARC.PolicyService.Bindings[2].Worker = "arm-opus"
	return cfg, decision
}

// onVertexChat serves a worker on Vertex AI's OpenAI-compatible endpoint:
// the registry's openai chat cell, which admits no steering instruction.
func onVertexChat(cfg *RouterConfig, worker, registryModel string) {
	cfg.VLLMEndpoints = append(cfg.VLLMEndpoints, VLLMEndpoint{
		Name: "vertex-" + worker, Address: "aiplatform.googleapis.com", Port: 443,
		Type: "vertex-ai", ProviderProfileName: "vertex",
	})
	cfg.ProviderProfiles["vertex"] = ProviderProfile{
		Type: "vertex-ai", BaseURL: "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi",
	}
	params := cfg.ModelConfig[worker]
	params.PreferredEndpoints = []string{"vertex-" + worker}
	params.APIFormat = APIFormatOpenAI
	params.ExternalModelIDs = map[string]string{"vertex-ai": "publisher/" + worker, "default": registryModel}
	cfg.ModelConfig[worker] = params
}

// On a Vertex AI worker a steered control its registry cell refuses is
// masked: the config loads, and the binding is listed so the offer leaves it
// out. The model is still served through its plain action.
func TestRaylineARCPolicyPackageV5MasksARefusedSteerOnVertex(t *testing.T) {
	cfg, decision := policyV5VertexOpusSteer(t)
	onVertexChat(cfg, "arm-opus", "anthropic/claude-opus-5")
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("a Vertex AI worker with a refused steer and an admitted plain action refused: %v", err)
	}
	masked := RaylineARCMaskedBindings(cfg, &decision)
	if len(masked) != 1 || masked[0].ActionID != policyV5GLMUp || masked[0].Model != "claude-opus-5" ||
		masked[0].Worker != "arm-opus" || masked[0].Cell != "anthropic/claude-opus-5 on openaixchat" ||
		!strings.Contains(masked[0].Reason, "is not admitted") {
		t.Fatalf("masked = %+v", masked)
	}
}

// The relaxation is Vertex AI's alone: the same refused steer on a worker
// that is not on Vertex AI fails the config, as it always has, and lists
// nothing.
func TestRaylineARCPolicyPackageV5RefusedSteerOffVertexStillFails(t *testing.T) {
	cfg, decision := policyV5VertexOpusSteer(t)
	params := cfg.ModelConfig["arm-opus"]
	params.APIFormat = APIFormatOpenAI // OpenRouter's chat cell, which refuses the steer
	cfg.ModelConfig["arm-opus"] = params
	err := validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "is not admitted for anthropic/claude-opus-5 on openrouterxchat") {
		t.Fatalf("a refused steer on an OpenRouter worker loaded: %v", err)
	}
	if masked := RaylineARCMaskedBindings(cfg, &decision); len(masked) != 0 {
		t.Fatalf("an OpenRouter worker's binding was masked: %+v", masked)
	}
}

// A model left with no admitted action, and a plain control the cell
// refuses, still fail the config on Vertex AI.
func TestRaylineARCPolicyPackageV5VertexMaskRefusals(t *testing.T) {
	// GLM's two actions are both steered: on Vertex AI's chat cell neither
	// is admitted, so GLM would never be offered.
	cfg, decision := policyV5Decision(t, readPolicyV5Fixture(t))
	onVertexChat(cfg, "arm-glm", "z-ai/glm-5.3-flash")
	err := validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), `every action of model "glm-5.3-flash"`) {
		t.Fatalf("a Vertex AI model with every action masked loaded: %v", err)
	}

	// Opus's plain action on a cell the registry does not have.
	cfg, decision = policyV5VertexOpusSteer(t)
	onVertexChat(cfg, "arm-opus", "anthropic/no-such-model")
	err = validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "policy_service binding for action "+policyV5OpusAction) {
		t.Fatalf("a refused plain action on Vertex AI loaded: %v", err)
	}
}
