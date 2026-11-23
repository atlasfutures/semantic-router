package config

import (
	"strings"
	"testing"
)

// onBedrockChat serves a worker on Amazon Bedrock's OpenAI-compatible
// endpoint: the registry's openai chat cell, which admits no steering
// instruction.
func onBedrockChat(cfg *RouterConfig, worker, registryModel string) {
	cfg.VLLMEndpoints = append(cfg.VLLMEndpoints, VLLMEndpoint{
		Name: "bedrock-" + worker, Address: "bedrock-runtime.us-east-1.amazonaws.com", Port: 443,
		Type: "bedrock", ProviderProfileName: "bedrock",
	})
	cfg.ProviderProfiles["bedrock"] = ProviderProfile{
		Type: "bedrock", BaseURL: "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
	}
	params := cfg.ModelConfig[worker]
	params.PreferredEndpoints = []string{"bedrock-" + worker}
	params.APIFormat = APIFormatOpenAI
	params.ExternalModelIDs = map[string]string{"bedrock": "us.publisher." + worker, "default": registryModel}
	cfg.ModelConfig[worker] = params
}

// On a Bedrock worker a steered control its registry cell refuses is masked,
// as on Vertex AI: the config loads, and the binding is listed so the offer
// leaves it out. The model is still served through its plain action.
func TestRaylineARCPolicyPackageV5MasksARefusedSteerOnBedrock(t *testing.T) {
	cfg, decision := policyV5VertexOpusSteer(t)
	onBedrockChat(cfg, "arm-opus", "anthropic/claude-opus-5")
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("a Bedrock worker with a refused steer and an admitted plain action refused: %v", err)
	}
	masked := RaylineARCMaskedBindings(cfg, &decision)
	if len(masked) != 1 || masked[0].ActionID != policyV5GLMUp || masked[0].Model != "claude-opus-5" ||
		masked[0].Worker != "arm-opus" || masked[0].Cell != "anthropic/claude-opus-5 on openaixchat" ||
		!strings.Contains(masked[0].Reason, "is not admitted") {
		t.Fatalf("masked = %+v", masked)
	}
}

// A model left with no admitted action, and a plain control the cell
// refuses, still fail the config on Bedrock.
func TestRaylineARCPolicyPackageV5BedrockMaskRefusals(t *testing.T) {
	cfg, decision := policyV5Decision(t, readPolicyV5Fixture(t))
	onBedrockChat(cfg, "arm-glm", "z-ai/glm-5.3-flash")
	err := validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), `every action of model "glm-5.3-flash"`) {
		t.Fatalf("a Bedrock model with every action masked loaded: %v", err)
	}

	cfg, decision = policyV5VertexOpusSteer(t)
	onBedrockChat(cfg, "arm-opus", "anthropic/no-such-model")
	err = validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "policy_service binding for action "+policyV5OpusAction) {
		t.Fatalf("a refused plain action on Bedrock loaded: %v", err)
	}
}

// A worker with one endpoint on Bedrock and one on OpenRouter is not served
// only by a managed cloud: a refused steer fails the config.
func TestRaylineARCPolicyPackageV5MixedBedrockWorkerStillFails(t *testing.T) {
	cfg, decision := policyV5VertexOpusSteer(t)
	params := cfg.ModelConfig["arm-opus"]
	openRouter := append([]string(nil), params.PreferredEndpoints...)
	onBedrockChat(cfg, "arm-opus", "anthropic/claude-opus-5")
	params = cfg.ModelConfig["arm-opus"]
	params.PreferredEndpoints = append(params.PreferredEndpoints, openRouter...)
	cfg.ModelConfig["arm-opus"] = params
	if err := validatePolicyDispatch(cfg, decision); err == nil {
		t.Fatal("a refused steer on a Bedrock-and-OpenRouter worker loaded")
	}
	if masked := RaylineARCMaskedBindings(cfg, &decision); len(masked) != 0 {
		t.Fatalf("a mixed worker's binding was masked: %+v", masked)
	}
}
