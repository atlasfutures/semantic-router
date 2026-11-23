package config

import (
	"bytes"
	"strings"
	"testing"
)

// policyV5OpusPlainHigh is the v5 fixture with the GLM "up" action replaced
// by a second, plain Opus action at native effort "high" (action ids are
// opaque to VSR). Opus then has the native-default action (the package's
// fallback) and a named-effort one; the registry admits "high" for Opus on
// OpenRouter's Messages cell only.
func policyV5OpusPlainHigh(t *testing.T, edit func([]byte) []byte) (*RouterConfig, Decision) {
	t.Helper()
	fixture := readPolicyV5Fixture(t)
	start := bytes.Index(fixture, []byte(`"action_id": "`+policyV5GLMUp+`"`))
	if start < 0 {
		t.Fatal("the fixture no longer holds the GLM up action")
	}
	end := bytes.Index(fixture[start:], []byte(`"trained_arm_ids"`))
	if end < 0 {
		t.Fatal("the GLM up action has no trained_arm_ids")
	}
	action := []byte(`"action_id": "` + policyV5GLMUp + `",
      "model": "claude-opus-5",
      "control": {
        "base": {
          "native": "high"
        },
        "budget_tokens": null,
        "instruction": null
      },
      "control_id": "8c4faa90467d7151b03810424e673723a62bdface213d36659862037aaf4dc2d",
      "level": null,
      `)
	manifest := append(append(append([]byte{}, fixture[:start]...), action...), fixture[start+end:]...)
	if edit != nil {
		manifest = edit(manifest)
	}
	cfg, decision := policyV5Decision(t, manifest)
	decision.Algorithm.RaylineARC.PolicyService.Bindings[2].Worker = "arm-opus"
	return cfg, decision
}

// On a Bedrock-only worker, a refused plain control at a named native effort
// is masked when the model keeps an admitted action and the action is not the
// fallback: the config loads and the binding is listed with its reason.
func TestRaylineARCPolicyPackageV5MasksARefusedNativeEffortOnBedrock(t *testing.T) {
	cfg, decision := policyV5OpusPlainHigh(t, nil)
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("the plain-high fixture refused on OpenRouter: %v", err)
	}
	onBedrockChat(cfg, "arm-opus", "anthropic/claude-opus-5")
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("a Bedrock worker with a refused native effort and an admitted default refused: %v", err)
	}
	masked := RaylineARCMaskedBindings(cfg, &decision)
	if len(masked) != 1 || masked[0].ActionID != policyV5GLMUp || masked[0].Worker != "arm-opus" ||
		masked[0].Reason != "native effort base not admitted on anthropic/claude-opus-5 on openaixchat" {
		t.Fatalf("masked = %+v", masked)
	}
}

// A refused native-default control, a refused named effort that is the
// fallback or a reference action, and the same refusal on an OpenRouter
// worker all still fail the config.
func TestRaylineARCPolicyPackageV5PlainMaskRefusals(t *testing.T) {
	cfg, decision := policyV5OpusPlainHigh(t, nil)
	onBedrockChat(cfg, "arm-opus", "anthropic/no-such-model")
	err := validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "policy_service binding for action "+policyV5OpusAction) {
		t.Fatalf("a refused native-default control on Bedrock loaded: %v", err)
	}

	asFallback := func(m []byte) []byte {
		return bytes.Replace(m, []byte(`"fallback_action_id": "`+policyV5OpusAction+`"`),
			[]byte(`"fallback_action_id": "`+policyV5GLMUp+`"`), 1)
	}
	asReference := func(m []byte) []byte {
		return bytes.Replace(m, []byte(`"opus": "`+policyV5OpusAction+`"`),
			[]byte(`"opus": "`+policyV5GLMUp+`"`), 1)
	}
	for name, edit := range map[string]func([]byte) []byte{"fallback": asFallback, "reference": asReference} {
		cfg, decision := policyV5OpusPlainHigh(t, edit)
		if err := validatePolicyDispatch(cfg, decision); err != nil {
			t.Fatalf("%s: the edited fixture refused on OpenRouter: %v", name, err)
		}
		onBedrockChat(cfg, "arm-opus", "anthropic/claude-opus-5")
		err := validatePolicyDispatch(cfg, decision)
		if err == nil || !strings.Contains(err.Error(), "policy_service binding for action "+policyV5GLMUp) {
			t.Fatalf("%s: a refused named effort on the %s action loaded: %v", name, name, err)
		}
	}

	// OpenRouter's Chat cell refuses "high" for Opus too, and OpenRouter is
	// not a managed cloud: no masking.
	cfg, decision = policyV5OpusPlainHigh(t, nil)
	params := cfg.ModelConfig["arm-opus"]
	params.APIFormat = APIFormatOpenAI
	cfg.ModelConfig["arm-opus"] = params
	err = validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "policy_service binding for action "+policyV5GLMUp) {
		t.Fatalf("a refused named effort on an OpenRouter worker loaded: %v", err)
	}
	if masked := RaylineARCMaskedBindings(cfg, &decision); len(masked) != 0 {
		t.Fatalf("an OpenRouter binding was masked: %+v", masked)
	}
}
