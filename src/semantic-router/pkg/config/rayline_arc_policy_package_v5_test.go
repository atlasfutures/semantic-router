package config

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The shared v5 fixture: an Opus native-only action and two GLM actions (the
// neutral level and "up"), all at the default base, under registry
// 3083a4b6...cff2. Both models' OpenRouter cells admit their instructions
// experimentally.
const (
	policyV5Fixture      = "../selection/raylinearc/testdata/policy_service/package_manifest.v5.json"
	policyV5OpusAction   = "0527fe7631783f956fc850185d3130926ed252b9e49e005eba68a8f884152321"
	policyV5GLMNone      = "9d2b81063cac1733ef6859b8fc061ba5192366aa0966c7ca5e055110041aa332"
	policyV5GLMUp        = "b18e89bdb29cfb16b9cf71e46207ebde0ad463e5256f4b2c837280d84f13d7d7"
	policyV5OpenRouter   = "https://openrouter.ai/api/v1"
	policyV5AnthropicURL = "https://api.anthropic.com"
)

// policyV5Decision is a policy-service decision serving the v5 fixture: Opus
// on OpenRouter's Messages wire, GLM on OpenRouter's Chat wire.
func policyV5Decision(t *testing.T, manifest []byte) (*RouterConfig, Decision) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "package.json")
	if err := os.WriteFile(path, manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(manifest)
	on := true
	cfg := &RouterConfig{BackendModels: BackendModels{
		ModelConfig: map[string]ModelParams{
			"arm-opus": {PreferredEndpoints: []string{"openrouter"}, APIFormat: APIFormatAnthropic},
			"arm-glm":  {PreferredEndpoints: []string{"openrouter"}, APIFormat: APIFormatOpenAI},
		},
		VLLMEndpoints: []VLLMEndpoint{
			{Name: "openrouter", Address: "openrouter.ai", Port: 443, ProviderProfileName: "openrouter"},
			{Name: "anthropic", Address: "api.anthropic.com", Port: 443, ProviderProfileName: "anthropic"},
		},
		ProviderProfiles: map[string]ProviderProfile{
			"openrouter": {Type: "openai", BaseURL: policyV5OpenRouter},
			"anthropic":  {Type: "anthropic", BaseURL: policyV5AnthropicURL},
		},
	}}
	decision := validRaylineARCDecision()
	decision.ModelRefs = []ModelRef{
		{Model: "arm-opus", ModelReasoningControl: ModelReasoningControl{UseReasoning: &on}},
		{Model: "arm-glm", ModelReasoningControl: ModelReasoningControl{UseReasoning: &on}},
	}
	arc := decision.Algorithm.RaylineARC
	arc.ArtifactDir, arc.ArtifactRevision, arc.Encoder = "", "", RaylineARCEncoderConfig{}
	arc.PolicyService = &RaylineARCPolicyServiceConfig{
		BaseURL:                   "http://policy-service:8000",
		TotalTimeoutSeconds:       60,
		PackageAlias:              "rayline/test-v5",
		PackageSHA256:             hex.EncodeToString(sum[:]),
		PackageManifest:           path,
		AllowExperimentalControls: true,
		Bindings: []RaylineARCPolicyBinding{
			{ActionID: policyV5OpusAction, Worker: "arm-opus"},
			{ActionID: policyV5GLMNone, Worker: "arm-glm"},
			{ActionID: policyV5GLMUp, Worker: "arm-glm"},
		},
	}
	return cfg, decision
}

func readPolicyV5Fixture(t *testing.T) []byte {
	t.Helper()
	raw, err := os.ReadFile(policyV5Fixture)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestRaylineARCPolicyPackageV5Loads(t *testing.T) {
	cfg, decision := policyV5Decision(t, readPolicyV5Fixture(t))
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("the v5 fixture refused: %v", err)
	}
	policy := decision.Algorithm.RaylineARC.PolicyService
	action, ok := policy.PackageV5Action(policyV5GLMUp)
	if !ok || action.Model != "z-ai/glm-5.3-flash" || action.Control.Instruction == nil ||
		action.Control.Instruction.Level != "up" {
		t.Fatalf("PackageV5Action = %+v, %v", action, ok)
	}
	if policy.PackageV5RegistryDiffers() {
		t.Fatal("the fixture names the registry VSR serves")
	}
	provider, format, err := RaylineARCRegistryCell(cfg, "arm-opus")
	if err != nil || provider != "openrouter" || format != "messages" {
		t.Fatalf("arm-opus cell = %s x %s, %v", provider, format, err)
	}
}

// thinking_controls_sha256 is informational: another artifact sha loads, and
// is reported.
func TestRaylineARCPolicyPackageV5RegistryShaIsInformational(t *testing.T) {
	manifest := bytes.Replace(readPolicyV5Fixture(t),
		[]byte(`"thinking_controls_sha256": "3083a4b6`), []byte(`"thinking_controls_sha256": "0083a4b6`), 1)
	cfg, decision := policyV5Decision(t, manifest)
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("a differing registry sha refused: %v", err)
	}
	if !decision.Algorithm.RaylineARC.PolicyService.PackageV5RegistryDiffers() {
		t.Fatal("the differing registry sha was not reported")
	}
}

func TestRaylineARCPolicyPackageV5Refusals(t *testing.T) {
	fixture := readPolicyV5Fixture(t)
	for name, test := range map[string]struct {
		edit func(cfg *RouterConfig, decision *Decision)
		want string
	}{
		"a manifest whose sha is not package_sha256": {
			edit: func(_ *RouterConfig, d *Decision) {
				d.Algorithm.RaylineARC.PolicyService.PackageSHA256 = strings.Repeat("b", 64)
			},
			want: "not package_sha256",
		},
		"an experimental instruction without the opt-in": {
			edit: func(_ *RouterConfig, d *Decision) {
				d.Algorithm.RaylineARC.PolicyService.AllowExperimentalControls = false
			},
			want: "experimental only",
		},
		"a v5 binding that declares a level": {
			edit: func(_ *RouterConfig, d *Decision) { d.Algorithm.RaylineARC.PolicyService.Bindings[2].Level = "up" },
			want: "only action_id and worker",
		},
		"a v5 binding that declares a model": {
			edit: func(_ *RouterConfig, d *Decision) {
				d.Algorithm.RaylineARC.PolicyService.Bindings[0].Model = "anthropic/claude-opus-5"
			},
			want: "only action_id and worker",
		},
		"dispatch_effort on a v5 package": {
			edit: func(_ *RouterConfig, d *Decision) {
				d.Algorithm.RaylineARC.PolicyService.DispatchEffort = RaylineARCPolicyDispatchEffortProviderDefault
			},
			want: "dispatch_effort serves v4",
		},
		"an unbound package action": {
			edit: func(_ *RouterConfig, d *Decision) {
				policy := d.Algorithm.RaylineARC.PolicyService
				policy.Bindings = policy.Bindings[:2]
			},
			want: "is not bound",
		},
		"a binding to an action the package lacks": {
			edit: func(_ *RouterConfig, d *Decision) {
				d.Algorithm.RaylineARC.PolicyService.Bindings[2].ActionID = strings.Repeat("c", 64)
			},
			want: "not in the package",
		},
		"a steer on a cell that refuses instructions": {
			// GLM's Anthropic-direct Messages cell admits only native controls.
			edit: func(cfg *RouterConfig, _ *Decision) {
				params := cfg.ModelConfig["arm-glm"]
				params.PreferredEndpoints, params.APIFormat = []string{"anthropic"}, APIFormatAnthropic
				cfg.ModelConfig["arm-glm"] = params
			},
			want: "is not admitted",
		},
		"a thinking-off mismatch": {
			edit: func(_ *RouterConfig, d *Decision) {
				off := false
				d.ModelRefs[1].UseReasoning = &off
			},
			want: "use_reasoning",
		},
		"the thinking lever": {
			edit: func(_ *RouterConfig, d *Decision) {
				d.Algorithm.RaylineARC.ThinkingLever = &RaylineARCThinkingLeverConfig{
					Enabled: true, Source: RaylineARCThinkingSourcePolicy,
					Workers: map[string]RaylineARCThinkingBindingConfig{},
				}
			},
			want: "thinking_lever",
		},
	} {
		t.Run(name, func(t *testing.T) {
			cfg, decision := policyV5Decision(t, fixture)
			test.edit(cfg, &decision)
			err := validatePolicyDispatch(cfg, decision)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

// package_manifest is for v5 packages; a v4 manifest there is refused, since
// a v4 package's bindings declare their dispatch.
func TestRaylineARCPolicyPackageManifestMustBeV5(t *testing.T) {
	v4, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v4.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, decision := policyV5Decision(t, v4)
	if err := validatePolicyDispatch(cfg, decision); err == nil || !strings.Contains(err.Error(), "must be a rayline.arc-policy-package.v5") {
		t.Fatalf("a v4 manifest loaded as v5: %v", err)
	}
}
