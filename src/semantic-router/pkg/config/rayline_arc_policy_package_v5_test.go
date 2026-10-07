package config

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The shared v5 fixture (pathfinder 9e6e236): an Opus native-only action and
// two GLM actions (the neutral level and "up"), all at the default base, under
// registry 3083a4b6...cff2. Actions name trained models (claude-opus-5,
// glm-5.3-flash), as published packages do; the workers serve, and the
// registry keys, provider model ids. Both models' OpenRouter cells admit their
// instructions experimentally.
const (
	policyV5Fixture      = "../selection/raylinearc/testdata/policy_service/package_manifest.v5.json"
	policyV5Canonical    = "../selection/raylinearc/testdata/policy_service/package_manifest.v5.canonical_v1.json"
	policyV5Yarn         = "../selection/raylinearc/testdata/policy_service/package_manifest.v5.yarn.json"
	policyV5OpusAction   = "864cc7a8eab47911b61a2b3b5d79dcb8ed81c8ae9f8a90e941aea5bf38fc21d8"
	policyV5GLMNone      = "20b6cddd7ae7cedc3ba8321cac7d962dca24b3c98dcf0408076adf40549e4a80"
	policyV5GLMUp        = "cf5bc08ecfab91cdc1e26c1c2b98e7256e4a9e64183210ceb619e9c4745ff055"
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
			// The served provider model ids, which admission keys on; the
			// endpoints carry no type, so they resolve under "vllm".
			"arm-opus": {
				PreferredEndpoints: []string{"openrouter"}, APIFormat: APIFormatAnthropic,
				ExternalModelIDs: map[string]string{"vllm": "anthropic/claude-opus-5"},
			},
			"arm-glm": {
				PreferredEndpoints: []string{"openrouter"}, APIFormat: APIFormatOpenAI,
				ExternalModelIDs: map[string]string{"vllm": "z-ai/glm-5.3-flash"},
			},
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
		ModelSchedule:             RaylineARCModelScheduleTaskTurnCompaction,
		TrainedModels:             map[string]string{"arm-opus": "claude-opus-5", "arm-glm": "glm-5.3-flash"},
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
	if !ok || action.Model != "glm-5.3-flash" || action.Control.Instruction == nil ||
		action.Control.Instruction.Level != "up" {
		t.Fatalf("PackageV5Action = %+v, %v", action, ok)
	}
	if policy.PackageV5RegistryDiffers() {
		t.Fatal("the fixture names the registry VSR serves")
	}
	provider, err := RaylineARCRegistryProvider(cfg, "arm-opus")
	if err != nil || provider != "openrouter" {
		t.Fatalf("arm-opus provider = %s, %v", provider, err)
	}
}

// A worker's target format is chosen per request from its accepted formats,
// so its control must be admitted on the cell of every one: Opus's
// OpenRouter Messages cell admits the steer, and its Chat cell does not.
func TestRaylineARCPolicyPackageV5AdmitsEveryAcceptedFormat(t *testing.T) {
	// The fixture's GLM "up" control, as an Opus action; action ids are
	// opaque to VSR.
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
	policy := decision.Algorithm.RaylineARC.PolicyService
	policy.Bindings[2].Worker = "arm-opus"
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("a Messages-only Opus worker refused the steer: %v", err)
	}
	params := cfg.ModelConfig["arm-opus"]
	params.AcceptedFormats = []string{APIFormatAnthropic, APIFormatOpenAI}
	cfg.ModelConfig["arm-opus"] = params
	err := validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "is not admitted for anthropic/claude-opus-5 on openrouterxchat") {
		t.Fatalf("a worker accepting an unverified Chat cell loaded: %v", err)
	}
	// Every Responses cell in registry 3083a4b6 refuses instructions, so a
	// steer on a worker that accepts Responses is refused on that cell.
	params.AcceptedFormats = []string{APIFormatAnthropic, APIFormatResponses}
	cfg.ModelConfig["arm-opus"] = params
	err = validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "is not admitted for anthropic/claude-opus-5 on openrouterxresponses") {
		t.Fatalf("a steer on a worker accepting Responses loaded: %v", err)
	}
}

// v5 controls are served on Responses workers where the registry admits
// them: every Responses cell admits the native-default control (Opus's
// action), and none admits an instruction.
func TestRaylineARCPolicyPackageV5OnResponsesWorkers(t *testing.T) {
	cfg, decision := policyV5Decision(t, readPolicyV5Fixture(t))
	for _, formats := range [][]string{{APIFormatResponses}, {APIFormatResponses, APIFormatAnthropic}} {
		params := cfg.ModelConfig["arm-opus"]
		params.APIFormat, params.AcceptedFormats = formats[0], formats
		cfg.ModelConfig["arm-opus"] = params
		if err := validatePolicyDispatch(cfg, decision); err != nil {
			t.Fatalf("the native-only Opus action on a %v worker refused: %v", formats, err)
		}
	}

	params := cfg.ModelConfig["arm-glm"]
	params.APIFormat, params.AcceptedFormats = APIFormatResponses, nil
	cfg.ModelConfig["arm-glm"] = params
	err := validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "is not admitted for z-ai/glm-5.3-flash on openrouterxresponses") {
		t.Fatalf("GLM's steered actions on a Responses worker: %v", err)
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
		"a model_schedule the package does not declare": {
			edit: func(_ *RouterConfig, d *Decision) { d.Algorithm.RaylineARC.PolicyService.ModelSchedule = "" },
			want: "the package declares",
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

// An episode keeps a bounded number of placers, one per worker and control
// shape; bindings that could need more are refused rather than evicting a
// placer whose instructions still need replaying.
func TestRaylineARCPolicyPackageV5BoundsControlShapes(t *testing.T) {
	var manifest map[string]any
	if err := json.Unmarshal(readPolicyV5Fixture(t), &manifest); err != nil {
		t.Fatal(err)
	}
	glm := manifest["actions"].([]any)[2].(map[string]any)
	var actions []any
	for index := 0; index < 17; index++ {
		action := map[string]any{}
		for key, value := range glm {
			action[key] = value
		}
		action["action_id"] = fmt.Sprintf("%064x", index+1)
		actions = append(actions, action)
	}
	manifest["actions"] = actions
	manifest["decision"].(map[string]any)["fallback_action_id"] = fmt.Sprintf("%064x", 1)
	raw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	cfg, decision := policyV5Decision(t, raw)
	on := true
	decision.ModelRefs = nil
	policy := decision.Algorithm.RaylineARC.PolicyService
	policy.Bindings = nil
	policy.TrainedModels = map[string]string{}
	for index := 0; index < 17; index++ {
		worker := fmt.Sprintf("arm-%d", index)
		cfg.ModelConfig[worker] = ModelParams{
			PreferredEndpoints: []string{"openrouter"}, APIFormat: APIFormatOpenAI,
			ExternalModelIDs: map[string]string{"vllm": "z-ai/glm-5.3-flash"},
		}
		policy.TrainedModels[worker] = "glm-5.3-flash"
		decision.ModelRefs = append(decision.ModelRefs, ModelRef{Model: worker, ModelReasoningControl: ModelReasoningControl{UseReasoning: &on}})
		policy.Bindings = append(policy.Bindings, RaylineARCPolicyBinding{ActionID: fmt.Sprintf("%064x", index+1), Worker: worker})
	}
	if err := validatePolicyDispatch(cfg, decision); err == nil || !strings.Contains(err.Error(), "control shapes") {
		t.Fatalf("17 control shapes loaded: %v", err)
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

// A configured package stating the optional encoding_profile members loads
// through startup validation when each holds its contract value, and is
// refused at startup otherwise.
func TestRaylineARCPolicyPackageV5EncodingProfileMembersAtStartup(t *testing.T) {
	const anchor = `"tool_definitions": "include_recorded",`
	fixture := readPolicyV5Fixture(t)
	if !bytes.Contains(fixture, []byte(anchor)) {
		t.Fatalf("the fixture no longer holds %q", anchor)
	}
	for members, accepted := range map[string]bool{
		`"conversation": "canonical_v1", "harness_injections": "strip_claude_code_2_1_v1",`: true,
		`"conversation": "canonical_v1",`: true,
		`"conversation": "recorded",`:     false,
		`"harness_injections": null,`:     false,
		`"harness_shell": "include",`:     true,
		`"harness_shell": "include_v2",`:  true,
		`"harness_shell": "include_v3",`:  false,
	} {
		manifest := bytes.Replace(fixture, []byte(anchor), []byte(anchor+" "+members), 1)
		cfg, decision := policyV5Decision(t, manifest)
		err := validatePolicyDispatch(cfg, decision)
		if accepted && err != nil {
			t.Errorf("%s: refused: %v", members, err)
		}
		if !accepted && (err == nil || !strings.Contains(err.Error(), "encoding_profile")) {
			t.Errorf("%s: err = %v, want an encoding_profile refusal", members, err)
		}
	}
}

// A two-stage package (pathfinder's stage_one part) loads through startup
// validation with stage_one carried opaque, and a stated null or a
// non-object stage_one is refused at startup (memex-desktop#7291).
func TestRaylineARCPolicyPackageV5TwoStageAtStartup(t *testing.T) {
	fixture := readPolicyV5Fixture(t)
	const part = `{"schema_version": "rayline.arc-stage-one.v1", "offered": [{"model": "glm-5.3-flash", "effort": null}], "proxy_arms": {}}`
	for value, accepted := range map[string]bool{part: true, `null`: false, `[]`: false, `"two"`: false} {
		manifest := bytes.Replace(fixture, []byte("{"), []byte(`{"stage_one": `+value+`,`), 1)
		cfg, decision := policyV5Decision(t, manifest)
		err := validatePolicyDispatch(cfg, decision)
		if accepted && err != nil {
			t.Errorf("stage_one %s: refused: %v", value, err)
		}
		if !accepted && (err == nil || !strings.Contains(err.Error(), "stage_one")) {
			t.Errorf("stage_one %s: err = %v, want a stage_one refusal", value, err)
		}
	}
}

// An action's model is the trained name, decoupled from providers (as in the
// fixture and published packages such as c27e1796); admission keys on the
// model the bound worker serves.
func TestRaylineARCPolicyPackageV5AdmitsOnTheServedModel(t *testing.T) {
	fixture := readPolicyV5Fixture(t)
	if bytes.Contains(fixture, []byte(`"model": "anthropic/`)) || bytes.Contains(fixture, []byte(`"model": "z-ai/`)) {
		t.Fatal("the fixture names a provider-qualified model")
	}
	cfg, decision := policyV5Decision(t, fixture)
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("trained action names refused: %v", err)
	}
	if action, _ := decision.Algorithm.RaylineARC.PolicyService.PackageV5Action(policyV5GLMUp); action.Model != "glm-5.3-flash" {
		t.Fatalf("action model = %q, want the trained name", action.Model)
	}

	// A worker serving a model the registry has no cell for is refused, even
	// though the action's own model has one.
	cfg, decision = policyV5Decision(t, readPolicyV5Fixture(t))
	opus := cfg.ModelConfig["arm-opus"]
	opus.ExternalModelIDs = map[string]string{"vllm": "vendor/unregistered"}
	cfg.ModelConfig["arm-opus"] = opus
	if err := validatePolicyDispatch(cfg, decision); err == nil || !strings.Contains(err.Error(), "arm-opus") {
		t.Fatalf("an unregistered served model: err = %v", err)
	}
}

// Which trained model a worker serves is declared, never inferred.
func TestRaylineARCPolicyPackageV5TrainedModelsAreDeclared(t *testing.T) {
	for name, test := range map[string]struct {
		trained map[string]string
		want    string
	}{
		"a worker without a declaration": {map[string]string{"arm-glm": "glm-5.3-flash"}, `declares no trained model for worker "arm-opus"`},
		// The served provider model id is not the trained name.
		"a worker declaring its served id": {
			map[string]string{"arm-opus": "claude-opus-5", "arm-glm": "z-ai/glm-5.3-flash"}, `serves trained model "z-ai/glm-5.3-flash"`,
		},
		"a declaration no binding uses": {
			map[string]string{"arm-opus": "claude-opus-5", "arm-glm": "glm-5.3-flash", "arm-spare": "x"}, `"arm-spare", which no binding uses`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			cfg, decision := policyV5Decision(t, readPolicyV5Fixture(t))
			decision.Algorithm.RaylineARC.PolicyService.TrainedModels = test.trained
			if err := validatePolicyDispatch(cfg, decision); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err = %v, want %q", err, test.want)
			}
		})
	}
}

// pathfinder's canonical_v1 fixture states both optional encoding_profile
// members; it loads through startup validation under the same bindings.
func TestRaylineARCPolicyPackageV5CanonicalFixtureLoads(t *testing.T) {
	raw, err := os.ReadFile(policyV5Canonical)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"conversation": "canonical_v1"`)) ||
		!bytes.Contains(raw, []byte(`"harness_injections": "strip_claude_code_2_1_v1"`)) {
		t.Fatal("the canonical_v1 fixture no longer states both encoding_profile members")
	}
	cfg, decision := policyV5Decision(t, raw)
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("the canonical_v1 fixture refused: %v", err)
	}
}

// pathfinder's ARC 0.4 fixture (a depth-12 YaRN encoder: layer block_11, rope
// yarn-f4, max_tokens 1010000) loads through startup validation under the same
// bindings, as a configured package_manifest does.
func TestRaylineARCPolicyPackageV5YarnFixtureLoads(t *testing.T) {
	raw, err := os.ReadFile(policyV5Yarn)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"rope": "yarn-f4"`)) || !bytes.Contains(raw, []byte(`"layer": "block_11"`)) {
		t.Fatal("the YaRN fixture no longer states rope and layer")
	}
	cfg, decision := policyV5Decision(t, raw)
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("the YaRN fixture refused: %v", err)
	}
}
