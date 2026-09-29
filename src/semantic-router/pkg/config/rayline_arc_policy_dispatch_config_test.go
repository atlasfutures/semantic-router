package config

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

const (
	// Trained model names, as a package names them; the workers' provider
	// model ids differ, since a worker is remappable.
	policyTestThinkModel = "think-trained"
	policyTestOffModel   = "off-trained"
	policyTestUpSuffix   = "Until the next steering instruction, reason more thoroughly before acting."
)

func policyTestString(value string) *string { return &value }

func policyTestInt(value int64) *int64 { return &value }

// policyDispatchFixture is a policy-service decision whose bindings declare
// dispatch: a thinking worker at two levels and a thinking-off worker.
func policyDispatchFixture() (*RouterConfig, Decision) {
	on, off := true, false
	cfg := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
		"arm-think": {ExternalModelIDs: map[string]string{"vllm": "vendor/think"}},
		"arm-off":   {ExternalModelIDs: map[string]string{"vllm": "vendor/off"}},
	}}}
	decision := validRaylineARCDecision()
	decision.ModelRefs = []ModelRef{
		{Model: "arm-think", ModelReasoningControl: ModelReasoningControl{UseReasoning: &on}},
		{Model: "arm-off", ModelReasoningControl: ModelReasoningControl{UseReasoning: &off}},
	}
	arc := decision.Algorithm.RaylineARC
	arc.ArtifactDir, arc.ArtifactRevision, arc.Encoder = "", "", RaylineARCEncoderConfig{}
	arc.PolicyService = &RaylineARCPolicyServiceConfig{
		BaseURL:             "http://policy-service:8000",
		TotalTimeoutSeconds: 60,
		PackageAlias:        "rayline/test",
		PackageSHA256:       strings.Repeat("a", 64),
	}
	arc.ThinkingLever = &RaylineARCThinkingLeverConfig{
		Enabled: true,
		Source:  RaylineARCThinkingSourcePolicy,
		Workers: map[string]RaylineARCThinkingBindingConfig{"arm-think": {
			Admission:    RaylineARCThinkingAdmissionCertified,
			Lever:        "prompt_steering_suffix",
			Emit:         "on_change",
			NeutralLevel: "none",
			Placements:   []string{"append_tail_user_text"},
			Levels: []RaylineARCThinkingLevelConfig{
				{Level: "none", Rank: 0, ControlSHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
				{Level: "up", Rank: 1, Suffix: policyTestUpSuffix,
					ControlSHA256: "8ffbd70c1da1d9dfb87da0b2b5bf1b95e03c4bc05b7a2f2e5ba7e4e1b5f35c5c"},
			},
		}},
	}
	// The fixture computes the control digest rather than pinning it, so the
	// lever's own digest test stays the one that pins the convention.
	binding := arc.ThinkingLever.Workers["arm-think"]
	binding.Levels[1].ControlSHA256 = binding.Binding().ControlSHA256(binding.Binding().Levels[1])
	arc.ThinkingLever.Workers["arm-think"] = binding
	arc.PolicyService.Bindings = []RaylineARCPolicyBinding{
		policyTestBinding("arm-think", "none", policyTestThinkModel, policyTestString("high"), nil, ""),
		policyTestBinding("arm-think", "up", policyTestThinkModel, policyTestString("high"), nil, policyTestUpSuffix),
		policyTestBinding("arm-off", "none", policyTestOffModel, policyTestString("none"), nil, ""),
	}
	return cfg, decision
}

func policyTestBinding(worker, level, model string, effort *string, budget *int64, suffix string) RaylineARCPolicyBinding {
	return RaylineARCPolicyBinding{
		ActionID: RaylineARCPolicyActionID(model, effort, budget, suffix),
		Worker:   worker, Level: level, Model: model, Effort: effort, ReasoningMaxTokens: budget,
	}
}

func validatePolicyDispatch(cfg *RouterConfig, decision Decision) error {
	if err := validateRaylineARCAlgorithmConfig(decision.Algorithm.RaylineARC); err != nil {
		return err
	}
	return validateRaylineARCDecisionContract(cfg, decision)
}

// The digest is pathfinder's arc_policy_contract.action_id. The first three
// ids are pathfinder's own, from the shared v4 fixture; the last was produced
// by the same Python canonical JSON over a model name that exercises every
// escape ensure_ascii writes.
func TestRaylineARCPolicyActionIDMatchesPathfinder(t *testing.T) {
	raw, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/package_manifest.v4.json")
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Actions []struct {
			ActionID           string  `json:"action_id"`
			Model              string  `json:"model"`
			Effort             *string `json:"effort"`
			ReasoningMaxTokens *int64  `json:"reasoning_max_tokens"`
			SteeringSuffix     string  `json:"steering_suffix"`
		} `json:"actions"`
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	if len(manifest.Actions) == 0 {
		t.Fatal("the fixture has no actions")
	}
	for _, action := range manifest.Actions {
		if got := RaylineARCPolicyActionID(action.Model, action.Effort, action.ReasoningMaxTokens, action.SteeringSuffix); got != action.ActionID {
			t.Fatalf("%s: action_id = %s, want %s", action.Model, got, action.ActionID)
		}
	}
	for _, golden := range []struct {
		model  string
		effort *string
		budget *int64
		suffix string
		want   string
	}{
		{"anthropic/claude-opus-5", policyTestString("high"), nil, "", "7799b8725ba01fa57e4b7effbdfb1b2c96d09878c141276eec78db4fede501b9"},
		{"qwen/qwen3.8-27b", nil, policyTestInt(4096), "Répondez brièvement — ok\n\t\"q\" \\ \U0001d11e \x7f",
			"563702649cd0744dc16599a2a9cb0d031e96709abb8b41fe84f91eb56e2f44fc"},
		{"vendor/modèle-\U0001d11e\"\\\n\x01\x7f", policyTestString("x_high"), nil, "",
			"32d6e0d4ff60f445848c695423875a574a5f1b9b7ec0089213e8799eb2906e89"},
	} {
		if got := RaylineARCPolicyActionID(golden.model, golden.effort, golden.budget, golden.suffix); got != golden.want {
			t.Fatalf("%q: action_id = %s, want %s", golden.model, got, golden.want)
		}
	}
}

func TestRaylineARCPolicyDispatchAcceptsBindingsThatReproduceTheirActions(t *testing.T) {
	cfg, decision := policyDispatchFixture()
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("valid dispatch refused: %v", err)
	}
	// The trained model may be served by any worker: remapping the action to
	// the other card is configuration, not a different action.
	remapped := decision.Algorithm.RaylineARC.PolicyService.Bindings[2]
	remapped.Worker, remapped.Level = "arm-off", "none"
	remapped.Model = policyTestThinkModel
	remapped.ActionID = RaylineARCPolicyActionID(policyTestThinkModel, remapped.Effort, nil, "")
	decision.Algorithm.RaylineARC.PolicyService.Bindings[2] = remapped
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("a remapped worker refused: %v", err)
	}
	cfg, decision = policyDispatchFixture()
	// A budget action on a reasoning worker, and a null-effort action.
	decision.Algorithm.RaylineARC.PolicyService.Bindings = append(decision.Algorithm.RaylineARC.PolicyService.Bindings,
		policyTestBinding("arm-think", "none", policyTestThinkModel, nil, policyTestInt(4096), ""),
		policyTestBinding("arm-think", "none", policyTestThinkModel, nil, nil, ""),
	)
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("budget and null-effort actions refused: %v", err)
	}
}

func TestRaylineARCPolicyDispatchRefusesWhatItCannotSend(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*RouterConfig, *Decision)
		wantErr string
	}{
		{"an action_id the declared dispatch does not digest to", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings[1].ActionID = RaylineARCPolicyActionID(
				policyTestThinkModel, policyTestString("high"), nil, "a suffix the lever does not send")
		}, "does not reproduce the action"},
		{"a steered thinking-off action", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings = append(d.Algorithm.RaylineARC.PolicyService.Bindings,
				policyTestBinding("arm-think", "up", policyTestThinkModel, policyTestString("none"), nil, policyTestUpSuffix))
		}, "thinking-off action (effort none) cannot be steered"},
		{"a thinking-off action on a reasoning worker", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings[0] = policyTestBinding(
				"arm-think", "none", policyTestThinkModel, policyTestString("none"), nil, "")
		}, "use_reasoning is true"},
		{"an effort on a thinking-off worker", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings[2] = policyTestBinding(
				"arm-off", "none", policyTestOffModel, policyTestString("low"), nil, "")
		}, "must state effort none"},
		{"a null-effort action on a thinking-off worker", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings[2] = policyTestBinding(
				"arm-off", "none", policyTestOffModel, nil, nil, "")
		}, "must state effort none"},
		{"a level the worker's lever does not bind", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings[2].Level = "up"
		}, `level "up" needs a thinking_lever binding`},
		{"a level missing from the lever binding", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings[1].Level = "down"
		}, `level "down" is not in worker "arm-think"`},
		{"an effort and a budget on a Chat worker", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings[0] = policyTestBinding(
				"arm-think", "none", policyTestThinkModel, policyTestString("high"), policyTestInt(4096), "")
		}, "refuses an effort and a reasoning budget together"},
		{"a worker_thinking base beside declared dispatch", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.WorkerThinking = map[string]RaylineARCWorkerThinkingConfig{
				"arm-think": {Level: "high", Wire: RaylineARCWorkerThinkingEffort, Effort: "high"},
			}
		}, "worker_thinking cannot be set"},
		{"bindings that only partly declare dispatch", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings[2].Model = ""
			d.Algorithm.RaylineARC.PolicyService.Bindings[2].Effort = nil
		}, "either every binding declares"},
		{"a policy lever over bindings that declare nothing", func(_ *RouterConfig, d *Decision) {
			for index := range d.Algorithm.RaylineARC.PolicyService.Bindings {
				binding := &d.Algorithm.RaylineARC.PolicyService.Bindings[index]
				binding.Model, binding.Effort = "", nil
			}
		}, "needs bindings that declare"},
		{"a rule lever in the policy-service mode", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.ThinkingLever.Source = RaylineARCThinkingSourceRule
			d.Algorithm.RaylineARC.ThinkingLever.Level = "up"
		}, "not served in the policy-service mode"},
		{"a fixed level with source policy", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.ThinkingLever.Level = "up"
		}, "level must be empty"},
		{"spacing with source policy", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.ThinkingLever.MinSpacingTurns = 2
		}, "min_spacing_turns must be 0"},
		{"an eligibility header with source policy", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.ThinkingLever.EligibilityHeader = "x-rayline-thinking-test"
		}, "eligibility_header is not served"},
		{"a malformed effort", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings[0].Effort = policyTestString("High")
		}, "plain effort name"},
		{"a connect timeout beyond the total", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.ConnectTimeoutSeconds = 61
		}, "connect_timeout_seconds"},
		{"an admission cap beyond the limit", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.MaxInflightCalls = maxRaylineARCInflightEncoderCalls + 1
		}, "max_inflight_calls"},
		{"a non-positive budget", func(_ *RouterConfig, d *Decision) {
			d.Algorithm.RaylineARC.PolicyService.Bindings[0].ReasoningMaxTokens = policyTestInt(0)
		}, "must be positive"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			cfg, decision := policyDispatchFixture()
			test.mutate(cfg, &decision)
			err := validatePolicyDispatch(cfg, decision)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

// A refusal names the action, so the operator can find it in the package.
func TestRaylineARCPolicyDispatchRefusalNamesTheAction(t *testing.T) {
	cfg, decision := policyDispatchFixture()
	steered := policyTestBinding("arm-think", "up", policyTestThinkModel, policyTestString("none"), nil, policyTestUpSuffix)
	decision.Algorithm.RaylineARC.PolicyService.Bindings = append(decision.Algorithm.RaylineARC.PolicyService.Bindings, steered)
	err := validatePolicyDispatch(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), steered.ActionID) {
		t.Fatalf("error = %v, want it to name action %s", err, steered.ActionID)
	}
}

// Decision-only bindings, as the first policy-service cell renders them, keep
// loading unchanged.
func TestRaylineARCPolicyBindingsWithoutDispatchStillLoad(t *testing.T) {
	cfg, decision := policyDispatchFixture()
	decision.Algorithm.RaylineARC.ThinkingLever = nil
	for index := range decision.Algorithm.RaylineARC.PolicyService.Bindings {
		binding := &decision.Algorithm.RaylineARC.PolicyService.Bindings[index]
		binding.Model, binding.Effort = "", nil
	}
	if err := validatePolicyDispatch(cfg, decision); err != nil {
		t.Fatalf("decision-only bindings refused: %v", err)
	}
}
