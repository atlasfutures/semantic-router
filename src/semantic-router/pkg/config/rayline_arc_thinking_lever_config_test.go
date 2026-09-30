package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func validThinkingLeverConfig() *RaylineARCThinkingLeverConfig {
	return &RaylineARCThinkingLeverConfig{
		Enabled: true,
		Source:  RaylineARCThinkingSourceRule,
		Level:   "down",
		Workers: map[string]RaylineARCThinkingBindingConfig{
			"z-ai/glm-5.3-flash@default": {
				Admission:  RaylineARCThinkingAdmissionCertified,
				Lever:      "prompt_steering_suffix",
				Emit:       "on_change",
				Placements: []string{"append_tail_user_text", "insert_user_after_tool_run"},
				Levels: []RaylineARCThinkingLevelConfig{
					{Level: "none", Rank: 0, ControlSHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
					{
						Level: "down", Rank: -1, Suffix: "Until the next steering instruction, use minimal deliberation.",
						ControlSHA256: "99fca31cf4f665d12f6695cb8c46311be0cc6e4a34829347d4cc9298b24cccf6",
					},
					{
						Level: "up", Rank: 1, Suffix: "Until the next steering instruction, reason more thoroughly.",
						ControlSHA256: "68a03386f70205c97e7a0002afa023c2b99b8ba98487a134c49824dd85d427be",
					},
				},
			},
		},
	}
}

func validateThinkingLeverDecision(t *testing.T, lever *RaylineARCThinkingLeverConfig) error {
	t.Helper()
	decision := validRaylineARCDecision()
	decision.Algorithm.RaylineARC.ThinkingLever = lever
	return validateDecisionAlgorithmConfig(decision.Name, decision.ModelRefs, decision.Algorithm)
}

func TestRaylineARCThinkingLeverIsOptional(t *testing.T) {
	if err := validateThinkingLeverDecision(t, nil); err != nil {
		t.Fatalf("absent block: %v", err)
	}
	// A disabled block is not validated, so an operator can stage bindings
	// before switching the lever on.
	if err := validateThinkingLeverDecision(t, &RaylineARCThinkingLeverConfig{Source: "anything"}); err != nil {
		t.Fatalf("disabled block: %v", err)
	}
	if err := validateThinkingLeverDecision(t, validThinkingLeverConfig()); err != nil {
		t.Fatalf("valid block: %v", err)
	}
}

func TestRaylineARCThinkingLeverRejectsWhatThePlannerWould(t *testing.T) {
	const worker = "z-ai/glm-5.3-flash@default"
	cases := []struct {
		name    string
		mutate  func(*RaylineARCThinkingLeverConfig)
		wantErr string
	}{
		{"unserved source", func(c *RaylineARCThinkingLeverConfig) { c.Source = "artifact" }, "is not served"},
		{"no level", func(c *RaylineARCThinkingLeverConfig) { c.Level = "" }, "level is required"},
		{"no workers", func(c *RaylineARCThinkingLeverConfig) { c.Workers = nil }, "worker binding"},
		{"unknown admission", func(c *RaylineARCThinkingLeverConfig) { c.Admission = "maybe" }, "admission"},
		{"level missing from a binding", func(c *RaylineARCThinkingLeverConfig) { c.Level = "max" }, "not in the binding"},
		{"experimental binding under certified admission", func(c *RaylineARCThinkingLeverConfig) {
			binding := c.Workers[worker]
			binding.Admission = RaylineARCThinkingAdmissionExperimental
			c.Workers[worker] = binding
		}, "experimental"},
		{"control digest that does not match the bytes", func(c *RaylineARCThinkingLeverConfig) {
			binding := c.Workers[worker]
			binding.Levels = append([]RaylineARCThinkingLevelConfig(nil), binding.Levels...)
			binding.Levels[1].Suffix = "Edited after the registry hashed it."
			c.Workers[worker] = binding
		}, "control_sha256 does not match"},
		{"ledger cap above the hard limit", func(c *RaylineARCThinkingLeverConfig) { c.MaxLedgerEntries = 100000 }, "max_ledger_entries"},
		{"placement of the other lever", func(c *RaylineARCThinkingLeverConfig) {
			binding := c.Workers[worker]
			binding.Placements = []string{"system_before_governed_turn"}
			c.Workers[worker] = binding
		}, "does not fit"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			lever := validThinkingLeverConfig()
			test.mutate(lever)
			err := validateThinkingLeverDecision(t, lever)
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestRaylineARCThinkingLeverAdmitsExperimentalWhenAsked(t *testing.T) {
	lever := validThinkingLeverConfig()
	lever.Admission = RaylineARCThinkingAdmissionExperimental
	binding := lever.Workers["z-ai/glm-5.3-flash@default"]
	binding.Admission = RaylineARCThinkingAdmissionExperimental
	lever.Workers["z-ai/glm-5.3-flash@default"] = binding
	if err := validateThinkingLeverDecision(t, lever); err != nil {
		t.Fatalf("experimental admission: %v", err)
	}
}

func TestRaylineARCThinkingLeverRoundTripsStrictly(t *testing.T) {
	algorithm := validRaylineARCDecision().Algorithm
	algorithm.RaylineARC.ThinkingLever = validThinkingLeverConfig()
	encoded, err := yaml.Marshal(algorithm)
	if err != nil {
		t.Fatal(err)
	}
	var decoded AlgorithmConfig
	if err := yaml.UnmarshalStrict(encoded, &decoded); err != nil {
		t.Fatalf("strict decode: %v", err)
	}
	if !reflect.DeepEqual(algorithm.RaylineARC.ThinkingLever, decoded.RaylineARC.ThinkingLever) {
		t.Fatalf("thinking_lever did not round-trip:\n%s", encoded)
	}
}

// A per_turn_effort lever sets a reasoning effort, which a thinking-off
// worker's off signal would contradict.
func TestRaylineARCPerTurnEffortLeverNeedsAReasoningWorker(t *testing.T) {
	off := false
	decision := validRaylineARCDecision()
	decision.ModelRefs = []ModelRef{
		{Model: "arm-off", ModelReasoningControl: ModelReasoningControl{UseReasoning: &off}},
		{Model: "arm-b"},
	}
	lever := validThinkingLeverConfig()
	lever.Workers = map[string]RaylineARCThinkingBindingConfig{"arm-off": {Lever: "per_turn_effort"}}
	decision.Algorithm.RaylineARC.ThinkingLever = lever
	err := validateRaylineARCDecisionContract(&RouterConfig{}, decision)
	if err == nil || !strings.Contains(err.Error(), "per_turn_effort lever cannot set its effort") {
		t.Fatalf("error = %v", err)
	}
}

func TestRaylineARCThinkingLeverBindsOnlySelectableThinkingWorkers(t *testing.T) {
	on, off := true, false
	binding := validThinkingLeverConfig().Workers["z-ai/glm-5.3-flash@default"]
	decision := validRaylineARCDecision()
	decision.ModelRefs = []ModelRef{
		{Model: "arm-on", ModelReasoningControl: ModelReasoningControl{UseReasoning: &on}},
		{Model: "arm-off", ModelReasoningControl: ModelReasoningControl{UseReasoning: &off}},
	}
	for worker, wantErr := range map[string]string{
		"arm-on":   "",
		"arm-off":  "", // a steering suffix may bind a thinking-off worker
		"arm-typo": "not one of the decision's modelRefs",
	} {
		lever := validThinkingLeverConfig()
		lever.Workers = map[string]RaylineARCThinkingBindingConfig{worker: binding}
		decision.Algorithm.RaylineARC.ThinkingLever = lever
		err := validateRaylineARCDecisionContract(&RouterConfig{}, decision)
		if (wantErr == "" && err != nil) || (wantErr != "" && (err == nil || !strings.Contains(err.Error(), wantErr))) {
			t.Fatalf("%s: error = %v, want %q", worker, err, wantErr)
		}
	}
}

func TestRaylineARCWorkerThinkingValidation(t *testing.T) {
	valid := map[string]RaylineARCWorkerThinkingConfig{
		"effort":  {Level: "high", Wire: RaylineARCWorkerThinkingEffort, Effort: "high"},
		"budget":  {Level: "cap_4096", Wire: RaylineARCWorkerThinkingBudget, MaxTokens: 4096},
		"default": {Level: "default", Wire: RaylineARCWorkerThinkingProviderDefault},
	}
	if err := validateRaylineARCWorkerThinking(valid); err != nil {
		t.Fatalf("valid bases refused: %v", err)
	}
	for name, base := range map[string]RaylineARCWorkerThinkingConfig{
		"effort with a budget":   {Level: "high", Wire: "effort", Effort: "high", MaxTokens: 10},
		"budget without tokens":  {Level: "low", Wire: "budget"},
		"default with an effort": {Level: "default", Wire: "provider_default", Effort: "max"},
		"unknown wire":           {Level: "high", Wire: "both"},
		"non-canonical effort":   {Level: "high", Wire: "effort", Effort: "High"},
		"missing level":          {Wire: "provider_default"},
	} {
		t.Run(name, func(t *testing.T) {
			if validateRaylineARCWorkerThinking(map[string]RaylineARCWorkerThinkingConfig{"w": base}) == nil {
				t.Fatal("invalid base accepted")
			}
		})
	}
}

func TestRaylineARCWorkerThinkingNeedsAThinkingModelRef(t *testing.T) {
	on, off := true, false
	decision := validRaylineARCDecision()
	decision.ModelRefs = []ModelRef{
		{Model: "arm-on", ModelReasoningControl: ModelReasoningControl{UseReasoning: &on}},
		{Model: "arm-off", ModelReasoningControl: ModelReasoningControl{UseReasoning: &off}},
	}
	base := RaylineARCWorkerThinkingConfig{Level: "high", Wire: "effort", Effort: "high"}
	for worker, wantErr := range map[string]string{"arm-on": "", "arm-off": "does not reason", "arm-typo": "not one of"} {
		decision.Algorithm.RaylineARC.WorkerThinking = map[string]RaylineARCWorkerThinkingConfig{worker: base}
		err := validateRaylineARCDecisionContract(&RouterConfig{}, decision)
		if (wantErr == "" && err != nil) || (wantErr != "" && (err == nil || !strings.Contains(err.Error(), wantErr))) {
			t.Fatalf("%s: error = %v, want %q", worker, err, wantErr)
		}
	}
}

func TestRaylineARCThinkingLeverEligibilityHeaderIsAFieldName(t *testing.T) {
	for header, ok := range map[string]bool{
		"x-rayline-thinking-test": true,
		"X-Upper":                 false,
		"has space":               false,
	} {
		lever := validThinkingLeverConfig()
		lever.EligibilityHeader = header
		err := validateThinkingLeverDecision(t, lever)
		if ok != (err == nil) {
			t.Fatalf("%q: error = %v", header, err)
		}
	}
	decision := validRaylineARCDecision()
	lever := validThinkingLeverConfig()
	lever.EligibilityHeader = decision.Algorithm.RaylineARC.Episode.IDHeader
	if validateThinkingLeverDecision(t, lever) == nil {
		t.Fatal("the episode id header was accepted as the eligibility header")
	}
}

// per_turn_effort writes a configuration-update message, which only
// OpenRouter's Chat wire reads; any other worker is refused at load.
func TestRaylineARCPerTurnEffortNeedsOpenRouterChat(t *testing.T) {
	routerWith := func(apiFormat string, profile ProviderProfile) *RouterConfig {
		return &RouterConfig{BackendModels: BackendModels{
			ModelConfig:      map[string]ModelParams{"worker": {PreferredEndpoints: []string{"backend"}, APIFormat: apiFormat}},
			VLLMEndpoints:    []VLLMEndpoint{{Name: "backend", Address: "provider", Port: 443, ProviderProfileName: "profile"}},
			ProviderProfiles: map[string]ProviderProfile{"profile": profile},
		}}
	}
	arc := &RaylineARCAlgorithmConfig{ThinkingLever: &RaylineARCThinkingLeverConfig{
		Enabled: true,
		Workers: map[string]RaylineARCThinkingBindingConfig{"worker": {Lever: "per_turn_effort"}},
	}}
	openRouter := ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"}
	if err := validateRaylineARCThinkingLeverTransports(routerWith(APIFormatOpenAI, openRouter), arc); err != nil {
		t.Fatalf("OpenRouter Chat refused: %v", err)
	}
	for name, cfg := range map[string]*RouterConfig{
		"a Messages worker":            routerWith(APIFormatAnthropic, ProviderProfile{Type: "anthropic", BaseURL: "https://api.anthropic.com"}),
		"a Chat worker off OpenRouter": routerWith(APIFormatOpenAI, ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"}),
	} {
		if err := validateRaylineARCThinkingLeverTransports(cfg, arc); err == nil || !strings.Contains(err.Error(), "per_turn_effort") {
			t.Fatalf("%s: error = %v", name, err)
		}
	}
	// A suffix lever writes text, which every wire carries.
	arc.ThinkingLever.Workers["worker"] = RaylineARCThinkingBindingConfig{Lever: "prompt_steering_suffix"}
	if err := validateRaylineARCThinkingLeverTransports(routerWith(APIFormatAnthropic, ProviderProfile{Type: "anthropic"}), arc); err != nil {
		t.Fatalf("a suffix lever on Messages refused: %v", err)
	}
}

// A worker_thinking base travels only on OpenRouter's Chat reasoning object;
// a worker it could not reach is refused rather than silently ignored.
func TestRaylineARCWorkerThinkingNeedsTheReasoningObject(t *testing.T) {
	routerWith := func(apiFormat string, profile ProviderProfile) *RouterConfig {
		return &RouterConfig{BackendModels: BackendModels{
			ModelConfig:      map[string]ModelParams{"worker": {PreferredEndpoints: []string{"backend"}, APIFormat: apiFormat}},
			VLLMEndpoints:    []VLLMEndpoint{{Name: "backend", Address: "provider", Port: 443, ProviderProfileName: "profile"}},
			ProviderProfiles: map[string]ProviderProfile{"profile": profile},
		}}
	}
	arc := &RaylineARCAlgorithmConfig{WorkerThinking: map[string]RaylineARCWorkerThinkingConfig{
		"worker": {Level: "high", Wire: RaylineARCWorkerThinkingEffort, Effort: "high"},
	}}
	if err := validateRaylineARCWorkerThinkingTransports(routerWith(APIFormatOpenAI, ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"}), arc); err != nil {
		t.Fatalf("OpenRouter Chat refused: %v", err)
	}
	// The dev cell's shape: a provider_default base on an openrouter-type
	// Chat backend (router-infra vsr.cloudrun.arc.redis.yaml on the
	// thinking-lever dev pin).
	providerDefault := &RaylineARCAlgorithmConfig{WorkerThinking: map[string]RaylineARCWorkerThinkingConfig{
		"worker": {Level: "default", Wire: RaylineARCWorkerThinkingProviderDefault},
	}}
	if err := validateRaylineARCWorkerThinkingTransports(routerWith(APIFormatOpenAI, ProviderProfile{Type: "openrouter", BaseURL: "https://openrouter.ai/api/v1"}), providerDefault); err != nil {
		t.Fatalf("provider_default on OpenRouter Chat refused: %v", err)
	}
	for name, cfg := range map[string]*RouterConfig{
		"a Messages worker":        routerWith(APIFormatAnthropic, ProviderProfile{Type: "anthropic", BaseURL: "https://api.anthropic.com"}),
		"a chat-template provider": routerWith(APIFormatOpenAI, ProviderProfile{Type: "vllm", BaseURL: "http://vllm.internal:8000/v1"}),
	} {
		if err := validateRaylineARCWorkerThinkingTransports(cfg, arc); err == nil {
			t.Fatalf("%s: accepted", name)
		}
	}
}
