package config

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkinglever"
)

const (
	RaylineARCThinkingSourceRule = "rule"
	// RaylineARCThinkingSourcePolicy takes each turn's level from the policy
	// service's decision: the level its chosen action binds.
	RaylineARCThinkingSourcePolicy          = "policy"
	RaylineARCThinkingAdmissionCertified    = "certified"
	RaylineARCThinkingAdmissionExperimental = "experimental"
)

// RaylineARCThinkingLeverConfig turns on the per-turn thinking lever for one
// ARC decision.
//
// A lever writes an item into the provider-bound transcript that the client
// never sees -- a steering instruction, or a reasoning-effort update -- and
// the router replays every earlier item on later turns, so the provider's
// prompt cache survives a change of level. Which bytes realise a level for a
// worker comes from the shared thinking-level registry, never from this
// router; Workers carries that compiled binding.
type RaylineARCThinkingLeverConfig struct {
	Enabled bool `yaml:"enabled,omitempty"`
	// Source says who chooses the level. "rule" asks for Level on every
	// governed turn; "policy", the only source in the policy-service mode,
	// asks for the level of the action the service chose.
	Source string `yaml:"source,omitempty"`
	Level  string `yaml:"level,omitempty"`
	// Admission "experimental" also admits bindings the registry marks
	// experimental. The default admits certified bindings only.
	Admission string `yaml:"admission,omitempty"`
	// MinSpacingTurns is the least number of committed turns between two
	// changes of level on an on-change binding.
	MinSpacingTurns uint64 `yaml:"min_spacing_turns,omitempty"`
	// MaxLedgerEntries caps the episode ledger below its hard limit. At the
	// cap the level in force is held and nothing more is written; zero
	// selects the hard limit.
	MaxLedgerEntries int `yaml:"max_ledger_entries,omitempty"`
	// EligibilityHeader, when set, limits steering to requests that carry
	// this header with the value "1": a test episode opts in, and every other
	// conversation on a bound worker is left exactly as it was. It never
	// chooses the level. Envoy must admit it to the router and keep it from
	// the provider.
	EligibilityHeader string `yaml:"eligibility_header,omitempty"`
	// Workers binds a lever to worker IDs. A worker without a binding is
	// never steered, and its turns leave the ledger as it was.
	Workers map[string]RaylineARCThinkingBindingConfig `yaml:"workers,omitempty"`
}

// RaylineARCThinkingBindingConfig is one worker's compiled lever binding.
type RaylineARCThinkingBindingConfig struct {
	// ExportSHA256 names the registry export this binding was compiled
	// from, so a training row can be traced to its evidence.
	ExportSHA256 string                          `yaml:"export_sha256,omitempty"`
	Admission    string                          `yaml:"admission"`
	Lever        string                          `yaml:"lever"`
	Emit         string                          `yaml:"emit"`
	NeutralLevel string                          `yaml:"neutral_level,omitempty"`
	Placements   []string                        `yaml:"placements"`
	Levels       []RaylineARCThinkingLevelConfig `yaml:"levels"`
}

type RaylineARCThinkingLevelConfig struct {
	Level  string `yaml:"level"`
	Rank   int    `yaml:"rank"`
	Suffix string `yaml:"suffix,omitempty"`
	Effort string `yaml:"effort,omitempty"`
	// ControlSHA256 is the registry's digest of this level's bytes. The
	// loader recomputes it and refuses a mismatch, so the router and the
	// trained artifact cannot disagree about which action a level is.
	ControlSHA256 string `yaml:"control_sha256"`
}

// Binding converts the configured binding to the planner's form.
func (cfg RaylineARCThinkingBindingConfig) Binding() thinkinglever.Binding {
	binding := thinkinglever.Binding{
		Lever:   thinkinglever.Lever(cfg.Lever),
		Emit:    thinkinglever.EmitMode(cfg.Emit),
		Neutral: cfg.NeutralLevel,
	}
	for _, placement := range cfg.Placements {
		binding.Placements = append(binding.Placements, thinkinglever.Placement(placement))
	}
	for _, level := range cfg.Levels {
		binding.Levels = append(binding.Levels, thinkinglever.Level{
			Name: level.Level, Rank: level.Rank, Suffix: level.Suffix, Effort: level.Effort,
		})
	}
	return binding
}

// validateRaylineARCThinkingLeverConfig refuses at load every binding the
// planner would refuse per turn, so a bad binding stops the cell starting
// rather than failing a user's turn.
func validateRaylineARCThinkingLeverConfig(cfg *RaylineARCThinkingLeverConfig, policyMode bool) error {
	if cfg == nil || !cfg.Enabled {
		return nil
	}
	fromPolicy := false
	switch {
	case policyMode && cfg.Source == RaylineARCThinkingSourcePolicy:
		fromPolicy = true
	case !policyMode && cfg.Source == RaylineARCThinkingSourceRule:
	case policyMode:
		return fmt.Errorf("source %q is not served in the policy-service mode; use %q", cfg.Source, RaylineARCThinkingSourcePolicy)
	default:
		return fmt.Errorf("source %q is not served; use %q", cfg.Source, RaylineARCThinkingSourceRule)
	}
	admitExperimental := false
	switch cfg.Admission {
	case "", RaylineARCThinkingAdmissionCertified:
	case RaylineARCThinkingAdmissionExperimental:
		admitExperimental = true
	default:
		return fmt.Errorf("admission %q must be certified or experimental", cfg.Admission)
	}
	if err := validateRaylineARCThinkingLevelSource(cfg, fromPolicy); err != nil {
		return err
	}
	if len(cfg.Workers) == 0 {
		return fmt.Errorf("at least one worker binding is required")
	}
	if cfg.MaxLedgerEntries < 0 || cfg.MaxLedgerEntries > thinkinglever.MaxLedgerLength {
		return fmt.Errorf("max_ledger_entries must be between 0 and %d", thinkinglever.MaxLedgerLength)
	}
	for worker, bindingConfig := range cfg.Workers {
		if err := validateRaylineARCThinkingBinding(bindingConfig, cfg.Level, admitExperimental); err != nil {
			return fmt.Errorf("workers[%q]: %w", worker, err)
		}
	}
	return nil
}

// validateRaylineARCThinkingLevelSource checks the fields that depend on who
// chooses the level. With source policy the decision chooses every turn, so
// nothing may override it: no fixed level, no spacing that would hold an
// older level, and no eligibility header that would skip the action's steer.
func validateRaylineARCThinkingLevelSource(cfg *RaylineARCThinkingLeverConfig, fromPolicy bool) error {
	if !fromPolicy {
		if cfg.Level == "" {
			return fmt.Errorf("level is required with source %q", RaylineARCThinkingSourceRule)
		}
		return nil
	}
	if cfg.Level != "" {
		return fmt.Errorf("level must be empty with source %q: the policy decision chooses it", RaylineARCThinkingSourcePolicy)
	}
	if cfg.MinSpacingTurns != 0 {
		return fmt.Errorf("min_spacing_turns must be 0 with source %q: spacing would hold a level the decision replaced", RaylineARCThinkingSourcePolicy)
	}
	if cfg.EligibilityHeader != "" {
		return fmt.Errorf("eligibility_header is not served with source %q: every decided action is dispatched", RaylineARCThinkingSourcePolicy)
	}
	return nil
}

func validateRaylineARCThinkingBinding(
	cfg RaylineARCThinkingBindingConfig,
	level string,
	admitExperimental bool,
) error {
	switch cfg.Admission {
	case RaylineARCThinkingAdmissionCertified:
	case RaylineARCThinkingAdmissionExperimental:
		if !admitExperimental {
			return fmt.Errorf("binding is experimental and admission is certified")
		}
	default:
		return fmt.Errorf("admission %q must be certified or experimental", cfg.Admission)
	}
	binding := cfg.Binding()
	if err := binding.Validate(); err != nil {
		return err
	}
	if _, ok := binding.Level(level); level != "" && !ok {
		return fmt.Errorf("level %q is not in the binding", level)
	}
	for index, levelConfig := range cfg.Levels {
		if want := binding.ControlSHA256(binding.Levels[index]); levelConfig.ControlSHA256 != want {
			return fmt.Errorf("level %q control_sha256 does not match its bytes", levelConfig.Level)
		}
	}
	return nil
}

// validateRaylineARCThinkingWorkers refuses a binding for a worker the
// decision cannot select -- it would never steer anything, usually a worker ID
// typo -- and for a worker that does not reason. A thinking-off arm emits no
// reasoning, so a lever cannot move its depth; it could only change the
// visible answer, which is not what the lever is for.
func validateRaylineARCThinkingWorkers(cfg *RaylineARCAlgorithmConfig, modelRefs []ModelRef) error {
	if cfg == nil || cfg.ThinkingLever == nil || !cfg.ThinkingLever.Enabled {
		return nil
	}
	reasons := make(map[string]bool, len(modelRefs))
	for _, modelRef := range modelRefs {
		reasons[modelRef.Model] = modelRef.UseReasoning != nil && *modelRef.UseReasoning
	}
	for worker := range cfg.ThinkingLever.Workers {
		thinking, selectable := reasons[worker]
		if !selectable {
			return fmt.Errorf("workers[%q] is not one of the decision's modelRefs", worker)
		}
		if !thinking {
			return fmt.Errorf("workers[%q] does not reason (use_reasoning is false), so a lever cannot steer it", worker)
		}
	}
	return nil
}

// validateRaylineARCThinkingLeverTransports refuses a per_turn_effort binding
// on a worker whose provider cannot read the item it writes: a
// configuration-update message exists only on OpenRouter's Chat wire, and the
// Messages and Responses encoders refuse it, so every governed turn on such a
// worker would fail at encoding.
func validateRaylineARCThinkingLeverTransports(cfg *RouterConfig, arc *RaylineARCAlgorithmConfig) error {
	if cfg == nil || arc == nil || arc.ThinkingLever == nil || !arc.ThinkingLever.Enabled {
		return nil
	}
	for worker, binding := range arc.ThinkingLever.Workers {
		if binding.Lever != string(thinkinglever.LeverPerTurnEffort) {
			continue
		}
		if format := strings.ToLower(strings.TrimSpace(cfg.GetModelAPIFormat(worker))); format != APIFormatOpenAI {
			return fmt.Errorf("workers[%q] dispatches %s, and per_turn_effort needs OpenRouter's Chat wire", worker, format)
		}
		for _, endpoint := range cfg.GetEndpointsForModel(worker) {
			profile, err := cfg.GetProviderProfileForEndpoint(endpoint.Name)
			if err != nil || !raylineARCOpenRouterProfile(profile) {
				return fmt.Errorf("workers[%q] reaches endpoint %q, which is not OpenRouter, and per_turn_effort needs OpenRouter's Chat wire", worker, endpoint.Name)
			}
		}
	}
	return nil
}

// raylineARCOpenRouterProfile reports whether a provider profile reaches
// OpenRouter, by its type or its host.
func raylineARCOpenRouterProfile(profile *ProviderProfile) bool {
	if profile == nil {
		return false
	}
	if strings.EqualFold(strings.TrimSpace(profile.Type), "openrouter") {
		return true
	}
	parsed, err := url.Parse(profile.BaseURL)
	if err != nil {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	return host == "openrouter.ai" || strings.HasSuffix(host, ".openrouter.ai")
}
