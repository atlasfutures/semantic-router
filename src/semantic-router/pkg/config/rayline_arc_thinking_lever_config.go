package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkinglever"
)

const (
	RaylineARCThinkingSourceRule            = "rule"
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
	// Source says who chooses the level. Only "rule" is served: every
	// governed turn of this decision asks for Level.
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
func validateRaylineARCThinkingLeverConfig(cfg *RaylineARCThinkingLeverConfig) error {
	if cfg == nil || !cfg.Enabled {
		return nil
	}
	if cfg.Source != RaylineARCThinkingSourceRule {
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
	if cfg.Level == "" {
		return fmt.Errorf("level is required with source %q", RaylineARCThinkingSourceRule)
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
	if _, ok := binding.Level(level); !ok {
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
// decision cannot select. It would never steer anything, which reads as a
// lever that is on but has no effect -- usually a worker ID typo.
func validateRaylineARCThinkingWorkers(cfg *RaylineARCAlgorithmConfig, workers map[string]bool) error {
	if cfg == nil || cfg.ThinkingLever == nil || !cfg.ThinkingLever.Enabled {
		return nil
	}
	for worker := range cfg.ThinkingLever.Workers {
		if !workers[worker] {
			return fmt.Errorf("workers[%q] is not one of the decision's modelRefs", worker)
		}
	}
	return nil
}
