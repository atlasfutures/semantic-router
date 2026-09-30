package config

import (
	"strings"
	"testing"
)

// The rayline_arc registry row is appended by init() in a fork file; pin it
// so a sync that drops the file fails here rather than at config load.
func TestRaylineARCRegisteredInDecisionAlgorithmRegistry(t *testing.T) {
	if !IsSupportedDecisionAlgorithmType(RaylineARCAlgorithmType) {
		t.Fatalf("%s is not a supported decision algorithm type", RaylineARCAlgorithmType)
	}
	field, ok := DecisionAlgorithmConfigField(RaylineARCAlgorithmType)
	if !ok || field != "rayline_arc" {
		t.Fatalf("DecisionAlgorithmConfigField(%q) = %q, %v; want rayline_arc, true", RaylineARCAlgorithmType, field, ok)
	}
	blocks := configuredDecisionAlgorithmBlocks(validRaylineARCDecision().Algorithm)
	if len(blocks) != 1 || blocks[0] != "rayline_arc" {
		t.Fatalf("configured blocks = %v, want [rayline_arc]", blocks)
	}
}

// The ARC decision contract is hooked into upstream's per-decision validator.
// A config that violates it must fail through that entry point, not only when
// the fork function is called directly.
func TestRaylineARCDecisionContractRunsThroughUpstreamValidator(t *testing.T) {
	decision := validRaylineARCDecision()
	decision.Adaptations = DecisionAdaptationsConfig{}
	useReasoning := false
	for index := range decision.ModelRefs {
		decision.ModelRefs[index].UseReasoning = &useReasoning
	}
	cfg := &RouterConfig{}
	cfg.ModelConfig = map[string]ModelParams{"public-arm-a": {}, "public-arm-b": {}}
	cfg.Decisions = []Decision{decision}

	err := validateDecisionModelContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), "requires adaptations.mode=bypass") {
		t.Fatalf("validateDecisionModelContracts() error = %v, want the ARC bypass contract", err)
	}
	err = validateDecisionExecution(cfg, decision)
	if err == nil || !strings.Contains(err.Error(), "requires adaptations.mode=bypass") {
		t.Fatalf("validateDecisionExecution() error = %v, want the ARC bypass contract", err)
	}
}

// hosted_tools validation joins upstream's global validator list by init().
func TestHostedToolContractsRegisteredAsGlobalValidator(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelConfig = map[string]ModelParams{"searcher": {HostedTools: []string{"web_serch"}}}
	err := runConfigContractValidators(cfg, globalConfigContractValidators)
	if err == nil || !strings.Contains(err.Error(), "hosted_tools") {
		t.Fatalf("global validators error = %v, want the hosted_tools refusal", err)
	}
}
