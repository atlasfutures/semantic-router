package config

import "testing"

// The reference configuration disables the stores that hold verbatim prompt and
// completion text, so a deployment built from it retains none on the ARC path.
// Asserting the YAML literal alone would be circular -- it would restate the
// file against itself. These tests parse the file through the real loader and
// assert the resolved runtime policy, which is what the router actually reads:
// a loader default that overrode one of these flags, or a per-decision plugin
// that re-enabled it, fails here.
//
// Scope note: response_api stays enabled on purpose. Disabling it returns 404
// for the whole /v1/responses endpoint at header validation, and ARC serves
// that format, so switching it off is a functional regression rather than a
// retention-only change. It is therefore deliberately absent below.

func TestReferenceConfigDisablesContentRetainingStores(t *testing.T) {
	cfg, err := ParseYAMLBytesWithoutEnvExpansion(readReferenceConfigYAML(t))
	if err != nil {
		t.Fatalf("failed to parse config/config.yaml: %v", err)
	}

	if cfg.Memory.Enabled {
		t.Error("stores.memory must stay disabled: it stores each turn verbatim with no TTL")
	}
	// auto_store is the field that reaches ARC: the response path falls back to
	// this global value for any decision without a memory plugin.
	if cfg.Memory.AutoStore {
		t.Error("stores.memory.auto_store must stay disabled: it is the global fallback for decisions without a memory plugin, which includes every rayline_arc decision")
	}
	if cfg.SemanticCache.Enabled {
		t.Error("stores.response_cache must stay disabled: it stores the query text and both bodies")
	}
	if cfg.RouterReplay.Enabled {
		t.Error("services.router_replay must stay disabled: it persists both bodies, the prompt, tool definitions and every tool-call argument and output")
	}
}

// TestReferenceConfigRaylineARCDecisionsPersistNothing asserts the resolved
// per-decision policy for every ARC decision, rather than the global flags. The
// global switch and the per-decision plugin are layered by
// EffectiveRouterReplayConfig, and the plugin wins, so a decision could persist
// prompt text with replay globally off. The config validator already refuses an
// ARC decision whose effective replay config is enabled; this covers the same
// contract from the reference config, and additionally checks that no ARC
// decision opts into the caching or memory plugins.
func TestReferenceConfigRaylineARCDecisionsPersistNothing(t *testing.T) {
	cfg, err := ParseYAMLBytesWithoutEnvExpansion(readReferenceConfigYAML(t))
	if err != nil {
		t.Fatalf("failed to parse config/config.yaml: %v", err)
	}

	arcDecisions := 0
	for _, recipe := range cfg.Recipes {
		for index := range recipe.Profile.Decisions {
			decision := &recipe.Profile.Decisions[index]
			if decision.Algorithm == nil || decision.Algorithm.Type != RaylineARCAlgorithmType {
				continue
			}
			arcDecisions++

			if replay := cfg.EffectiveRouterReplayConfig(decision); replay != nil && replay.Enabled {
				t.Errorf(
					"decision %q (algorithm.type=%s): effective router_replay is enabled; ARC episode requests must not be persisted",
					decision.Name, RaylineARCAlgorithmType,
				)
			}
			if cfg.IsCacheEnabledForDecisionObject(decision) {
				t.Errorf(
					"decision %q: response_cache resolves enabled, which would store the query text and both bodies",
					decision.Name,
				)
			}
			if memory := decision.GetMemoryConfig(); memory != nil && memory.Enabled {
				t.Errorf(
					"decision %q: memory plugin resolves enabled, which would store each turn verbatim",
					decision.Name,
				)
			}
		}
	}

	// Guard against the assertions above passing because the loop found
	// nothing: the reference config is required to carry ARC decisions.
	if arcDecisions == 0 {
		t.Fatalf("config/config.yaml must include at least one algorithm.type=%s decision", RaylineARCAlgorithmType)
	}
}
