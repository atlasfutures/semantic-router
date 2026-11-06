/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// RaylineARCToolFreeOverride is one tier's opt-in Stage-2 tool-free override
// (pathfinder arc_policy_contract.ToolFreeTierOverride, pathfinder#6164): a
// request that carries no tool definitions decides at the tier's preset cost
// weight divided by VMultiplier (V x v_multiplier), with ExcludedModels also
// out of the offer and LatencyWeight in place of the stage's own.
type RaylineARCToolFreeOverride struct {
	// PresetCostWeight is the tier's own stage-one cost weight
	// (stage_one.presets); CostWeight is the one a tool-free request
	// decides at.
	PresetCostWeight float64
	CostWeight       float64
	VMultiplier      float64
	ExcludedModels   []string
	LatencyWeight    float64
}

// PackageToolFreeOverride is the package's tool-free override for the tier
// named tier (a decisions_v2 name), when the package declares one.
func (cfg *RaylineARCPolicyServiceConfig) PackageToolFreeOverride(tier string) (RaylineARCToolFreeOverride, bool) {
	if cfg == nil || cfg.PackageManifest == "" {
		return RaylineARCToolFreeOverride{}, false
	}
	pkg, err := cfg.loadPackageV5()
	if err != nil {
		return RaylineARCToolFreeOverride{}, false
	}
	override, ok := pkg.toolFree[tier]
	if !ok {
		return RaylineARCToolFreeOverride{}, false
	}
	override.ExcludedModels = append([]string(nil), override.ExcludedModels...)
	return override, true
}

// parseStageOneToolFreeOverrides reads stage_one.tool_free_overrides. Absent
// (or no stage_one) is nil: every request decides as before. A stated one is
// checked as pathfinder's contract checks it: each key names a declared
// preset, v_multiplier is positive and finite, latency_weight non-negative
// and finite, excluded models are unique nonblank offered models that leave
// at least one offered. The policy service applies it; VSR reads it to know
// which tiers to ask the service's verdict for.
func parseStageOneToolFreeOverrides(raw json.RawMessage) (map[string]RaylineARCToolFreeOverride, error) {
	if raw == nil {
		return nil, nil
	}
	var declared struct {
		ToolFreeOverrides json.RawMessage `json:"tool_free_overrides"`
	}
	// A stated null is absent, as pathfinder's StageOnePart reads it.
	if json.Unmarshal(raw, &declared) != nil || declared.ToolFreeOverrides == nil ||
		string(declared.ToolFreeOverrides) == "null" {
		return nil, nil // stage_one's own shape is checkStageOne's
	}
	var part struct {
		ToolFreeOverrides json.RawMessage    `json:"tool_free_overrides"`
		Presets           map[string]float64 `json:"presets"`
		Offered           []struct {
			Model string `json:"model"`
		} `json:"offered"`
	}
	if err := json.Unmarshal(raw, &part); err != nil {
		return nil, fmt.Errorf("stage_one with tool_free_overrides: %w", err)
	}
	var stated map[string]json.RawMessage
	if json.Unmarshal(part.ToolFreeOverrides, &stated) != nil {
		return nil, fmt.Errorf("stage_one.tool_free_overrides %s is not an object", part.ToolFreeOverrides)
	}
	if len(stated) == 0 {
		return nil, fmt.Errorf("stage_one.tool_free_overrides names no tier")
	}
	offered := make(map[string]bool, len(part.Offered))
	for _, candidate := range part.Offered {
		offered[candidate.Model] = true
	}
	overrides := make(map[string]RaylineARCToolFreeOverride, len(stated))
	for name, value := range stated {
		preset, ok := part.Presets[name]
		if !ok {
			return nil, fmt.Errorf("stage_one.tool_free_overrides names tier %q, which stage_one.presets does not", name)
		}
		var fields struct {
			VMultiplier    *float64        `json:"v_multiplier"`
			ExcludedModels json.RawMessage `json:"excluded_models"`
			LatencyWeight  json.RawMessage `json:"latency_weight"`
		}
		decoder := json.NewDecoder(bytes.NewReader(value))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&fields); err != nil || string(value) == "null" {
			return nil, fmt.Errorf("stage_one.tool_free_overrides %q is not an override object", name)
		}
		// An omitted member takes its default; a stated null is refused, as
		// pathfinder's ToolFreeTierOverride refuses it.
		var excludedModels []string
		var latencyWeight *float64
		if fields.ExcludedModels != nil &&
			(string(fields.ExcludedModels) == "null" || json.Unmarshal(fields.ExcludedModels, &excludedModels) != nil) {
			return nil, fmt.Errorf("stage_one.tool_free_overrides %q excluded_models must be a list of names", name)
		}
		if fields.LatencyWeight != nil &&
			(string(fields.LatencyWeight) == "null" || json.Unmarshal(fields.LatencyWeight, &latencyWeight) != nil) {
			return nil, fmt.Errorf("stage_one.tool_free_overrides %q latency_weight must be non-negative and finite", name)
		}
		if fields.VMultiplier == nil || !(*fields.VMultiplier > 0) || math.IsInf(*fields.VMultiplier, 0) {
			return nil, fmt.Errorf("stage_one.tool_free_overrides %q v_multiplier must be positive and finite", name)
		}
		override := RaylineARCToolFreeOverride{
			PresetCostWeight: preset, CostWeight: preset / *fields.VMultiplier, VMultiplier: *fields.VMultiplier,
		}
		if latencyWeight != nil {
			if !(*latencyWeight >= 0) || math.IsInf(*latencyWeight, 0) {
				return nil, fmt.Errorf("stage_one.tool_free_overrides %q latency_weight must be non-negative and finite", name)
			}
			override.LatencyWeight = *latencyWeight
		}
		excluded := make(map[string]bool, len(excludedModels))
		for _, model := range excludedModels {
			if strings.TrimSpace(model) == "" || excluded[model] {
				return nil, fmt.Errorf("stage_one.tool_free_overrides %q excluded models must be unique nonblank names", name)
			}
			if !offered[model] {
				return nil, fmt.Errorf("stage_one.tool_free_overrides %q excludes %q, which stage one does not offer", name, model)
			}
			excluded[model] = true
		}
		if len(excluded) >= len(offered) {
			return nil, fmt.Errorf("stage_one.tool_free_overrides %q excludes every offered model", name)
		}
		override.ExcludedModels = append([]string(nil), excludedModels...)
		overrides[name] = override
	}
	return overrides, nil
}
