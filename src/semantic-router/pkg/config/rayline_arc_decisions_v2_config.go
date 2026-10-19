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
	"fmt"
	"math"
	"regexp"
)

// RaylineARCDecisionsV2Config is a cell's ARC Decisions API v2 request
// members (pathfinder#5443). Every member is optional and an absent one is
// not sent, so a cell without the block sends the v1 request byte for byte:
// the service decodes strictly, and one that predates v2 refuses the members.
type RaylineARCDecisionsV2Config struct {
	// Name is sent as selection.operating_point.name in place of "default".
	// A tier cell names its alias (swift, sage). It is a record label only:
	// the service does not score by it.
	Name string `yaml:"name,omitempty"`
	// CostWeight is the cell's V, sent as selection.operating_point.cost_weight.
	// The service honours it within the bounds the package declares for the
	// stage that chooses the model and refuses it otherwise. A tier cell
	// (Swift, Keen, Sage) states its V here, so one package serves every tier.
	CostWeight *float64 `yaml:"cost_weight,omitempty"`
	// TierPoints are named V presets the service also decides at, from the
	// same forward pass (selection.tier_points). The picks are recorded,
	// never served.
	TierPoints []RaylineARCTierPoint `yaml:"tier_points,omitempty"`
	// ResponseFields names the per-action response blocks to return:
	// price_per_mtok and thinking.
	ResponseFields []string `yaml:"response_fields,omitempty"`
}

// RaylineARCTierPoint is one named V preset.
type RaylineARCTierPoint struct {
	Name       string   `yaml:"name"`
	CostWeight *float64 `yaml:"cost_weight"`
}

const (
	RaylineARCResponseFieldPricePerMtok = "price_per_mtok"
	RaylineARCResponseFieldThinking     = "thinking"

	// maxRaylineARCTierPoints is the service's MAX_TIER_POINTS.
	maxRaylineARCTierPoints = 8
	// maxRaylineARCTierPointNameLength bounds a tier name, which becomes a
	// log key.
	maxRaylineARCTierPointNameLength = 64
)

// raylineARCOperatingPointName is a short lowercase identifier.
var raylineARCOperatingPointName = regexp.MustCompile(`^[a-z][a-z0-9_-]{0,31}$`)

func validateRaylineARCDecisionsV2Config(cfg *RaylineARCDecisionsV2Config) error {
	if cfg == nil {
		return nil
	}
	if cfg.Name != "" && !raylineARCOperatingPointName.MatchString(cfg.Name) {
		return fmt.Errorf("decisions_v2 name must match %s", raylineARCOperatingPointName)
	}
	if cfg.CostWeight != nil && !validRaylineARCCostWeight(*cfg.CostWeight) {
		return fmt.Errorf("decisions_v2 cost_weight must be a finite number >= 0")
	}
	if len(cfg.TierPoints) > maxRaylineARCTierPoints {
		return fmt.Errorf("decisions_v2 tier_points takes at most %d points", maxRaylineARCTierPoints)
	}
	names := make(map[string]bool, len(cfg.TierPoints))
	for _, point := range cfg.TierPoints {
		if point.Name == "" || len(point.Name) > maxRaylineARCTierPointNameLength {
			return fmt.Errorf("decisions_v2 tier_points name must be 1 to %d bytes", maxRaylineARCTierPointNameLength)
		}
		if names[point.Name] {
			return fmt.Errorf("decisions_v2 tier_points name %q repeats", point.Name)
		}
		names[point.Name] = true
		if point.CostWeight == nil || !validRaylineARCCostWeight(*point.CostWeight) {
			return fmt.Errorf("decisions_v2 tier_points %q cost_weight must be a finite number >= 0", point.Name)
		}
	}
	fields := make(map[string]bool, len(cfg.ResponseFields))
	for _, field := range cfg.ResponseFields {
		switch field {
		case RaylineARCResponseFieldPricePerMtok, RaylineARCResponseFieldThinking:
		default:
			return fmt.Errorf("decisions_v2 response_fields must be %s or %s",
				RaylineARCResponseFieldPricePerMtok, RaylineARCResponseFieldThinking)
		}
		if fields[field] {
			return fmt.Errorf("decisions_v2 response_fields %q repeats", field)
		}
		fields[field] = true
	}
	return nil
}

func validRaylineARCCostWeight(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0
}
