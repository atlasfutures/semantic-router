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

package raylinearc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
)

// The policy service advertises which opt-in request members it serves on
// the packages listing when asked with detail=capabilities (pathfinder#5454):
// a top-level capabilities list, and per package the cost-weight bounds a
// request's operating point may name. A service that predates the detail
// ignores the parameter and states neither, so an absent field is an
// unsupported capability, never a supported one.
const (
	PolicyCapabilityDecisionsV2    = "decisions_v2"
	PolicyCapabilityTierPoints     = "tier_points"
	PolicyCapabilityResponseFields = "response_fields"
	PolicyCapabilityExploration    = "exploration"
	// PolicyCapabilityExplorationTemperature: selection.exploration in the
	// temperature shape (pathfinder#5637). Proposed name: a service that
	// does not advertise it is never sent that shape.
	PolicyCapabilityExplorationTemperature = "exploration_temperature"
	// PolicyCapabilityCallerPrices: routing_prices.basis caller, one caller
	// price world for the decision and every tier pick (pathfinder#5668).
	PolicyCapabilityCallerPrices = "caller_prices"
)

// Readiness classes for a service that cannot serve what the cell sends.
const (
	// PolicyCapabilityUnsupportedClass: the listing does not advertise a
	// capability the cell's request members need.
	PolicyCapabilityUnsupportedClass = "capability_unsupported"
	// PolicyCostWeightNotAdjustableClass: the cell sends a cost weight and
	// the served package states no adjustable bounds.
	PolicyCostWeightNotAdjustableClass = "cost_weight_not_adjustable"
	// PolicyCostWeightOutOfBoundsClass: a configured cost weight lies
	// outside the served package's adjustable bounds.
	PolicyCostWeightOutOfBoundsClass = "cost_weight_out_of_bounds"
)

// PolicyCostWeightBounds is a package's adjustable_cost_weight: the closed
// interval of cost weights its model choice honours (for a two-stage package,
// stage one's bounds intersected with stage two's).
type PolicyCostWeightBounds struct {
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

// Contains reports whether value lies within the bounds, inclusive.
func (bounds PolicyCostWeightBounds) Contains(value float64) bool {
	return value >= bounds.Min && value <= bounds.Max
}

// PolicyAdjustableCostWeight is a listing entry's adjustable_cost_weight:
// absent (a listing without the capabilities detail), null (the package
// declares no bounds) or the bounds. Stated tells absent from null, so a
// listing re-encodes as it was received.
type PolicyAdjustableCostWeight struct {
	Stated bool
	Bounds *PolicyCostWeightBounds
}

// IsZero is the absent member, which omitzero leaves out.
func (member PolicyAdjustableCostWeight) IsZero() bool {
	return !member.Stated
}

func (member PolicyAdjustableCostWeight) MarshalJSON() ([]byte, error) {
	if member.Bounds == nil {
		return []byte("null"), nil
	}
	return json.Marshal(member.Bounds)
}

// UnmarshalJSON refuses unknown members, a missing or non-finite bound, and
// min above max.
func (member *PolicyAdjustableCostWeight) UnmarshalJSON(body []byte) error {
	member.Stated = true
	member.Bounds = nil
	if bytes.Equal(bytes.TrimSpace(body), []byte("null")) {
		return nil
	}
	var raw struct {
		Min *float64 `json:"min"`
		Max *float64 `json:"max"`
	}
	if err := decodeStrict(body, &raw); err != nil {
		return err
	}
	if raw.Min == nil || raw.Max == nil {
		return fmt.Errorf("adjustable_cost_weight states min and max")
	}
	if !finite(*raw.Min) || !finite(*raw.Max) || *raw.Min > *raw.Max {
		return fmt.Errorf("adjustable_cost_weight bounds are not an interval")
	}
	member.Bounds = &PolicyCostWeightBounds{Min: *raw.Min, Max: *raw.Max}
	return nil
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// PolicyServiceRequirements is what a cell's request members need of the
// service: the capabilities it must advertise and the cost weights the
// served package's bounds must hold. The zero value needs nothing, so a cell
// without opt-in members is served by a service of any age.
type PolicyServiceRequirements struct {
	Capabilities []string
	CostWeights  []float64
}

// CheckPolicyServiceRequirements fails unless the listing advertises every
// required capability and the loaded package's bounds hold every configured
// cost weight. The error is a PolicyServiceError whose Detail names the
// missing capability or the cost weight, so readiness can say why.
func CheckPolicyServiceRequirements(
	listing *PolicyPackagesResponse,
	loaded *PolicyLoadedPackage,
	required PolicyServiceRequirements,
) error {
	for _, capability := range required.Capabilities {
		if listing == nil || !slices.Contains(listing.Capabilities, capability) {
			return &PolicyServiceError{
				Class:  PolicyCapabilityUnsupportedClass,
				Detail: map[string]any{"capability": capability},
			}
		}
	}
	if len(required.CostWeights) == 0 {
		return nil
	}
	if loaded == nil || loaded.AdjustableCostWeight.Bounds == nil {
		return &PolicyServiceError{Class: PolicyCostWeightNotAdjustableClass, Detail: map[string]any{}}
	}
	bounds := *loaded.AdjustableCostWeight.Bounds
	for _, weight := range required.CostWeights {
		if !bounds.Contains(weight) {
			return &PolicyServiceError{
				Class: PolicyCostWeightOutOfBoundsClass,
				Detail: map[string]any{
					"cost_weight": weight,
					"min":         bounds.Min,
					"max":         bounds.Max,
				},
			}
		}
	}
	return nil
}
