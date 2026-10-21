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
	"encoding/json"
	"fmt"
)

// Optional decide-response blocks that only an opt-in request member elicits:
// Decisions API v2 (pathfinder#5443: response_fields, selection.tier_points)
// and Stage-1 exploration (pathfinder#5436: selection.exploration). A request
// that omits those members never receives them, so these types only let the
// strict decoder accept them once a caller sends the members. The Decisions
// API v2 request members (PolicyTierPoint here, response_fields and
// operating_point.cost_weight on the request) are sent only when a cell's
// decisions_v2 configures them; selection.exploration only when a cell's
// exploration does and the turn's user is eligible.

// PolicyPricePerMtok is the USD-per-million-token rates the decision priced
// an action's arm at. A leg without a rate is null, never 0.
type PolicyPricePerMtok struct {
	Input      *float64 `json:"input"`
	CacheRead  *float64 `json:"cache_read"`
	CacheWrite *float64 `json:"cache_write"`
	Output     *float64 `json:"output"`
}

// PolicyActionThinking is the thinking control an action dispatches, read
// from the package catalog. Null is "not part of this action's control";
// ControlID is null for a v4 action.
type PolicyActionThinking struct {
	Effort        *string `json:"effort"`
	BudgetTokens  *int64  `json:"budget_tokens"`
	SteeringLevel *string `json:"steering_level"`
	ControlID     *string `json:"control_id"`
}

// PolicyTierPoint is one selection.tier_points entry: a named cost weight
// that must lie within the package's declared bounds, as the request's own
// operating_point.cost_weight must.
type PolicyTierPoint struct {
	Name       string  `json:"name"`
	CostWeight float64 `json:"cost_weight"`
}

// PolicyTierPick is the decision at one requested tier point, from the same
// forward pass as the decision.
type PolicyTierPick struct {
	SelectedActionID string               `json:"selected_action_id"`
	SelectedArmID    string               `json:"selected_arm_id"`
	Model            string               `json:"model"`
	Thinking         PolicyActionThinking `json:"thinking"`
	// Exploration is this point's own draw (u = draw(salt, decision_key,
	// point name), pathfinder#5664), when the service reports one; absent
	// from a service that explores only the decision itself, and then the
	// point's temperature is unsupported, never assumed applied.
	Exploration *PolicyExplorationRecord `json:"exploration,omitempty"`
}

// PolicyDecisionStageOne is decision.stage_one.
type PolicyDecisionStageOne struct {
	Exploration *PolicyExplorationRecord `json:"exploration"`
}

// PolicyExplorationBasis records how a tau was calibrated.
type PolicyExplorationBasis struct {
	R             float64 `json:"r"`
	PackageSHA256 string  `json:"package_sha256"`
	ReadoutPin    string  `json:"readout_pin"`
}

// PolicyExplorationRequest is selection.exploration in one of its two
// shapes. Legacy (pathfinder#5436): a raw-score tau, the calibration it came
// from and the draw's salt. Temperature (pathfinder#5637): a true softmax
// temperature over per-decision normalized scores, with the surface asking,
// who set it and whether the turn may be training data; set exactly when
// Temperature is non-nil, and then Tau and Basis are not sent. The service
// refuses a request carrying both, so one value encodes only one.
type PolicyExplorationRequest struct {
	Tau   float64
	Basis PolicyExplorationBasis
	Salt  string
	// Temperature selects the temperature shape.
	Temperature *PolicyExplorationTemperature
}

// PolicyExplorationTemperature is the temperature shape's members beside the
// salt (pathfinder#5637, "What VSR and memex send and accept").
type PolicyExplorationTemperature struct {
	// Temperature is T >= 0; 0 is greedy.
	Temperature float64 `json:"temperature"`
	// Surface is "hosted" (a served turn) or "decisions" (POST /v1/routes).
	Surface string `json:"surface"`
	// Source is "default" (the cell's own T) or "caller" (the end caller's).
	Source string `json:"source"`
	// TrainingData is "allowed" or "not_allowed".
	TrainingData string `json:"training_data"`
	// Calibration is the default T's provenance, logged and never checked:
	// {r, package_sha256, readout_pin}. Absent for a caller's T.
	Calibration *PolicyExplorationBasis `json:"calibration,omitempty"`
}

// Exploration surfaces, sources and training-data values (pathfinder#5637).
const (
	PolicyExplorationSurfaceHosted    = "hosted"
	PolicyExplorationSurfaceDecisions = "decisions"
	PolicyExplorationSourceDefault    = "default"
	PolicyExplorationSourceCaller     = "caller"
	PolicyTrainingDataAllowed         = "allowed"
	PolicyTrainingDataNotAllowed      = "not_allowed"
)

type policyExplorationLegacyWire struct {
	Tau   float64                `json:"tau"`
	Basis PolicyExplorationBasis `json:"basis"`
	Salt  string                 `json:"salt"`
}

type policyExplorationTemperatureWire struct {
	Temperature  float64                 `json:"temperature"`
	Salt         string                  `json:"salt"`
	Surface      string                  `json:"surface"`
	Source       string                  `json:"source"`
	TrainingData string                  `json:"training_data"`
	Calibration  *PolicyExplorationBasis `json:"calibration,omitempty"`
}

// MarshalJSON writes the one shape the request is in.
func (request PolicyExplorationRequest) MarshalJSON() ([]byte, error) {
	if member := request.Temperature; member != nil {
		return json.Marshal(policyExplorationTemperatureWire{
			Temperature: member.Temperature, Salt: request.Salt, Surface: member.Surface,
			Source: member.Source, TrainingData: member.TrainingData, Calibration: member.Calibration,
		})
	}
	return json.Marshal(policyExplorationLegacyWire{Tau: request.Tau, Basis: request.Basis, Salt: request.Salt})
}

// UnmarshalJSON reads either shape, strictly, and refuses one carrying both.
func (request *PolicyExplorationRequest) UnmarshalJSON(body []byte) error {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(body, &probe); err != nil {
		return err
	}
	_, hasTemperature := probe["temperature"]
	_, hasTau := probe["tau"]
	switch {
	case hasTemperature && hasTau:
		return fmt.Errorf("exploration carries both tau and temperature")
	case hasTemperature:
		var wire policyExplorationTemperatureWire
		if err := decodeStrict(body, &wire); err != nil {
			return err
		}
		*request = PolicyExplorationRequest{Salt: wire.Salt, Temperature: &PolicyExplorationTemperature{
			Temperature: wire.Temperature, Surface: wire.Surface, Source: wire.Source,
			TrainingData: wire.TrainingData, Calibration: wire.Calibration,
		}}
		return nil
	default:
		var wire policyExplorationLegacyWire
		if err := decodeStrict(body, &wire); err != nil {
			return err
		}
		*request = PolicyExplorationRequest{Tau: wire.Tau, Basis: wire.Basis, Salt: wire.Salt}
		return nil
	}
}

// PolicyStageOneCandidate is a Stage-1 candidate id ({model, effort}); the
// contract types it as an open string map whose values may be null.
type PolicyStageOneCandidate map[string]*string

// PolicyExplorationRecord is what a decision under a requested tau served
// (pathfinder docs/arc_serving_contract.md, "Exploration"). Offered, Scores
// and Pi align in catalog order; a null score is an ineligible candidate.
// Tau, Propensity and Explored feed llm_usage, so a null or missing one stays
// nil rather than reading as 0 or false.
type PolicyExplorationRecord struct {
	Applied           bool                      `json:"applied"`
	Reason            *string                   `json:"reason"`
	Tau               *float64                  `json:"tau"`
	RequestedTau      float64                   `json:"requested_tau"`
	Basis             PolicyExplorationBasis    `json:"basis"`
	Salt              string                    `json:"salt"`
	U                 *float64                  `json:"u"`
	Offered           []PolicyStageOneCandidate `json:"offered"`
	Scores            []*float64                `json:"scores"`
	Pi                []float64                 `json:"pi"`
	Greedy            PolicyStageOneCandidate   `json:"greedy"`
	Chosen            PolicyStageOneCandidate   `json:"chosen"`
	Propensity        *float64                  `json:"propensity"`
	Explored          *bool                     `json:"explored"`
	PackageSHA256     string                    `json:"package_sha256"`
	DecisionKey       *string                   `json:"decision_key"`
	Held              bool                      `json:"held"`
	OriginDecisionKey *string                   `json:"origin_decision_key"`
	// Override is "proxy_remap" when the drawn model was served through its
	// proxy arm, else null; a label only.
	Override *string `json:"override"`

	// The temperature shape's record members (pathfinder#5637, "Contract
	// records"; pathfinder#5666 ExplorationRecordV2), absent from a legacy
	// record. A record of that shape names
	// no tau, requested_tau or basis. Normalization, PiSoftmax, Floor and
	// Calibration are kept as sent, for the log; VSR reads only Temperature,
	// RequestedTemperature and the floor's applied_mass.
	Surface              *string         `json:"surface,omitempty"`
	Source               *string         `json:"source,omitempty"`
	TrainingData         *string         `json:"training_data,omitempty"`
	RequestedTemperature *float64        `json:"requested_temperature,omitempty"`
	Temperature          *float64        `json:"temperature,omitempty"`
	Normalization        json.RawMessage `json:"normalization,omitempty"`
	PiSoftmax            json.RawMessage `json:"pi_softmax,omitempty"`
	Floor                json.RawMessage `json:"floor,omitempty"`
	Calibration          json.RawMessage `json:"calibration,omitempty"`
	// Clamped: a hosted caller's requested T was above the deployment's
	// cap. Result names the Decisions result (tier point) the draw was keyed
	// by, null otherwise (pathfinder#5666, ExplorationRecordV2).
	Clamped *bool   `json:"clamped,omitempty"`
	Result  *string `json:"result,omitempty"`
}

// FloorApplied reports whether floor mass was in the record's distribution
// (the floor's applied_mass above 0): nil when the record states no floor or
// a floor without a readable applied_mass, never a guessed false.
func (record *PolicyExplorationRecord) FloorApplied() *bool {
	if record == nil || len(record.Floor) == 0 || string(record.Floor) == "null" {
		return nil
	}
	var floor struct {
		AppliedMass *float64 `json:"applied_mass"`
	}
	if json.Unmarshal(record.Floor, &floor) != nil || floor.AppliedMass == nil {
		return nil
	}
	applied := *floor.AppliedMass > 0
	return &applied
}
