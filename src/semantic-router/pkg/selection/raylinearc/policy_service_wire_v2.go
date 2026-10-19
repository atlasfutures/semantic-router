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

// PolicyExplorationRequest is selection.exploration: a Stage-1 temperature,
// the calibration it came from and the draw's salt.
type PolicyExplorationRequest struct {
	Tau   float64                `json:"tau"`
	Basis PolicyExplorationBasis `json:"basis"`
	Salt  string                 `json:"salt"`
}

// PolicyStageOneCandidate is a Stage-1 candidate id ({model, effort}); the
// contract types it as an open string map whose values may be null.
type PolicyStageOneCandidate map[string]*string

// PolicyExplorationRecord is what a decision under a requested tau served
// (pathfinder docs/arc_serving_contract.md, "Exploration"). Offered, Scores
// and Pi align in catalog order; a null score is an ineligible candidate.
type PolicyExplorationRecord struct {
	Applied           bool                      `json:"applied"`
	Reason            *string                   `json:"reason"`
	Tau               float64                   `json:"tau"`
	RequestedTau      float64                   `json:"requested_tau"`
	Basis             PolicyExplorationBasis    `json:"basis"`
	Salt              string                    `json:"salt"`
	U                 *float64                  `json:"u"`
	Offered           []PolicyStageOneCandidate `json:"offered"`
	Scores            []*float64                `json:"scores"`
	Pi                []float64                 `json:"pi"`
	Greedy            PolicyStageOneCandidate   `json:"greedy"`
	Chosen            PolicyStageOneCandidate   `json:"chosen"`
	Propensity        float64                   `json:"propensity"`
	Explored          bool                      `json:"explored"`
	PackageSHA256     string                    `json:"package_sha256"`
	DecisionKey       *string                   `json:"decision_key"`
	Held              bool                      `json:"held"`
	OriginDecisionKey *string                   `json:"origin_decision_key"`
	// Override is "proxy_remap" when the drawn model was served through its
	// proxy arm, else null; a label only.
	Override *string `json:"override"`
}
