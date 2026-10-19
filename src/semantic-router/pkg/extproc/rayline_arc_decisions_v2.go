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

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// policyDecisionsV2 is a cell's ARC Decisions API v2 request members
// (pathfinder#5443), from its decisions_v2 config. A nil one sends none, so
// the request is v1's.
type policyDecisionsV2 struct {
	costWeight     *float64
	tierPoints     []raylinearc.PolicyTierPoint
	responseFields []string
}

// arcFailurePolicyOperatingPointMismatch is a decision the service took at
// another cost weight than the cell sent: it is not the cell's tier.
const arcFailurePolicyOperatingPointMismatch = "policy_operating_point_mismatch"

func newPolicyDecisionsV2(cfg *config.RaylineARCDecisionsV2Config) *policyDecisionsV2 {
	if cfg == nil {
		return nil
	}
	members := &policyDecisionsV2{
		responseFields: append([]string(nil), cfg.ResponseFields...),
	}
	if cfg.CostWeight != nil {
		weight := *cfg.CostWeight
		members.costWeight = &weight
	}
	for _, point := range cfg.TierPoints {
		members.tierPoints = append(members.tierPoints, raylinearc.PolicyTierPoint{
			Name: point.Name, CostWeight: *point.CostWeight,
		})
	}
	return members
}

// apply writes the configured members into a decide request. An unset member
// stays absent: cost_weight null, tier_points and response_fields omitted.
func (members *policyDecisionsV2) apply(request *raylinearc.PolicyDecisionRequest) {
	if members == nil {
		return
	}
	if members.costWeight != nil {
		weight := *members.costWeight
		request.Selection.OperatingPoint.CostWeight = &weight
	}
	if len(members.tierPoints) > 0 {
		request.Selection.TierPoints = append([]raylinearc.PolicyTierPoint(nil), members.tierPoints...)
	}
	if len(members.responseFields) > 0 {
		request.ResponseFields = append([]string(nil), members.responseFields...)
	}
}

// operatingPointMismatch reports a decision taken at another cost weight than
// the cell sent, or echoed with none. Only a cell that sends one checks it.
func (members *policyDecisionsV2) operatingPointMismatch(response *raylinearc.PolicyDecisionResponse) bool {
	if members == nil || members.costWeight == nil {
		return false
	}
	echoed := response.Decision.OperatingPoint.CostWeight
	return echoed == nil || *echoed != *members.costWeight
}

// record keeps the resolved operating point and the tier picks on the turn's
// trace, for its selection log. The picks are recorded only: the turn serves
// the decision's own action. Only the configured tier names are kept, so no
// service-chosen key becomes a log key; a configured tier with no pick is
// named as missing rather than dropped.
func (members *policyDecisionsV2) record(trace *selection.RaylineARCTrace, response *raylinearc.PolicyDecisionResponse) {
	if members == nil {
		return
	}
	point := response.Decision.OperatingPoint
	trace.PolicyOperatingPoint = &point
	for _, tier := range members.tierPoints {
		pick, ok := response.Decision.TierPicks[tier.Name]
		if !ok {
			trace.PolicyTierPicksMissing = append(trace.PolicyTierPicksMissing, tier.Name)
			continue
		}
		if trace.PolicyTierPicks == nil {
			trace.PolicyTierPicks = make(map[string]raylinearc.PolicyTierPick, len(members.tierPoints))
		}
		trace.PolicyTierPicks[tier.Name] = pick
	}
}

// addPolicyDecisionsV2Fields writes a decisions_v2 cell's operating point and
// tier picks into its selection log; a cell without the block logs neither.
func addPolicyDecisionsV2Fields(fields map[string]interface{}, trace *selection.RaylineARCTrace) {
	if trace.PolicyOperatingPoint == nil {
		return
	}
	point := map[string]interface{}{
		"name":        trace.PolicyOperatingPoint.Name,
		"point_id":    trace.PolicyOperatingPoint.PointID,
		"cost_weight": nil,
	}
	if weight := trace.PolicyOperatingPoint.CostWeight; weight != nil {
		point["cost_weight"] = *weight
	}
	fields["policy_operating_point"] = point
	if len(trace.PolicyTierPicksMissing) > 0 {
		fields["policy_tier_picks_missing"] = trace.PolicyTierPicksMissing
	}
	if len(trace.PolicyTierPicks) == 0 {
		return
	}
	picks := make(map[string]interface{}, len(trace.PolicyTierPicks))
	for name, pick := range trace.PolicyTierPicks {
		picks[name] = map[string]interface{}{
			"selected_action_id": pick.SelectedActionID,
			"selected_arm_id":    pick.SelectedArmID,
			"model":              pick.Model,
		}
	}
	fields["policy_tier_picks"] = picks
}
