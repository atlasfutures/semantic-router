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
	"encoding/json"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A package's stage_one.tool_free_overrides (pathfinder#6164) is applied by
// the policy service, which reads the request's tools and its
// operating_point name (the cell's decisions_v2 tier) from the decide
// request VSR already sends. VSR only asks for the service's verdict and
// records it; it never infers that the override applied.

// toolFreeOverrideApplied is llm_usage tool_free_override for a decision
// the service says took the override; otherwise the field is the service's
// reason, or null when it stated no verdict.
const toolFreeOverrideApplied = "applied"

// policyCellHasToolFreeOverride reports whether the cell's package declares
// a tool-free override for the cell's decisions_v2 tier.
func policyCellHasToolFreeOverride(policy *config.RaylineARCPolicyServiceConfig) bool {
	if policy == nil || policy.DecisionsV2 == nil || policy.DecisionsV2.Name == "" {
		return false
	}
	_, ok := policy.PackageToolFreeOverride(policy.DecisionsV2.Name)
	return ok
}

// askToolFreeOverride names response_fields tool_free_override on a cell
// whose tier has an override, of a service that lists the capability (an
// older one refuses the field).
func (scorer *policyServiceScorer) askToolFreeOverride(request *raylinearc.PolicyDecisionRequest) {
	if !scorer.toolFreeTier || !scorer.advertises(raylinearc.PolicyCapabilityToolFreeOverride) ||
		slices.Contains(request.ResponseFields, raylinearc.PolicyCapabilityToolFreeOverride) {
		return
	}
	request.ResponseFields = append(append([]string(nil), request.ResponseFields...), raylinearc.PolicyCapabilityToolFreeOverride)
}

// recordToolFreeOverride keeps, for a request that asked, the service's
// verdict and VSR's own reading of whether the request carried tools (a
// log-only cross-check, never the verdict).
func recordToolFreeOverride(
	trace *selection.RaylineARCTrace,
	request *raylinearc.PolicyDecisionRequest,
	response *raylinearc.PolicyDecisionResponse,
	client raylinearc.PolicyClientRequest,
) {
	if !slices.Contains(request.ResponseFields, raylinearc.PolicyCapabilityToolFreeOverride) {
		return
	}
	toolFree := policyRequestToolFree(client)
	trace.RequestToolFree = &toolFree
	stageOne := response.Decision.StageOne
	if stageOne == nil || stageOne.ToolFreeOverride == nil {
		return
	}
	verdict := stageOne.ToolFreeOverride
	switch {
	case verdict.Applied:
		trace.ToolFreeOverride = toolFreeOverrideApplied
	case verdict.Reason != nil && *verdict.Reason != "" && *verdict.Reason != toolFreeOverrideApplied:
		trace.ToolFreeOverride = *verdict.Reason
	}
}

// policyRequestToolFree reports whether a request carries no tool
// definitions: its top-level tools absent, null or an empty array, as
// pathfinder#6164 defines the class. It reads the tools the decide request
// forwards, which every ingress codec fills from the client's own body
// (policyClientRequestOf: tools for Anthropic Messages and OpenAI chat, the
// Responses body's tools, [] when it sent none). Tools that are not an array
// are counted as tools, never as none.
func policyRequestToolFree(request raylinearc.PolicyClientRequest) bool {
	if len(request.Tools) == 0 {
		return true
	}
	var tools []json.RawMessage
	return json.Unmarshal(request.Tools, &tools) == nil && len(tools) == 0
}
