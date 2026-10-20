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
	"errors"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Eligibility headers for Stage-1 exploration (pathfinder#5436). Exploration
// is limited to users who are not opted out of data collection for training,
// and never runs in privacy mode. The router cannot know either fact itself,
// so the gateway in front of it states both, and a request that does not
// state them is excluded. They are read only with trust_turn_signal_headers,
// and never forwarded upstream.
const (
	// raylineARCPrivacyModeHeader is "false" for a request outside privacy
	// mode; "true", any other value or none excludes it.
	raylineARCPrivacyModeHeader = "x-rayline-privacy-mode"
	// raylineARCTrainingDataHeader is "allowed" when the request's user (and
	// team) is not opted out of training-data collection; any other value or
	// none excludes it.
	raylineARCTrainingDataHeader = "x-rayline-training-data"
)

// Why an exploration-configured cell sent no exploration on a turn, as
// logged (exploration_excluded).
const (
	explorationExcludedEvalCell         = "eval_cell"
	explorationExcludedSideCall         = "side_call"
	explorationExcludedUntrustedHeaders = "signal_headers_untrusted"
	explorationExcludedPrivacyMode      = "privacy_mode"
	explorationExcludedPrivacyUnknown   = "privacy_mode_unknown"
	explorationExcludedTrainingOptOut   = "training_data_opted_out"
	explorationExcludedTrainingUnknown  = "training_data_unknown"
)

// policyExploration is a cell's selection.exploration member, from its
// exploration config. A nil one sends none, so every decision is greedy.
type policyExploration struct {
	request  raylinearc.PolicyExplorationRequest
	evalCell bool
	// trusted is trust_turn_signal_headers: the eligibility headers are the
	// gateway's, not the client's.
	trusted bool
}

func newPolicyExploration(policy *config.RaylineARCPolicyServiceConfig) *policyExploration {
	if policy == nil || policy.Exploration == nil {
		return nil
	}
	cfg := policy.Exploration
	return &policyExploration{
		request: raylinearc.PolicyExplorationRequest{
			Tau: *cfg.Tau,
			Basis: raylinearc.PolicyExplorationBasis{
				R: *cfg.Basis.R, PackageSHA256: cfg.Basis.PackageSHA256, ReadoutPin: cfg.Basis.ReadoutPin,
			},
			Salt: cfg.Salt,
		},
		evalCell: policy.EvalCell,
		trusted:  policy.TrustTurnSignalHeaders,
	}
}

// mayExplore reports whether any turn of the cell can carry the member: an
// eval cell, or one whose signal headers are untrusted, excludes every turn
// in exclusion below, so it never sends one.
func (exploration *policyExploration) mayExplore() bool {
	return exploration != nil && !exploration.evalCell && exploration.trusted
}

// exclusion names why this turn may not explore, or "" when it may. Every
// fact it reads fails closed: an unknown privacy mode or consent excludes.
func (exploration *policyExploration) exclusion(headers map[string]string, sideCall bool) string {
	switch {
	case exploration.evalCell:
		return explorationExcludedEvalCell
	case sideCall:
		// A side call commits nothing to the episode: no boundary to draw at.
		return explorationExcludedSideCall
	case !exploration.trusted:
		return explorationExcludedUntrustedHeaders
	}
	switch privacy, stated := headers[raylineARCPrivacyModeHeader]; {
	case !stated || strings.TrimSpace(privacy) == "":
		return explorationExcludedPrivacyUnknown
	case strings.ToLower(strings.TrimSpace(privacy)) != "false":
		return explorationExcludedPrivacyMode
	}
	switch consent, stated := headers[raylineARCTrainingDataHeader]; {
	case !stated || strings.TrimSpace(consent) == "":
		return explorationExcludedTrainingUnknown
	case strings.ToLower(strings.TrimSpace(consent)) != "allowed":
		return explorationExcludedTrainingOptOut
	}
	return ""
}

// apply writes the member into an eligible turn's decide request and returns
// why it did not, "" when it did or when the cell has no exploration.
func (exploration *policyExploration) apply(
	request *raylinearc.PolicyDecisionRequest,
	headers map[string]string,
	sideCall bool,
) string {
	if exploration == nil {
		return ""
	}
	if reason := exploration.exclusion(headers, sideCall); reason != "" {
		return reason
	}
	member := exploration.request
	request.Selection.Exploration = &member
	return ""
}

// record keeps what the turn's exploration did on its trace: the exclusion,
// or the service's record of the draw. A request that asked for tau > 0 and
// got no record back is marked, so it is never read as a greedy turn.
func (exploration *policyExploration) record(
	trace *selection.RaylineARCTrace,
	request *raylinearc.PolicyDecisionRequest,
	response *raylinearc.PolicyDecisionResponse,
	excluded string,
) {
	if exploration == nil {
		return
	}
	trace.PolicyExplorationExcluded = excluded
	if request.Selection.Exploration == nil {
		return
	}
	trace.PolicyExplorationSent = true
	if stageOne := response.Decision.StageOne; stageOne != nil && stageOne.Exploration != nil {
		record := *stageOne.Exploration
		trace.PolicyExploration = &record
	}
}

// addPolicyExplorationFields writes an exploration cell's outcome into its
// selection log: the full record the service returned, or why the turn sent
// none. A cell without exploration logs neither.
func addPolicyExplorationFields(fields map[string]interface{}, trace *selection.RaylineARCTrace) {
	if trace.PolicyExplorationExcluded != "" {
		fields["exploration_excluded"] = trace.PolicyExplorationExcluded
	}
	if !trace.PolicyExplorationSent {
		return
	}
	fields["exploration_sent"] = true
	if trace.PolicyExploration == nil {
		return
	}
	// The record as the service's JSON names it, so the log line carries the
	// contract's field names.
	encoded, err := json.Marshal(trace.PolicyExploration)
	if err != nil {
		return
	}
	var record map[string]interface{}
	if json.Unmarshal(encoded, &record) == nil {
		fields["exploration"] = record
	}
}

// policyExplorationRefused reports the service's refusal of the cell's
// exploration member.
func policyExplorationRefused(err error) bool {
	var failure *raylinearc.PolicyServiceError
	return errors.As(err, &failure) && failure.Class == raylinearc.PolicyExplorationRefusedClass
}

// explorationUsage is the llm_usage members of a call's exploration record
// (pathfinder#5436 "For VSR"): all nil when the decision carried none.
type explorationUsage struct {
	propensity  *float64
	tau         *float64
	explored    *bool
	decisionKey *string
}

func policyExplorationUsage(trace *selection.RaylineARCTrace) explorationUsage {
	if trace == nil || trace.PolicyExploration == nil {
		return explorationUsage{}
	}
	record := trace.PolicyExploration
	copyFloat := func(value *float64) *float64 {
		if value == nil {
			return nil
		}
		copied := *value
		return &copied
	}
	var explored *bool
	if record.Explored != nil {
		value := *record.Explored
		explored = &value
	}
	// A held turn joins its boundary through the key that chose its model:
	// the service's origin, or on a relaxed call, which states none, VSR's
	// own record of the latest boundary.
	key := record.DecisionKey
	if record.Held {
		key = record.OriginDecisionKey
		if key == nil {
			key = trace.PolicyExplorationHeldKey
		}
	}
	var decisionKey *string
	if key != nil {
		copied := *key
		decisionKey = &copied
	}
	return explorationUsage{
		propensity: copyFloat(record.Propensity), tau: copyFloat(record.Tau), explored: explored, decisionKey: decisionKey,
	}
}

// recordPolicyExplorationOrigin keeps a relaxed episode's own record of its
// latest Stage-1 boundary in the state this main turn commits, and on a held
// turn names that boundary's key for llm_usage (pathfinder#5477, "For VSR").
// A strict call's origin is the service's, so a strict turn keeps none.
func recordPolicyExplorationOrigin(
	trace *selection.RaylineARCTrace,
	turn *raylinearc.PolicyEpisodeState,
	relaxed bool,
) {
	if !relaxed || trace.PolicyNextState == nil {
		return
	}
	var previous *raylinearc.PolicyExplorationOrigin
	if turn != nil {
		previous = turn.Exploration
	}
	trace.PolicyExplorationHeldKey = previous.HeldDecisionKey(trace.PolicyExploration)
	trace.PolicyNextState.Exploration = raylinearc.NextPolicyExplorationOrigin(previous, trace.PolicyExploration)
}

// stagePolicyExplorationOrigin stores the origin this turn's decision left
// with its staged boundary. A decided boundary is the episode's latest even
// when its dispatch then fails, so the older origin it replaced or cleared
// must not survive the failure for a later held turn to join.
func stagePolicyExplorationOrigin(staged *raylinearc.EpisodeState, next *raylinearc.PolicyEpisodeState) {
	if staged == nil || staged.Policy == nil || next == nil {
		return
	}
	staged.Policy.Exploration = next.Exploration.Clone()
}

// keepStoredExplorationOrigin keeps the stored origin in a refusal's
// exclusion state: a refusal changes the boundary and the exclusions only,
// and the context it is recorded in predates the decision's origin.
func keepStoredExplorationOrigin(policy, stored *raylinearc.PolicyEpisodeState) {
	if policy == nil {
		return
	}
	policy.Exploration = nil
	if stored != nil {
		policy.Exploration = stored.Exploration.Clone()
	}
}
