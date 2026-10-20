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

import "errors"

// PolicyExplorationOrigin is a relaxed episode's own record of its latest
// Stage-1 boundary: the boundary's decision key and the candidate it served
// (pathfinder docs/arc_serving_contract.md, "Exploration", "For VSR"). A
// relaxed call keeps no episode memory, so the service answers a held turn
// with a null origin_decision_key; VSR joins the turn to its boundary from
// this record, and only while it is the latest boundary's.
type PolicyExplorationOrigin struct {
	DecisionKey string                  `json:"decision_key"`
	Chosen      PolicyStageOneCandidate `json:"chosen"`
}

// Bounds of a stored origin: a decision key is the service's digest, and a
// candidate id is a few short members ({model, effort}).
const (
	maxExplorationDecisionKeyBytes = 128
	maxExplorationCandidateMembers = 8
	maxExplorationCandidateBytes   = 128
)

// NextPolicyExplorationOrigin is the origin a main turn leaves for the next
// one, from its decision's exploration record:
//   - a boundary's record (not held, with a decision key and a chosen
//     candidate) replaces it;
//   - a held turn whose candidate is the recorded boundary's keeps it;
//   - anything else clears it: a decision with no record (exploration off or
//     excluded for the turn, or no decide at all) may have been a later
//     boundary, and a held turn serving another candidate shows the record
//     is not the latest boundary's. A cleared origin leaves the key null,
//     never an older boundary's.
func NextPolicyExplorationOrigin(
	previous *PolicyExplorationOrigin,
	record *PolicyExplorationRecord,
) *PolicyExplorationOrigin {
	switch {
	case record == nil:
		return nil
	case record.Held:
		if previous.joins(record) {
			return previous.Clone()
		}
		return nil
	case record.DecisionKey != nil && *record.DecisionKey != "" && len(record.Chosen) > 0:
		origin := &PolicyExplorationOrigin{DecisionKey: *record.DecisionKey, Chosen: cloneStageOneCandidate(record.Chosen)}
		if origin.validate() != nil {
			return nil
		}
		return origin
	default:
		return nil
	}
}

// HeldDecisionKey is the key a held turn's call joins through when the
// service stated no origin: this record's, when the held candidate is the
// one its boundary chose; nil otherwise.
func (origin *PolicyExplorationOrigin) HeldDecisionKey(record *PolicyExplorationRecord) *string {
	if record == nil || !record.Held || record.OriginDecisionKey != nil || !origin.joins(record) {
		return nil
	}
	key := origin.DecisionKey
	return &key
}

func (origin *PolicyExplorationOrigin) joins(record *PolicyExplorationRecord) bool {
	return origin != nil && origin.DecisionKey != "" && len(origin.Chosen) > 0 &&
		sameStageOneCandidate(origin.Chosen, record.Chosen)
}

// Clone is a copy the caller may change.
func (origin *PolicyExplorationOrigin) Clone() *PolicyExplorationOrigin {
	if origin == nil {
		return nil
	}
	return &PolicyExplorationOrigin{DecisionKey: origin.DecisionKey, Chosen: cloneStageOneCandidate(origin.Chosen)}
}

// validate refuses a stored origin NextPolicyExplorationOrigin could not
// have produced.
func (origin *PolicyExplorationOrigin) validate() error {
	if origin == nil {
		return nil
	}
	malformed := errors.New("ARC policy episode exploration origin is malformed")
	if origin.DecisionKey == "" || len(origin.DecisionKey) > maxExplorationDecisionKeyBytes ||
		len(origin.Chosen) == 0 || len(origin.Chosen) > maxExplorationCandidateMembers {
		return malformed
	}
	for member, value := range origin.Chosen {
		if member == "" || len(member) > maxExplorationCandidateBytes ||
			value != nil && len(*value) > maxExplorationCandidateBytes {
			return malformed
		}
	}
	return nil
}

func sameStageOneCandidate(a, b PolicyStageOneCandidate) bool {
	if len(a) != len(b) {
		return false
	}
	for member, value := range a {
		other, present := b[member]
		if !present || (value == nil) != (other == nil) || value != nil && *value != *other {
			return false
		}
	}
	return true
}

func cloneStageOneCandidate(candidate PolicyStageOneCandidate) PolicyStageOneCandidate {
	if candidate == nil {
		return nil
	}
	cloned := make(PolicyStageOneCandidate, len(candidate))
	for member, value := range candidate {
		if value != nil {
			copied := *value
			value = &copied
		}
		cloned[member] = value
	}
	return cloned
}
