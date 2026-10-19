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

import "testing"

// The exploration guardrails (pathfinder#5436) are selection refusals with
// their own bounded reasons, not "other".
func TestPolicyRefusalReasonNamesTheExplorationGuardrails(t *testing.T) {
	for _, reason := range []string{
		"exploration_tau_exceeds_ceiling", "exploration_basis_mismatch", "exploration_unsupported_by_package",
	} {
		failure := PolicyErrorResponse{Error: "selection_refused", Detail: map[string]any{"reason": reason}}
		if class := policyFailureClass(failure); class != PolicySelectionRefusedClass {
			t.Errorf("%s: class %q", reason, class)
		}
		if got := PolicyRefusalReason(failure.Detail); got != reason {
			t.Errorf("%s: reason %q", reason, got)
		}
	}
}
