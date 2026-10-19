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
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"
)

// The exploration guardrails (pathfinder#5436) are selection refusals with
// their own bounded reasons, not "other", classed as the cell's exploration
// config against its service: exploration_refused.
func TestPolicyRefusalReasonNamesTheExplorationGuardrails(t *testing.T) {
	for _, reason := range []string{
		"exploration_tau_exceeds_ceiling", "exploration_basis_mismatch", "exploration_unsupported_by_package",
	} {
		failure := PolicyErrorResponse{Error: "selection_refused", Detail: map[string]any{"reason": reason}}
		if class := policyFailureClass(failure); class != PolicyExplorationRefusedClass {
			t.Errorf("%s: class %q", reason, class)
		}
		if got := PolicyRefusalReason(failure.Detail); got != reason {
			t.Errorf("%s: reason %q", reason, got)
		}
	}
}

// Only under selection_refused, and only the reason travels on.
func TestPolicyServiceClassifiesExplorationRefusals(t *testing.T) {
	for _, tc := range []struct{ code, reason, want string }{
		{"selection_refused", "exploration_tau_exceeds_ceiling", PolicyExplorationRefusedClass},
		{"selection_refused", "exploration_basis_mismatch", PolicyExplorationRefusedClass},
		{"selection_refused", "exploration_unsupported_by_package", PolicyExplorationRefusedClass},
		{"unsupported_request", "exploration_basis_mismatch", "unsupported_request"},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(http.StatusUnprocessableEntity)
			_ = json.NewEncoder(writer).Encode(map[string]any{
				"error": tc.code, "detail": map[string]any{"reason": tc.reason, "tau": 9},
			})
		}))
		client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: time.Second})
		_, err := client.Decide(context.Background(), PolicyDecisionRequest{})
		server.Close()
		var failure *PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != tc.want {
			t.Fatalf("%s/%s: err = %v, want class %s", tc.code, tc.reason, err, tc.want)
		}
		if tc.want == PolicyExplorationRefusedClass && !reflect.DeepEqual(failure.Detail, map[string]any{"reason": tc.reason}) {
			t.Fatalf("%s: detail %v, want only the reason", tc.reason, failure.Detail)
		}
	}
}

// The decide body VSR encodes for pathfinder's exploration request fixture is
// the fixture; without the member the selection carries no exploration key.
func TestEncodePolicyDecisionRequestExplorationMember(t *testing.T) {
	body := readPolicyFixture(t, "decision_request_exploration.v1.json")
	var request PolicyDecisionRequest
	if err := decodeStrict(body, &request); err != nil {
		t.Fatal(err)
	}
	if request.Selection.Exploration == nil || request.Selection.Exploration.Tau != 0.02 {
		t.Fatalf("decoded exploration %+v", request.Selection.Exploration)
	}
	encoded, err := EncodePolicyDecisionRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	var want, got map[string]any
	if err = json.Unmarshal(body, &want); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("encoded\n%s\nwant the fixture\n%s", encoded, body)
	}
	request.Selection.Exploration = nil
	if encoded, err = EncodePolicyDecisionRequest(request); err != nil {
		t.Fatal(err)
	}
	got = nil
	if err = json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if _, sent := got["selection"].(map[string]any)["exploration"]; sent {
		t.Fatalf("exploration sent unset: %s", encoded)
	}
}
