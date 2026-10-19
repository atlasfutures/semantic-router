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

// pathfinder#5443: the decide body VSR encodes for the Decisions API v2
// fixture's request is the fixture, members included; with the members unset
// the body has no tier_points or response_fields and a null cost_weight.
func TestEncodePolicyDecisionRequestDecisionsV2Members(t *testing.T) {
	body := readPolicyFixture(t, "decision_request_decisions_v2.v1.json")
	var request PolicyDecisionRequest
	if err := decodeStrict(body, &request); err != nil {
		t.Fatal(err)
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

	request.Selection.TierPoints, request.ResponseFields = nil, nil
	request.Selection.OperatingPoint.CostWeight = nil
	encoded, err = EncodePolicyDecisionRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	got = nil
	if err = json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if _, present := got["response_fields"]; present {
		t.Fatalf("response_fields sent unset: %s", encoded)
	}
	selection := got["selection"].(map[string]any)
	if _, present := selection["tier_points"]; present {
		t.Fatalf("tier_points sent unset: %s", encoded)
	}
	point := selection["operating_point"].(map[string]any)
	if value, present := point["cost_weight"]; !present || value != nil {
		t.Fatalf("cost_weight %v, want null as v1 sends it", value)
	}
}

// The Decisions API v2 refusals are the cell's configuration against its
// package: their own class, with only the reason kept, and only under the
// error code that carries it.
func TestPolicyServiceClassifiesDecisionsV2Refusals(t *testing.T) {
	for _, tc := range []struct {
		code, reason, want string
	}{
		{"selection_refused", "operating_point_out_of_bounds", PolicyDecisionsV2RefusedClass},
		{"selection_refused", "operating_point_not_adjustable", PolicyDecisionsV2RefusedClass},
		{"unsupported_request", "tier_points_unsupported", PolicyDecisionsV2RefusedClass},
		// A reason under another code keeps that code's class.
		{"unsupported_request", "operating_point_out_of_bounds", "unsupported_request"},
		{"selection_refused", "tier_points_unsupported", PolicySelectionRefusedClass},
		{"selection_refused", "stage_one_held_unknown", PolicyStageOneHeldUnknownClass},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(http.StatusUnprocessableEntity)
			_ = json.NewEncoder(writer).Encode(map[string]any{
				"error":  tc.code,
				"detail": map[string]any{"reason": tc.reason, "note": "dropped"},
			})
		}))
		client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: time.Second})
		_, err := client.Decide(context.Background(), PolicyDecisionRequest{})
		server.Close()
		var failure *PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != tc.want {
			t.Fatalf("%s/%s: err = %v, want class %s", tc.code, tc.reason, err, tc.want)
		}
		if tc.want == PolicyDecisionsV2RefusedClass &&
			(len(failure.Detail) != 1 || failure.Detail["reason"] != tc.reason) {
			t.Fatalf("%s: detail = %v, want only the reason", tc.reason, failure.Detail)
		}
	}
}
