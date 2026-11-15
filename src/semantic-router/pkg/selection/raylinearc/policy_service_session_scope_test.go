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
	"strings"
	"testing"
	"time"
)

// An invalid_request refusing session_scope keeps detail.field, and nothing
// else; any other invalid_request keeps no detail.
func TestPolicyServiceKeepsARefusedSessionScopeField(t *testing.T) {
	for _, tc := range []struct {
		field any
		kept  bool
	}{
		{PolicySessionScopeField, true},
		{"episode_mode", false},
		{nil, false},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(writer).Encode(map[string]any{"error": "invalid_request",
				"detail": map[string]any{"field": tc.field, "note": "dropped"}})
		}))
		client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: time.Second})
		_, err := client.Decide(context.Background(), PolicyDecisionRequest{})
		server.Close()
		var failure *PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != "invalid_request" {
			t.Fatalf("field %v: err = %v, want invalid_request", tc.field, err)
		}
		if tc.kept != (len(failure.Detail) == 1 && failure.Detail["field"] == PolicySessionScopeField) ||
			(!tc.kept && failure.Detail != nil) {
			t.Fatalf("field %v: detail = %v", tc.field, failure.Detail)
		}
	}
}

// session_scope is sent only when set, so an unscoped decide body is
// byte-identical to one from before the member.
func TestPolicyDecisionRequestEncodesSessionScopeOnlyWhenSet(t *testing.T) {
	unscoped, err := EncodePolicyDecisionRequest(PolicyDecisionRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(unscoped), "session_scope") {
		t.Fatalf("unscoped body = %s", unscoped)
	}
	scoped, err := EncodePolicyDecisionRequest(PolicyDecisionRequest{SessionScope: PolicySessionScopeLookup})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(scoped), `"session_scope":"lookup"`) {
		t.Fatalf("scoped body = %s", scoped)
	}
}
