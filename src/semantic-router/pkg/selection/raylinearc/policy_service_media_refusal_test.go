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

// pathfinder#5281: a typed media refusal gets its own class and keeps only a
// sub_reason from the projection's closed set; anything else is "other".
func TestPolicyServiceClassifiesMediaRefusalWithABoundedSubReason(t *testing.T) {
	for _, tc := range []struct {
		subReason any
		want      string
	}{
		{"undecodable_image", "undecodable_image"},
		{"document_unsupported", "document_unsupported"},
		{"image bytes from a client " + strings.Repeat("x", 4096), "other"},
		{42, "other"},
		{nil, "other"},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(http.StatusUnprocessableEntity)
			_ = json.NewEncoder(writer).Encode(map[string]any{"error": "unsupported_request",
				"detail": map[string]any{"reason": "media_refused", "sub_reason": tc.subReason, "note": "dropped"}})
		}))
		client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: time.Second})
		_, err := client.Decide(context.Background(), PolicyDecisionRequest{})
		server.Close()
		var failure *PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != PolicyMediaRefusedClass {
			t.Fatalf("sub_reason %v: err = %v, want class %s", tc.subReason, err, PolicyMediaRefusedClass)
		}
		if len(failure.Detail) != 1 || failure.Detail["sub_reason"] != tc.want {
			t.Fatalf("sub_reason %v: detail = %v, want only sub_reason %s", tc.subReason, failure.Detail, tc.want)
		}
	}
}
