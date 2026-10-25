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
	"strings"
	"testing"
	"time"
)

// The delta decide fixture encodes, through the router's own writer, to the
// fixture's JSON value: its shell carries no messages member and the delta's
// messages follow it.
func TestEncodePolicyDecisionRequestWritesTheTurnDeltaFixture(t *testing.T) {
	body := readPolicyFixture(t, "decision_request_turn_delta.v1.json")
	var request PolicyDecisionRequest
	if err := decodeStrict(body, &request); err != nil {
		t.Fatal(err)
	}
	if request.Request.Shell != PolicyShellMessages || request.TurnDelta == nil || len(request.TurnDelta.Messages) != 1 {
		t.Fatalf("decoded %+v", request)
	}
	encoded, err := EncodePolicyDecisionRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	var want, got any
	if err := json.Unmarshal(body, &want); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("delta decide encodes as\n%s", encoded)
	}
	// The delta's messages are the client's bytes, unescaped.
	request.TurnDelta.Messages = []json.RawMessage{json.RawMessage(`{"role":"user","content":"a <b> & c"}`)}
	encoded, err = EncodePolicyDecisionRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"messages":[{"role":"user","content":"a <b> & c"}]`) {
		t.Fatalf("delta messages were rewritten: %s", encoded)
	}
}

// A Responses shell is instructions and tools, never input.
func TestPolicyClientRequestResponsesShellRoundTrips(t *testing.T) {
	instructions := "be brief"
	shell := PolicyClientRequest{Shell: PolicyShellInput, Instructions: &instructions, Tools: json.RawMessage(`[]`)}
	encoded, err := shell.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != `{"instructions":"be brief","tools":[]}` {
		t.Fatalf("shell = %s", encoded)
	}
	var decoded PolicyClientRequest
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Shell != PolicyShellInput || decoded.Input != nil || *decoded.Instructions != instructions {
		t.Fatalf("decoded %+v", decoded)
	}
}

// The append writes the fixture's value, and its reply decodes strictly with
// the lease; a reply without hold, or a lease bound to another prefix, is
// refused.
func TestPolicyAppendWireMatchesTheFixtures(t *testing.T) {
	body := readPolicyFixture(t, "append_request.v1.json")
	var request PolicyAppendRequest
	if err := decodeStrict(body, &request); err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodePolicyAppendRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	var want, got any
	_ = json.Unmarshal(body, &want)
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("append encodes as\n%s", encoded)
	}
	reply := readPolicyFixture(t, "append_response.v1.json")
	response, err := DecodePolicyAppendResponse(reply)
	if err != nil {
		t.Fatal(err)
	}
	if response.Hold == nil || response.Hold.TTLSeconds != 120 || response.Messages != 4 {
		t.Fatalf("decoded %+v", response)
	}
	var raw map[string]any
	_ = json.Unmarshal(reply, &raw)
	raw["hold"] = nil
	noLease, _ := json.Marshal(raw)
	if response, err := DecodePolicyAppendResponse(noLease); err != nil || response.Hold != nil {
		t.Fatalf("a null hold: %+v, %v", response, err)
	}
	delete(raw, "hold")
	missing, _ := json.Marshal(raw)
	if _, err := DecodePolicyAppendResponse(missing); err == nil {
		t.Fatal("a reply without hold was accepted")
	}
	_ = json.Unmarshal(reply, &raw)
	raw["hold"].(map[string]any)["valid_while"].(map[string]any)["prefix_digest"] = strings.Repeat("0", 64)
	other, _ := json.Marshal(raw)
	if _, err := DecodePolicyAppendResponse(other); err == nil {
		t.Fatal("a lease bound to another prefix was accepted")
	}
}

// Append posts to the append path; a service that predates it answers 404,
// which is its own class, and a staged-turn refusal keeps its reason.
func TestPolicyServiceClientAppendClassesItsFailures(t *testing.T) {
	reply := readPolicyFixture(t, "append_response.v1.json")
	var status int
	var body string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/rayline/arc/policy/append" {
			t.Errorf("path %s", request.URL.Path)
		}
		writer.WriteHeader(status)
		_, _ = writer.Write([]byte(body))
	}))
	defer server.Close()
	client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: time.Second})
	var request PolicyAppendRequest
	_ = decodeStrict(readPolicyFixture(t, "append_request.v1.json"), &request)
	status, body = http.StatusOK, string(reply)
	if response, err := client.Append(context.Background(), request); err != nil || response.Hold == nil {
		t.Fatalf("append: %+v, %v", response, err)
	}
	for _, tc := range []struct {
		status int
		body   string
		class  string
	}{
		{http.StatusNotFound, `{"detail":"Not Found"}`, PolicyAppendUnsupportedClass},
		{http.StatusMethodNotAllowed, ``, PolicyAppendUnsupportedClass},
		{http.StatusUnprocessableEntity, `{"error":"unsupported_request","detail":{"reason":"session_mismatch"}}`, PolicySessionMismatchClass},
		{http.StatusUnprocessableEntity, `{"error":"unsupported_request","detail":{"reason":"turn_delta_unsupported_by_package"}}`, PolicyTurnDeltaUnsupportedClass},
		{http.StatusOK, `{"schema_version":"other"}`, "decode"},
	} {
		status, body = tc.status, tc.body
		_, err := client.Append(context.Background(), request)
		var failure *PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != tc.class {
			t.Errorf("status %d %s: %v, want class %s", tc.status, tc.body, err, tc.class)
		}
	}
}
