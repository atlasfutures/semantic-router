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
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

const policyFixtures = "testdata/policy_service"

func readPolicyFixture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(policyFixtures, name))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// The fixtures are pathfinder's; SHA256SUMS pins the copy so drift is a
// deliberate re-sync, not an edit.
func TestPolicyFixturesMatchTheirPinnedDigests(t *testing.T) {
	seen := 0
	for _, line := range strings.Split(strings.TrimSpace(string(readPolicyFixture(t, "SHA256SUMS"))), "\n") {
		want, name, ok := strings.Cut(line, "  ")
		if !ok {
			t.Fatalf("malformed SHA256SUMS line %q", line)
		}
		digest := sha256.Sum256(readPolicyFixture(t, name))
		if got := hex.EncodeToString(digest[:]); got != want {
			t.Errorf("%s: sha256 %s, pinned %s", name, got, want)
		}
		seen++
	}
	if seen != 5 {
		t.Fatalf("SHA256SUMS pins %d fixtures, want 5", seen)
	}
}

// Every fixture decodes strictly and re-encodes to the same JSON value, so no
// field is dropped, renamed or defaulted on the Go side.
func TestPolicyWireTypesRoundTripTheFixtures(t *testing.T) {
	cases := map[string]any{
		"package_manifest.v4.json":  &PolicyPackageManifest{},
		"decision_request.v1.json":  &PolicyDecisionRequest{},
		"decision_response.v1.json": &PolicyDecisionResponse{},
		"packages_response.v1.json": &PolicyPackagesResponse{},
		"error_responses.v1.json":   &[]PolicyErrorResponse{},
	}
	for name, target := range cases {
		body := readPolicyFixture(t, name)
		if err := decodeStrict(body, target); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		encoded, err := json.Marshal(target)
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
			t.Errorf("%s does not round-trip:\nwant %v\ngot  %v", name, want, got)
		}
	}
}

func TestDecodePolicyDecisionResponseRefusesAmbiguousDecisions(t *testing.T) {
	body := readPolicyFixture(t, "decision_response.v1.json")
	response, err := DecodePolicyDecisionResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Shadow) != 2 || response.Shadow[0].Scored == nil || response.Shadow[1].Failure == nil {
		t.Fatalf("shadow results decoded as %+v", response.Shadow)
	}
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatal(err)
	}
	for _, action := range raw["actions"].([]any) {
		action.(map[string]any)["selected"] = true
	}
	twoSelected, _ := json.Marshal(raw)
	if _, err := DecodePolicyDecisionResponse(twoSelected); err == nil {
		t.Fatal("a response marking every action selected was accepted")
	}
	shadow := raw["shadow"].([]any)[0].(map[string]any)
	for _, action := range raw["actions"].([]any) {
		action.(map[string]any)["selected"] = false
	}
	raw["actions"].([]any)[1].(map[string]any)["selected"] = true
	for _, action := range shadow["actions"].([]any) {
		action.(map[string]any)["selected"] = false
	}
	shadowNone, _ := json.Marshal(raw)
	if _, err := DecodePolicyDecisionResponse(shadowNone); err == nil {
		t.Fatal("a shadow result marking no action selected was accepted")
	}
	unknown := bytes.Replace(body, []byte(`"quality_parity_established"`), []byte(`"surprise": 1, "quality_parity_established"`), 1)
	if _, err := DecodePolicyDecisionResponse(unknown); err == nil {
		t.Fatal("a response with an unknown field was accepted")
	}
}

func TestDecodePolicyPackageManifestRefusesLivePrices(t *testing.T) {
	body := readPolicyFixture(t, "package_manifest.v4.json")
	if _, err := DecodePolicyPackageManifest(body); err != nil {
		t.Fatal(err)
	}
	live := bytes.Replace(body, []byte(`"live_prices_affect_decisions": false`), []byte(`"live_prices_affect_decisions": true`), 1)
	if bytes.Equal(live, body) {
		t.Fatal("fixture no longer carries live_prices_affect_decisions")
	}
	if _, err := DecodePolicyPackageManifest(live); err == nil {
		t.Fatal("a package letting live prices affect decisions was accepted")
	}
}

func TestDecodePolicyPackagesResponseRefusesAnotherSchema(t *testing.T) {
	body := readPolicyFixture(t, "packages_response.v1.json")
	if _, err := DecodePolicyPackagesResponse(body); err != nil {
		t.Fatal(err)
	}
	older := bytes.Replace(body, []byte(PolicyPackagesSchema), []byte("rayline.arc.policy-packages.v0"), 1)
	if _, err := DecodePolicyPackagesResponse(older); err == nil {
		t.Fatal("a packages listing with another schema was accepted")
	}
}

func TestPolicyServiceClientBoundsTheConnect(t *testing.T) {
	for configured, want := range map[time.Duration]time.Duration{0: DefaultPolicyServiceConnectTimeout, 2 * time.Second: 2 * time.Second} {
		client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: "https://example.invalid", ConnectTimeout: configured})
		transport, ok := client.http.Transport.(*http.Transport)
		if !ok || transport.TLSHandshakeTimeout != want || transport.DialContext == nil {
			t.Fatalf("connect %v: transport = %+v", configured, transport)
		}
	}
	for code, want := range map[string]string{"session_busy": "session_busy", "no such code": "service_error"} {
		if got := PolicyServiceErrorClass(code); got != want {
			t.Fatalf("%q -> %q", code, got)
		}
	}
}
