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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
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
	if seen != 9 {
		t.Fatalf("SHA256SUMS pins %d fixtures, want 9", seen)
	}
}

// Every fixture decodes strictly and re-encodes to the same JSON value, so no
// field is dropped, renamed or defaulted on the Go side.
func TestPolicyWireTypesRoundTripTheFixtures(t *testing.T) {
	cases := map[string]any{
		"package_manifest.v4.json": &PolicyPackageManifest{},
		"package_manifest.v5.json": &PolicyPackageManifestV5{},
		// The v5 fixture with both optional encoding_profile members.
		"package_manifest.v5.canonical_v1.json": &PolicyPackageManifestV5{},
		"decision_request.v1.json":              &PolicyDecisionRequest{},
		"decision_request_responses.v1.json":    &PolicyDecisionRequest{},
		"decision_request_relaxed.v1.json":      &PolicyDecisionRequest{},
		"decision_response.v1.json":             &PolicyDecisionResponse{},
		"packages_response.v1.json":             &PolicyPackagesResponse{},
		"error_responses.v1.json":               &[]PolicyErrorResponse{},
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

// A v5 manifest decodes, and each action's control_id is recomputed from its
// control: a control edited without its id, or a level that is not the
// instruction's, is refused (ADR 0107 decision 8).
func TestDecodePolicyPackageManifestV5RecomputesControlIDs(t *testing.T) {
	body := readPolicyFixture(t, "package_manifest.v5.json")
	manifest, err := DecodePolicyPackageManifestV5(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Actions) != 3 || manifest.ThinkingControlsSHA256 == "" {
		t.Fatalf("decoded %+v", manifest)
	}
	if schema, err := PolicyPackageSchemaOf(body); err != nil || schema != PolicyPackageSchemaV5 {
		t.Fatalf("schema = %q, %v", schema, err)
	}
	if _, err := DecodePolicyPackageManifest(body); err == nil {
		t.Fatal("the v4 decoder accepted a v5 manifest")
	}
	for name, edit := range map[string][2]string{
		"control edited without its id": {`"native": "default"`, `"native": "high"`},
		"level not the instruction's": {`"level": "up",
      "trained_arm_ids"`, `"level": "down",
      "trained_arm_ids"`},
		"a control member omitted": {`"budget_tokens": null,
        "instruction": null`, `"instruction": null`},
		"unknown action field": {`"control_id": "628d`, `"surprise": 1, "control_id": "628d`},
	} {
		changed := bytes.Replace(body, []byte(edit[0]), []byte(edit[1]), 1)
		if bytes.Equal(changed, body) {
			t.Fatalf("%s: the fixture no longer holds %q", name, edit[0])
		}
		if _, err := DecodePolicyPackageManifestV5(changed); err == nil {
			t.Errorf("%s: accepted", name)
		}
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

// A Responses request without instructions states them as null, as every
// nullable member of this wire is written.
func TestPolicyResponsesRequestWritesNullInstructions(t *testing.T) {
	encoded, err := json.Marshal(PolicyClientRequest{Input: []json.RawMessage{json.RawMessage(`{"type":"message","role":"user","content":"hi"}`)}})
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != `{"input":[{"type":"message","role":"user","content":"hi"}],"instructions":null}` {
		t.Fatalf("encoded %s", encoded)
	}
	var decoded PolicyClientRequest
	if err := json.Unmarshal(encoded, &decoded); err != nil || decoded.Instructions != nil || len(decoded.Input) != 1 {
		t.Fatalf("decoded %+v, %v", decoded, err)
	}
}

// Published packages state encoding_profile.conversation (pathfinder's
// "canonical_v1" projection) and may state harness_injections; both decoders
// read them, accept the value the contract defines for each, and refuse any
// other.
func TestDecodePolicyPackageConversation(t *testing.T) {
	const anchor = `"tool_definitions": "include_recorded",`
	decoders := map[string]func([]byte) error{
		"package_manifest.v4.json": func(b []byte) error { _, err := DecodePolicyPackageManifest(b); return err },
		"package_manifest.v5.json": func(b []byte) error { _, err := DecodePolicyPackageManifestV5(b); return err },
	}
	for fixture, decode := range decoders {
		body := readPolicyFixture(t, fixture)
		if !bytes.Contains(body, []byte(anchor)) {
			t.Fatalf("%s no longer holds %q", fixture, anchor)
		}
		for member, values := range map[string]map[string]bool{
			"conversation":       {`"canonical_v1"`: true, `"transcript_v9"`: false, `""`: false, `null`: false, `1`: false},
			"harness_injections": {`"strip_claude_code_2_1_v1"`: true, `"keep"`: false, `""`: false, `null`: false},
		} {
			for value, accepted := range values {
				changed := bytes.Replace(body, []byte(anchor), []byte(anchor+` "`+member+`": `+value+`,`), 1)
				if err := decode(changed); (err == nil) != accepted {
					t.Errorf("%s %s %q: err = %v", fixture, member, value, err)
				}
			}
		}
		both := bytes.Replace(body, []byte(anchor),
			[]byte(anchor+` "conversation": "canonical_v1", "harness_injections": "strip_claude_code_2_1_v1",`), 1)
		if err := decode(both); err != nil {
			t.Errorf("%s with both members: %v", fixture, err)
		}
		if err := decode(body); err != nil {
			t.Errorf("%s without conversation: %v", fixture, err)
		}
	}
}

// pathfinder's canonical_v1 fixture decodes, carrying both optional
// encoding_profile members with the contract's values.
func TestDecodePolicyPackageManifestV5CanonicalFixture(t *testing.T) {
	manifest, err := DecodePolicyPackageManifestV5(readPolicyFixture(t, "package_manifest.v5.canonical_v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	profile := manifest.EncodingProfile
	if string(profile.Conversation) != `"canonical_v1"` || string(profile.HarnessInjections) != `"strip_claude_code_2_1_v1"` {
		t.Fatalf("encoding_profile = %s, %s", profile.Conversation, profile.HarnessInjections)
	}
	if len(manifest.Actions) != 3 {
		t.Fatalf("decoded %d actions", len(manifest.Actions))
	}
}

// A relaxed request carries episode_mode; a strict one omits the field, the
// form every caller sent before it existed (pathfinder#3068).
func TestPolicyDecisionRequestEpisodeMode(t *testing.T) {
	var relaxed PolicyDecisionRequest
	if err := decodeStrict(readPolicyFixture(t, "decision_request_relaxed.v1.json"), &relaxed); err != nil {
		t.Fatal(err)
	}
	if relaxed.EpisodeMode != PolicyEpisodeModeRelaxed {
		t.Fatalf("relaxed fixture episode_mode = %q", relaxed.EpisodeMode)
	}
	relaxed.EpisodeMode = ""
	encoded, err := json.Marshal(relaxed)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "episode_mode") {
		t.Fatalf("a strict request names episode_mode: %s", encoded)
	}
}

// A relaxed decision answers session_revision null, which decodes as absent.
func TestDecodePolicyDecisionResponseAcceptsANullRevision(t *testing.T) {
	var body map[string]any
	if err := json.Unmarshal(readPolicyFixture(t, "decision_response.v1.json"), &body); err != nil {
		t.Fatal(err)
	}
	body["encoding"].(map[string]any)["session_revision"] = nil
	raw, _ := json.Marshal(body)
	response, err := DecodePolicyDecisionResponse(raw)
	if err != nil {
		t.Fatalf("decode with a null revision: %v", err)
	}
	if response.Encoding.SessionRevision != nil {
		t.Fatalf("session_revision = %d, want nil", *response.Encoding.SessionRevision)
	}
}

// A relaxed call the service cannot serve gets its own bounded class; any
// other unsupported_request keeps the contract's code.
func TestPolicyServiceClassifiesRelaxedUnsupported(t *testing.T) {
	for _, tc := range []struct {
		detail map[string]any
		want   string
	}{
		{map[string]any{"reason": "relaxed_unsupported_by_encoder"}, PolicyRelaxedUnsupportedClass},
		{map[string]any{"reason": "relaxed_needs_unretained_runtime"}, PolicyRelaxedUnsupportedClass},
		{map[string]any{"reason": "projection_refused"}, "unsupported_request"},
		{map[string]any{}, "unsupported_request"},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(http.StatusUnprocessableEntity)
			_ = json.NewEncoder(writer).Encode(map[string]any{"error": "unsupported_request", "detail": tc.detail})
		}))
		client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: time.Second})
		_, err := client.Decide(context.Background(), PolicyDecisionRequest{})
		server.Close()
		var failure *PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != tc.want {
			t.Fatalf("detail %v: err = %v, want class %s", tc.detail, err, tc.want)
		}
	}
}

// fixtureWith is a fixture's JSON object with edit applied.
func fixtureWith(t *testing.T, name string, edit func(map[string]any)) []byte {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(readPolicyFixture(t, name), &body); err != nil {
		t.Fatal(err)
	}
	edit(body)
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// A service from pathfinder#3120/#3138 on reports package verification on
// decide responses, their shadow scores and the packages listing. VSR accepts
// both values, and still accepts its absence (a service that predates it).
func TestPolicyResponsesAcceptPackageVerification(t *testing.T) {
	for _, value := range []PolicyVerification{PolicyPackageVerified, PolicyPackageUnverified} {
		decide := fixtureWith(t, "decision_response.v1.json", func(body map[string]any) {
			body["package_verification"] = value
			for _, shadow := range body["shadow"].([]any) {
				if scored := shadow.(map[string]any); scored["error"] == nil {
					scored["package_verification"] = value
				}
			}
		})
		response, err := DecodePolicyDecisionResponse(decide)
		if err != nil {
			t.Fatalf("decide with package_verification %q: %v", value, err)
		}
		if response.PackageVerification != value || response.Shadow[0].Scored.PackageVerification != value {
			t.Fatalf("package_verification = %q (shadow %q), want %q",
				response.PackageVerification, response.Shadow[0].Scored.PackageVerification, value)
		}
		packages := fixtureWith(t, "packages_response.v1.json", func(body map[string]any) {
			for _, loaded := range body["packages"].([]any) {
				loaded.(map[string]any)["verification"] = value
			}
		})
		listing, err := DecodePolicyPackagesResponse(packages)
		if err != nil {
			t.Fatalf("packages with verification %q: %v", value, err)
		}
		if got := listing.Packages[0].Verification; got != value {
			t.Fatalf("verification = %q, want %q", got, value)
		}
	}
	response, err := DecodePolicyDecisionResponse(readPolicyFixture(t, "decision_response.v1.json"))
	if err != nil || response.PackageVerification != "" {
		t.Fatalf("decide without package_verification = %v, %v", response, err)
	}
}

// A verification outside the contract's two values, or an explicit null, is
// refused, as any other malformed response is; only omission means unknown.
func TestPolicyResponsesRefuseAnUnknownVerification(t *testing.T) {
	for _, bad := range []any{"trusted", nil, 1} {
		decide := fixtureWith(t, "decision_response.v1.json", func(body map[string]any) {
			body["package_verification"] = bad
		})
		if _, err := DecodePolicyDecisionResponse(decide); err == nil {
			t.Fatalf("decide accepted package_verification %v", bad)
		}
		shadow := fixtureWith(t, "decision_response.v1.json", func(body map[string]any) {
			body["shadow"].([]any)[0].(map[string]any)["package_verification"] = bad
		})
		if _, err := DecodePolicyDecisionResponse(shadow); err == nil {
			t.Fatalf("decide accepted shadow package_verification %v", bad)
		}
		packages := fixtureWith(t, "packages_response.v1.json", func(body map[string]any) {
			body["packages"].([]any)[0].(map[string]any)["verification"] = bad
		})
		if _, err := DecodePolicyPackagesResponse(packages); err == nil {
			t.Fatalf("packages accepted verification %v", bad)
		}
	}
}
