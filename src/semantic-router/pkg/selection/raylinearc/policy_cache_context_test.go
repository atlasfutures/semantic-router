package raylinearc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
)

var cacheContextNow = time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)

func actionIDFor(arm int) string {
	return fmt.Sprintf("%064x", arm+1)
}

func intPtr(value int) *int { return &value }

// identityArms binds actionIDFor(arm) to arm.
func identityArms(actionID string) (int, bool) {
	for arm := 0; arm < 64; arm++ {
		if actionIDFor(arm) == actionID {
			return arm, true
		}
	}
	return 0, false
}

func TestEpisodeRecordKeepsTheCallEvidence(t *testing.T) {
	state, err := NewEpisodeState(3)
	if err != nil {
		t.Fatal(err)
	}
	if err := state.Commit(0, 900, cacheContextNow.Add(-time.Minute)); err != nil {
		t.Fatal(err)
	}
	state.Warmth[0].Call = &cachecontext.CallCache{
		ActionID: actionIDFor(0), CompletedAt: cacheContextNow.Add(-time.Minute), TTLSeconds: 3600, TTLSource: "anthropic cache_control ephemeral 1h",
		PromptTokens: intPtr(1800), CacheReadTokens: intPtr(0), CacheWriteTokens: nil,
	}
	if err := state.Commit(1, 500, cacheContextNow.Add(-time.Second)); err != nil {
		t.Fatal(err)
	}
	state.Warmth[1].Call = &cachecontext.CallCache{ActionID: actionIDFor(1), CompletedAt: cacheContextNow.Add(-time.Second)}
	payload, err := marshalEpisodeState(state, 3, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	// An older router refuses the record by its schema.
	if !bytes.Contains(payload, []byte(`"schema_version":"`+episodeStateSchemaV7+`"`)) {
		t.Fatalf("a record carrying calls is not %s: %s", episodeStateSchemaV7, payload)
	}
	restored, _, err := unmarshalEpisodeState(payload, 3, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(cachecontext.BuildPolicyCacheContext(restored.Warmth, cacheContextNow, identityArms))
	want, _ := json.Marshal(cachecontext.BuildPolicyCacheContext(state.Warmth, cacheContextNow, identityArms))
	if string(got) != string(want) {
		t.Fatalf("restored\n got %s\nwant %s", got, want)
	}
	if !strings.Contains(string(got), `"cache_read_tokens":0,"cache_write_tokens":null`) {
		t.Fatalf("a reported zero must stay zero and an unreported leg null: %s", got)
	}
	if !strings.Contains(string(got), `"ttl_s":null,"ttl_source":null,"prompt_tokens":null`) {
		t.Fatalf("arm 1's unknowns must be null: %s", got)
	}

	// The same record as a router before this change wrote it: no call, and
	// its earlier schema.
	var old map[string]interface{}
	if err := json.Unmarshal(payload, &old); err != nil {
		t.Fatal(err)
	}
	for _, warmth := range old["warmth"].([]interface{}) {
		if warmth != nil {
			delete(warmth.(map[string]interface{}), "call")
		}
	}
	old["schema_version"] = episodeStateSchemaV2
	oldPayload, _ := json.Marshal(old)
	oldState, _, err := unmarshalEpisodeState(oldPayload, 3, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	if oldState.Warmth[0] == nil || oldState.Warmth[0].Call != nil {
		t.Fatalf("an old record's warmth = %+v, want no call", oldState.Warmth[0])
	}
	if context := cachecontext.BuildPolicyCacheContext(oldState.Warmth, cacheContextNow, identityArms); len(context.Actions) != 0 {
		t.Fatalf("an old record names no action, so it reports none: %+v", context.Actions)
	}
}

// Pathfinder's cache_context fixture decodes strictly into the router's
// request, and the decide body writes it back as it is.
func TestPathfinderGoldenCacheContextEncodesAsTheDecideBody(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("testdata", "cache_context", "pathfinder_decision_request_cache_context.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	var request PolicyDecisionRequest
	if err := decodeStrict(body, &request); err != nil {
		t.Fatal(err)
	}
	if request.CacheContext == nil || len(request.CacheContext.Actions) != 2 {
		t.Fatalf("decoded cache_context %+v", request.CacheContext)
	}
	encoded, err := EncodePolicyDecisionRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	requireSameJSON(t, "golden decide body", body, encoded)
}

// A request without cache_context leaves it out, as before the field.
func TestPolicyDecisionRequestOmitsAbsentCacheContext(t *testing.T) {
	encoded, err := EncodePolicyDecisionRequest(PolicyDecisionRequest{SchemaVersion: PolicyDecisionRequestSchema})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte("cache_context")) {
		t.Fatalf("absent cache_context was written: %s", encoded)
	}
}

// v7 is exactly the record whose warmth carries a call: one without calls
// keeps its earlier schema, and a mislabelled record is refused.
func TestEpisodeStateV7IsExactlyTheRecordWithCalls(t *testing.T) {
	state, _ := NewEpisodeState(2)
	if err := state.Commit(0, 10, cacheContextNow.Add(-time.Second)); err != nil {
		t.Fatal(err)
	}
	plain, err := marshalEpisodeState(state, 1, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(plain, []byte(episodeStateSchemaV7)) {
		t.Fatalf("a record without calls was written as v7: %s", plain)
	}
	relabelled := bytes.Replace(plain, []byte(episodeStateSchemaV2), []byte(episodeStateSchemaV7), 1)
	if _, _, err := unmarshalEpisodeState(relabelled, 2, cacheContextNow); err == nil {
		t.Fatal("a v7 record without calls was accepted")
	}
	state.Warmth[0].Call = &cachecontext.CallCache{ActionID: actionIDFor(0)}
	withCalls, err := marshalEpisodeState(state, 1, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	downgraded := bytes.Replace(withCalls, []byte(episodeStateSchemaV7), []byte(episodeStateSchemaV2), 1)
	if _, _, err := unmarshalEpisodeState(downgraded, 2, cacheContextNow); err == nil {
		t.Fatal("a v2 record carrying calls was accepted")
	}
}

// Each entry carries the epoch its call was made in: one from before a
// compaction keeps its older epoch, so the service reads it as another
// context; one with no recorded epoch omits the member.
// A call's epoch survives the record, written as v8; a v7 record (calls,
// no epochs) still reads, and each schema is exactly its record.
func TestEpisodeRecordKeepsTheCallsEpoch(t *testing.T) {
	state, _ := NewEpisodeState(2)
	if err := state.Commit(0, 10, cacheContextNow.Add(-time.Second)); err != nil {
		t.Fatal(err)
	}
	state.Warmth[0].Call = &cachecontext.CallCache{ActionID: actionIDFor(0), CompletedAt: cacheContextNow.Add(-time.Second)}
	v7, err := marshalEpisodeState(state, 1, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(v7, []byte(episodeStateSchemaV7)) {
		t.Fatalf("calls without epochs are not v7: %s", v7)
	}
	if _, _, err := unmarshalEpisodeState(v7, 2, cacheContextNow); err != nil {
		t.Fatalf("a v7 record was refused: %v", err)
	}
	state.Warmth[0].Call = &cachecontext.CallCache{ActionID: actionIDFor(0), CompletedAt: cacheContextNow.Add(-time.Second), ContextEpoch: "2"}
	v8, err := marshalEpisodeState(state, 1, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(v8, []byte(episodeStateSchemaV8)) {
		t.Fatalf("a call with an epoch is not v8: %s", v8)
	}
	restored, _, err := unmarshalEpisodeState(v8, 2, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	if restored.Warmth[0].Call.ContextEpoch != "2" {
		t.Fatalf("restored epoch %q", restored.Warmth[0].Call.ContextEpoch)
	}
	for name, payload := range map[string][]byte{
		"v8 labelled v7":      bytes.Replace(v8, []byte(episodeStateSchemaV8), []byte(episodeStateSchemaV7), 1),
		"v7 labelled v8":      bytes.Replace(v7, []byte(episodeStateSchemaV7), []byte(episodeStateSchemaV8), 1),
		"v8 labelled earlier": bytes.Replace(v8, []byte(episodeStateSchemaV8), []byte(episodeStateSchemaV2), 1),
	} {
		if _, _, err := unmarshalEpisodeState(payload, 2, cacheContextNow); err == nil {
			t.Fatalf("%s was accepted", name)
		}
	}
}
