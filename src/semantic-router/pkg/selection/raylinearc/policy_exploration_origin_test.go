package raylinearc

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func explorationCandidate(model string, effort *string) PolicyStageOneCandidate {
	return PolicyStageOneCandidate{"model": &model, "effort": effort}
}

func boundaryRecord(key string, chosen PolicyStageOneCandidate) *PolicyExplorationRecord {
	return &PolicyExplorationRecord{DecisionKey: &key, Chosen: chosen}
}

func heldRecord(chosen PolicyStageOneCandidate, origin *string) *PolicyExplorationRecord {
	return &PolicyExplorationRecord{Held: true, Chosen: chosen, OriginDecisionKey: origin}
}

// Each turn restates the origin from its record: a boundary replaces it, a
// held turn on the boundary's candidate keeps it and reads its key, and
// anything else clears it, so no later held turn joins an older boundary.
func TestNextPolicyExplorationOrigin(t *testing.T) {
	high := "high"
	flash := explorationCandidate("glm-5.3-flash", nil)
	stored := &PolicyExplorationOrigin{DecisionKey: "key-1", Chosen: flash}
	serviceOrigin := "service-key"
	for _, tc := range []struct {
		name     string
		previous *PolicyExplorationOrigin
		record   *PolicyExplorationRecord
		wantNext string
		wantKey  string
	}{
		{name: "first boundary", record: boundaryRecord("key-1", flash), wantNext: "key-1"},
		{name: "later boundary replaces", previous: stored, record: boundaryRecord("key-2", explorationCandidate("claude-opus-5", &high)), wantNext: "key-2"},
		{name: "held on the boundary's candidate", previous: stored, record: heldRecord(explorationCandidate("glm-5.3-flash", nil), nil), wantNext: "key-1", wantKey: "key-1"},
		{name: "held with the service's origin", previous: stored, record: heldRecord(flash, &serviceOrigin), wantNext: "key-1"},
		{name: "held on another model", previous: stored, record: heldRecord(explorationCandidate("claude-opus-5", &high), nil)},
		{name: "held on another effort", previous: stored, record: heldRecord(explorationCandidate("glm-5.3-flash", &high), nil)},
		{name: "held with no recorded boundary", record: heldRecord(flash, nil)},
		{name: "decision without a record", previous: stored},
		{name: "caller-held decision", previous: stored, record: &PolicyExplorationRecord{}},
		{name: "boundary without a chosen candidate", previous: stored, record: boundaryRecord("key-2", nil)},
		{name: "boundary with an oversized key", previous: stored, record: boundaryRecord(strings.Repeat("k", 129), flash)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			next := NextPolicyExplorationOrigin(tc.previous, tc.record)
			if got := ""; next != nil {
				got = next.DecisionKey
				if got != tc.wantNext {
					t.Fatalf("next origin %q, want %q", got, tc.wantNext)
				}
			} else if tc.wantNext != "" {
				t.Fatalf("next origin cleared, want %q", tc.wantNext)
			}
			key := tc.previous.HeldDecisionKey(tc.record)
			if (key == nil) != (tc.wantKey == "") || key != nil && *key != tc.wantKey {
				t.Fatalf("held key %v, want %q", key, tc.wantKey)
			}
		})
	}
	// The kept origin is a copy: changing it leaves the stored one intact.
	kept := NextPolicyExplorationOrigin(stored, heldRecord(flash, nil))
	*kept.Chosen["model"] = "changed"
	if *stored.Chosen["model"] != "glm-5.3-flash" {
		t.Fatal("the kept origin aliases the stored one")
	}
}

// Next drops the origin, so a turn that did not restate it (a capacity hold
// that decided nothing) leaves none; Clone keeps a deep copy.
func TestPolicyStateNextDropsTheExplorationOrigin(t *testing.T) {
	state := &PolicyEpisodeState{Exploration: &PolicyExplorationOrigin{DecisionKey: "key-1", Chosen: explorationCandidate("m", nil)}}
	cloned := state.Clone()
	if cloned.Exploration == nil || cloned.Exploration == state.Exploration {
		t.Fatalf("clone origin %p of %p", cloned.Exploration, state.Exploration)
	}
	messages := []json.RawMessage{json.RawMessage(`{"role":"user","content":"go"}`)}
	if next := state.Next(messages, actionIDFor(0), "arm"); next.Exploration != nil {
		t.Fatalf("Next kept %+v", next.Exploration)
	}
}

// A record whose policy state carries an origin is written as v9 and reads
// back whole; one without keeps its earlier schema and bytes, so existing
// state still reads; each schema is exactly its record, and a malformed
// stored origin is refused.
func TestEpisodeRecordKeepsTheExplorationOrigin(t *testing.T) {
	now := cacheContextNow
	state, _ := NewEpisodeState(2)
	if err := state.Commit(0, 10, now.Add(-time.Second)); err != nil {
		t.Fatal(err)
	}
	state.Warmth[0].Call = &CallCache{ActionID: actionIDFor(0), CompletedAt: now.Add(-time.Second), ContextEpoch: "0"}
	state.Policy = &PolicyEpisodeState{}
	v8, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(v8, []byte(episodeStateSchemaV8)) || bytes.Contains(v8, []byte(`"exploration"`)) {
		t.Fatalf("a record without an origin changed its bytes: %s", v8)
	}
	if _, _, err = unmarshalEpisodeState(v8, 2, now); err != nil {
		t.Fatalf("a v8 record was refused: %v", err)
	}

	high := "high"
	state.Policy.Exploration = &PolicyExplorationOrigin{DecisionKey: "3b1f0c5a", Chosen: explorationCandidate("glm-5.3-flash", nil)}
	v9, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(v9, []byte(episodeStateSchemaV9)) {
		t.Fatalf("a record with an origin is not v9: %s", v9)
	}
	restored, _, err := unmarshalEpisodeState(v9, 2, now)
	if err != nil {
		t.Fatal(err)
	}
	origin := restored.Policy.Exploration
	if origin == nil || origin.DecisionKey != "3b1f0c5a" || *origin.Chosen["model"] != "glm-5.3-flash" ||
		origin.Chosen["effort"] != nil || len(origin.Chosen) != 2 || restored.Warmth[0].Call.ContextEpoch != "0" {
		t.Fatalf("restored origin %+v", origin)
	}
	if !sameStageOneCandidate(origin.Chosen, explorationCandidate("glm-5.3-flash", nil)) ||
		sameStageOneCandidate(origin.Chosen, explorationCandidate("glm-5.3-flash", &high)) {
		t.Fatal("the restored candidate does not compare as stored")
	}

	// An origin alone, with no calls, is v9 as well.
	bare, _ := NewEpisodeState(2)
	bare.Policy = &PolicyEpisodeState{Exploration: state.Policy.Exploration.Clone()}
	bareV9, err := marshalEpisodeState(bare, 1, now)
	if err != nil || !bytes.Contains(bareV9, []byte(episodeStateSchemaV9)) {
		t.Fatalf("bare origin record %s: %v", bareV9, err)
	}
	if _, _, err = unmarshalEpisodeState(bareV9, 2, now); err != nil {
		t.Fatalf("bare v9 refused: %v", err)
	}

	// A staged boundary's origin alone, before the episode has a policy
	// context, is v9 too, and it survives the round trip.
	staged, _ := NewEpisodeState(2)
	staged.PolicyBoundary = &PolicyBoundaryDecision{
		Arm: 1, PrefixLen: 1, PrefixDigest: strings.Repeat("a", 64), Exploration: state.Policy.Exploration.Clone(),
	}
	stagedV9, err := marshalEpisodeState(staged, 1, now)
	if err != nil || !bytes.Contains(stagedV9, []byte(episodeStateSchemaV9)) {
		t.Fatalf("staged boundary origin record %s: %v", stagedV9, err)
	}
	restoredStaged, _, err := unmarshalEpisodeState(stagedV9, 2, now)
	if err != nil || restoredStaged.PolicyBoundary.Exploration == nil ||
		restoredStaged.PolicyBoundary.Exploration.DecisionKey != "3b1f0c5a" {
		t.Fatalf("staged boundary origin restored as %+v: %v", restoredStaged, err)
	}
	if _, _, err := unmarshalEpisodeState(
		bytes.Replace(stagedV9, []byte(episodeStateSchemaV9), []byte(episodeStateSchema), 1), 2, now,
	); err == nil {
		t.Fatal("a staged boundary origin labelled v3 was accepted")
	}

	for name, payload := range map[string][]byte{
		"v9 labelled v8":      bytes.Replace(v9, []byte(episodeStateSchemaV9), []byte(episodeStateSchemaV8), 1),
		"v8 labelled v9":      bytes.Replace(v8, []byte(episodeStateSchemaV8), []byte(episodeStateSchemaV9), 1),
		"bare v9 labelled v3": bytes.Replace(bareV9, []byte(episodeStateSchemaV9), []byte(episodeStateSchema), 1),
		"empty decision key":  bytes.Replace(v9, []byte(`"decision_key":"3b1f0c5a"`), []byte(`"decision_key":""`), 1),
		"empty candidate":     bytes.Replace(v9, []byte(`"chosen":{"effort":null,"model":"glm-5.3-flash"}`), []byte(`"chosen":{}`), 1),
		"unknown member":      bytes.Replace(v9, []byte(`"decision_key":"3b1f0c5a"`), []byte(`"decision_key":"3b1f0c5a","extra":1`), 1),
	} {
		if bytes.Equal(payload, v9) || bytes.Equal(payload, v8) || bytes.Equal(payload, bareV9) {
			t.Fatalf("%s: the payload did not change", name)
		}
		if _, _, err := unmarshalEpisodeState(payload, 2, now); err == nil {
			t.Fatalf("%s was accepted", name)
		}
	}
}
