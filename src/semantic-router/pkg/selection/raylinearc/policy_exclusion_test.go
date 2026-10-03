package raylinearc

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// An exclusion lasts exactly as long as its context: a turn that extends the
// conversation, and a side call, keep it; a compaction and a prefix break
// start a fresh context and lift it.
func TestPolicyExclusionLastsAsLongAsItsContext(t *testing.T) {
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	roles := []string{"user"}
	stored := (&PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a").WithExclusion("vendor/off", "refusal")
	next := append(append([]json.RawMessage(nil), task...),
		json.RawMessage(`{"role":"assistant","content":"done"}`), json.RawMessage(`{"role":"user","content":"thanks"}`))
	nextRoles := []string{"user", "assistant", "user"}
	rewritten := []json.RawMessage{json.RawMessage(`{"role":"user","content":"something else"}`)}

	for name, tc := range map[string]struct {
		messages []json.RawMessage
		roles    []string
		signals  PolicyTurnSignals
		keeps    bool
	}{
		"extends":      {next, nextRoles, PolicyTurnSignals{}, true},
		"side call":    {next, nextRoles, PolicyTurnSignals{CallKind: PolicyCallSide}, true},
		"compaction":   {next, nextRoles, PolicyTurnSignals{Compaction: &PolicyCompactionSignal{Ordinal: 1}}, false},
		"prefix break": {rewritten, roles, PolicyTurnSignals{}, false},
	} {
		turn, _, _ := PolicyTurn(stored, tc.messages, tc.roles, 1, tc.signals)
		if turn.Excludes("vendor/off") != tc.keeps {
			t.Fatalf("%s: exclusions %v, want kept=%v", name, turn.Exclusions, tc.keeps)
		}
		if tc.keeps && !turn.Next(tc.messages, strings.Repeat("b", 64), "arm-b").Excludes("vendor/off") {
			t.Fatalf("%s: the committed turn dropped the exclusion", name)
		}
	}
}

// An exclusion recorded before any turn committed opens on the empty
// prefix, which the next request extends, so it is not read as a fresh
// context.
func TestPolicyExclusionBeforeTheFirstCommitIsKept(t *testing.T) {
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	start, _, _ := PolicyTurn(nil, task, []string{"user"}, 0, PolicyTurnSignals{})
	for name, context := range map[string]*PolicyEpisodeState{"no state": nil, "session start": start} {
		excluded := context.WithExclusion("vendor/off", "refusal")
		if err := excluded.Validate(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		turn, _, transition := PolicyTurn(excluded, task, []string{"user"}, 0, PolicyTurnSignals{})
		if !turn.Excludes("vendor/off") || transition != PolicyTransitionExtends {
			t.Fatalf("%s: transition %s, exclusions %v", name, transition, turn.Exclusions)
		}
	}
}

func TestPolicyExclusionIsIdempotentAndBounded(t *testing.T) {
	state := &PolicyEpisodeState{}
	state = state.WithExclusion("vendor/off", "refusal").WithExclusion("vendor/off", "refusal")
	if len(state.Exclusions) != 1 {
		t.Fatalf("a repeated exclusion was recorded twice: %v", state.Exclusions)
	}
	if cloned := state.Clone(); len(cloned.Exclusions) == 1 {
		cloned.Exclusions[0].Model = "vendor/other"
		if !state.Excludes("vendor/off") {
			t.Fatal("a clone shares its exclusions with the original")
		}
	}
	for index := 0; index < MaxPolicyExclusions+4; index++ {
		state = state.WithExclusion("vendor/m"+strings.Repeat("x", index), "refusal")
	}
	if len(state.Exclusions) != MaxPolicyExclusions || state.Validate() != nil {
		t.Fatalf("exclusions grew to %d", len(state.Exclusions))
	}
	state.Exclusions = append(state.Exclusions, PolicyExclusion{Model: "vendor/over", Class: "refusal"})
	if state.Validate() == nil {
		t.Fatal("an over-bound record validated")
	}
	if (&PolicyEpisodeState{Exclusions: []PolicyExclusion{{Model: "vendor/off"}}}).Validate() == nil {
		t.Fatal("an exclusion without a class validated")
	}
}

// Exclusions are v4 and written only when present, so every other episode
// keeps its v3 bytes, and an older router refuses a v4 record by its schema.
func TestEpisodeStateWireCarriesExclusionsUnderV4(t *testing.T) {
	now := time.Date(2026, 10, 3, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	state.Policy = (&PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a")
	plain, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(plain), "exclusions") || !strings.Contains(string(plain), "episode-state.v3") {
		t.Fatalf("an episode without exclusions changed shape: %s", plain)
	}

	state.Policy = state.Policy.WithExclusion("vendor/off", "refusal")
	payload, err := marshalEpisodeState(state, 2, now)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(payload), `"schema_version":"rayline.arc.episode-state.v4"`) {
		t.Fatalf("exclusions written without v4: %s", payload)
	}
	decoded, _, err := unmarshalEpisodeState(payload, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if !decoded.Policy.Excludes("vendor/off") || decoded.Policy.Exclusions[0].Class != "refusal" {
		t.Fatalf("exclusions did not round-trip: %+v", decoded.Policy.Exclusions)
	}
	v3 := strings.Replace(string(payload), "episode-state.v4", "episode-state.v3", 1)
	if _, _, err := unmarshalEpisodeState([]byte(v3), 1, now); err == nil {
		t.Fatal("a v3 record carrying exclusions was accepted")
	}
	v4 := strings.Replace(string(plain), "episode-state.v3", "episode-state.v4", 1)
	if _, _, err := unmarshalEpisodeState([]byte(v4), 1, now); err == nil {
		t.Fatal("a v4 record without exclusions was accepted")
	}
}
