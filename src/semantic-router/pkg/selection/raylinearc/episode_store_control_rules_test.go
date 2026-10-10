package raylinearc

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkingcontrol"
)

// A stored episode keeps its first control's rule set, so a fold episode's
// next turn, reloaded from the store, still carries the rules it was founded
// with and is not refused as a rule-set switch.
func TestStoredControlPlacementKeepsTheRuleSet(t *testing.T) {
	anchor, previous := thinkingcontrol.Anchor{Index: 0, Digest: "d"}, "c"
	for _, instruction := range []thinkingcontrol.Instruction{
		{},
		{InbandSystem: thinkingcontrol.InbandSystemFold},
		{Unit: thinkingcontrol.UnitConfigurationUpdate},
	} {
		first := thinkingcontrol.Control{Native: "default", Instruction: &instruction}
		placements := []ControlPlacement{{Key: "k", State: thinkingcontrol.PlacerState{
			Format: thinkingcontrol.FormatMessages, First: &first, Ledger: []thinkingcontrol.LedgerItem{},
			PreviousAnchor: &anchor, PreviousControl: &previous, Calls: 1,
		}}}
		raw, err := json.Marshal(controlPlacementsToWire(placements))
		if err != nil {
			t.Fatal(err)
		}
		var wires []episodeControlWire
		if err = json.Unmarshal(raw, &wires); err != nil {
			t.Fatal(err)
		}
		back, err := controlPlacementsFromWire(wires)
		if err != nil {
			t.Fatal(err)
		}
		got := back[0].State.First.Instruction
		if got == nil || got.Unit != instruction.Unit || got.InbandSystem != instruction.InbandSystem {
			t.Fatalf("stored rule set %+v came back as %+v", instruction, got)
		}
	}
}

func TestStoredControlPlacementRefusesAnUnknownRuleSet(t *testing.T) {
	bad := []episodeControlFirstWire{
		{Native: "default", Lever: true, InbandSystem: "fold_later_v9"},
		{Native: "default", Lever: true, Unit: "other_unit"},
		{Native: "default", Lever: true, Unit: thinkingcontrol.UnitConfigurationUpdate, InbandSystem: thinkingcontrol.InbandSystemFold},
		{Native: "default", InbandSystem: thinkingcontrol.InbandSystemFold},
	}
	for _, first := range bad {
		wire := episodeControlWire{
			Key: "k", Format: thinkingcontrol.FormatMessages, First: &first, Texts: []string{},
			Ledger: []episodeControlItemWire{}, Calls: 1,
		}
		if _, err := controlPlacementsFromWire([]episodeControlWire{wire}); err == nil {
			t.Fatalf("an unknown stored rule set %+v loaded", first)
		}
	}
}

// A stored episode keeps the previous call's ADR 0129 refusal, so a retry
// reloaded from the store repeats it rather than reporting nothing refused.
func TestStoredControlPlacementKeepsTheRefusal(t *testing.T) {
	anchor, previous, refused := thinkingcontrol.Anchor{Index: 0, Digest: "d"}, "c", thinkingcontrol.RefusedImageToolTail
	first := thinkingcontrol.Control{Native: "default", Instruction: &thinkingcontrol.Instruction{}}
	placements := []ControlPlacement{{Key: "k", State: thinkingcontrol.PlacerState{
		Format: thinkingcontrol.FormatChat, First: &first, Ledger: []thinkingcontrol.LedgerItem{},
		PreviousAnchor: &anchor, PreviousControl: &previous, PreviousRefused: &refused, Calls: 1,
	}}}
	raw, err := json.Marshal(controlPlacementsToWire(placements))
	if err != nil {
		t.Fatal(err)
	}
	var wires []episodeControlWire
	if err = json.Unmarshal(raw, &wires); err != nil {
		t.Fatal(err)
	}
	back, err := controlPlacementsFromWire(wires)
	if err != nil {
		t.Fatal(err)
	}
	if got := back[0].State.PreviousRefused; got == nil || *got != refused {
		t.Fatalf("stored refusal came back as %v", got)
	}
}

// A stored control's rule set and refusal are v5 and written only when
// present, so every other episode keeps its v3 or v4 bytes, and a router
// that predates them refuses the record by its schema rather than failing its
// strict decode on an unknown field.
func TestEpisodeStateWireCarriesControlRulesAndRefusalsUnderV5(t *testing.T) {
	now := time.Date(2026, 10, 8, 12, 0, 0, 0, time.UTC)
	anchor, previous, refused := thinkingcontrol.Anchor{Index: 0, Digest: "d"}, "c", thinkingcontrol.RefusedImageToolTail
	placement := func(instruction thinkingcontrol.Instruction, refusal *string) []ControlPlacement {
		first := thinkingcontrol.Control{Native: "default", Instruction: &instruction}
		return []ControlPlacement{{Key: "k", State: thinkingcontrol.PlacerState{
			Format: thinkingcontrol.FormatChat, First: &first, Ledger: []thinkingcontrol.LedgerItem{},
			PreviousAnchor: &anchor, PreviousControl: &previous, PreviousRefused: refusal, Calls: 1,
		}}}
	}
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	for _, tc := range []struct {
		name       string
		controls   []ControlPlacement
		exclusions bool
		want       string
	}{
		{"text rules", placement(thinkingcontrol.Instruction{}, nil), false, "v3"},
		{"text rules with exclusions", placement(thinkingcontrol.Instruction{}, nil), true, "v4"},
		{"fold rules", placement(thinkingcontrol.Instruction{InbandSystem: thinkingcontrol.InbandSystemFold}, nil), false, "v5"},
		{"unit rules", placement(thinkingcontrol.Instruction{Unit: thinkingcontrol.UnitConfigurationUpdate}, nil), false, "v5"},
		{"a refusal", placement(thinkingcontrol.Instruction{}, &refused), false, "v5"},
		{"a refusal with exclusions", placement(thinkingcontrol.Instruction{}, &refused), true, "v5"},
	} {
		state, err := NewEpisodeState(1)
		if err != nil {
			t.Fatal(err)
		}
		state.Policy = (&PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a")
		if tc.exclusions {
			state.Policy = state.Policy.WithExclusion("vendor/off", "refusal")
		}
		state.Controls = tc.controls
		payload, err := marshalEpisodeState(state, 1, now)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if !strings.Contains(string(payload), `"schema_version":"rayline.arc.episode-state.`+tc.want+`"`) {
			t.Fatalf("%s: written as %s, want %s", tc.name, payload, tc.want)
		}
		if _, _, err = unmarshalEpisodeState(payload, 1, now); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		for _, other := range []string{"v3", "v4", "v5"} {
			if other == tc.want {
				continue
			}
			relabelled := strings.Replace(string(payload), "episode-state."+tc.want, "episode-state."+other, 1)
			if _, _, err = unmarshalEpisodeState([]byte(relabelled), 1, now); err == nil {
				t.Errorf("%s: a %s record relabelled %s was accepted", tc.name, tc.want, other)
			}
		}
	}
}

// A stored ADR 0131 item keeps its spelling, so a kimi-k3 Chat episode
// reloaded from the store replays its steer on the tool message it joined.
func TestStoredControlPlacementKeepsTheAppendToToolSpelling(t *testing.T) {
	anchor, previous, inForce := thinkingcontrol.Anchor{Index: 2, Digest: "d"}, "c", "up"
	first := thinkingcontrol.Control{Native: "default", Instruction: &thinkingcontrol.Instruction{}}
	placements := []ControlPlacement{{Key: "k", State: thinkingcontrol.PlacerState{
		Format: thinkingcontrol.FormatChat, First: &first, InForce: &inForce,
		Ledger: []thinkingcontrol.LedgerItem{{
			Anchor: 2, PrefixDigest: "d", Placement: thinkingcontrol.PlacementAppendToTool, Text: "steer",
			Kind: thinkingcontrol.WrittenInstruction,
		}},
		PreviousAnchor: &anchor, PreviousControl: &previous, Calls: 1,
	}}}
	raw, err := json.Marshal(controlPlacementsToWire(placements))
	if err != nil {
		t.Fatal(err)
	}
	var wires []episodeControlWire
	if err = json.Unmarshal(raw, &wires); err != nil {
		t.Fatal(err)
	}
	back, err := controlPlacementsFromWire(wires)
	if err != nil {
		t.Fatal(err)
	}
	if got := back[0].State.Ledger; len(got) != 1 || got[0].Placement != thinkingcontrol.PlacementAppendToTool {
		t.Fatalf("stored spelling came back as %+v", got)
	}
	// A reader that does not know the code refuses the record, never
	// replays the steer somewhere else.
	wires[0].Ledger[0].Placement = "z"
	if _, err := controlPlacementsFromWire(wires); err == nil || !strings.Contains(err.Error(), "malformed") {
		t.Fatalf("an unknown stored placement loaded: %v", err)
	}
}
