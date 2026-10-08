package raylinearc

import (
	"encoding/json"
	"testing"

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
