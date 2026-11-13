package thinkingcontrol

import (
	"strings"
	"testing"
)

// placeMessages places one Messages body of the given user turns (each
// followed by a reply, except the last) and returns the rendered messages.
func placeMessages(t *testing.T, placer *Placer, control *Control, turns int) (string, Receipt) {
	t.Helper()
	var units []string
	for turn := 0; turn < turns; turn++ {
		if turn > 0 {
			units = append(units, `{"role":"assistant","content":"reply"}`)
		}
		units = append(units, `{"role":"user","content":"step `+string(rune('0'+turn))+`"}`)
	}
	body, err := parseJSON([]byte(`{"messages":[` + strings.Join(units, ",") + `]}`))
	if err != nil {
		t.Fatal(err)
	}
	placed, receipt, err := placer.place(body, control, tailRules{})
	if err != nil {
		t.Fatal(err)
	}
	out, err := dump(placed.get("messages"))
	if err != nil {
		t.Fatal(err)
	}
	return string(out), receipt
}

// One transcript keeps one placer whatever control each call draws: a
// native-only call, or one at another base, replays the items earlier calls
// wrote, so the transcript the provider caches only grows; and a retry the
// policy decided again with another control resends the same bytes.
func TestAPlacerReplaysItsItemsUnderEveryControl(t *testing.T) {
	steer := &Control{Native: "high", Instruction: &Instruction{Level: "up", Text: "think harder", NeutralText: "as usual"}}
	native := &Control{Native: "default"}
	placer, err := NewPlacer(FormatMessages)
	if err != nil {
		t.Fatal(err)
	}
	first, _ := placeMessages(t, placer, steer, 1)
	if !strings.Contains(first, "think harder") {
		t.Fatalf("the steer was not written: %s", first)
	}
	second, receipt := placeMessages(t, placer, native, 2)
	if !strings.HasPrefix(second, strings.TrimSuffix(first, "]")) {
		t.Fatalf("a native-only call dropped the steer from the cached prefix:\n was %s\n now %s", first, second)
	}
	if receipt.LevelInForce == nil || *receipt.LevelInForce != "up" || receipt.Written != nil {
		t.Fatalf("receipt = %+v, want the replayed steer in force and nothing written", receipt)
	}
	retried, receipt := placeMessages(t, placer, steer, 2)
	if retried != second || !receipt.Retry || receipt.Written != nil {
		t.Fatalf("a re-decided retry changed the bytes (retry %v):\n was %s\n now %s", receipt.Retry, second, retried)
	}
}
