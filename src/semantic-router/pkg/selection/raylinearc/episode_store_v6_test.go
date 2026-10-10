package raylinearc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkingcontrol"
)

// A record past the 16 placers or upstream records a v5 reader keeps is v6,
// whatever else it carries; every record within them keeps its v3, v4 or v5
// bytes, so a router that predates v6 still reads it, and refuses a v6 one
// by its schema rather than by a count it does not allow.
func TestEpisodeStateWireWritesV6OnlyPastTheLegacyBounds(t *testing.T) {
	now := time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC)
	anchor, previous, refused := thinkingcontrol.Anchor{Index: 0, Digest: "d"}, "c", thinkingcontrol.RefusedImageToolTail
	placements := func(count int, refusal *string) []ControlPlacement {
		var out []ControlPlacement
		for index := 0; index < count; index++ {
			first := thinkingcontrol.Control{Native: "default", Instruction: &thinkingcontrol.Instruction{}}
			state := thinkingcontrol.PlacerState{
				Format: thinkingcontrol.FormatChat, First: &first, Ledger: []thinkingcontrol.LedgerItem{},
				PreviousAnchor: &anchor, PreviousControl: &previous, Calls: 1,
			}
			if index == 0 {
				state.PreviousRefused = refusal
			}
			out = WithControlPlacement(out, ControlPlacement{Key: fmt.Sprintf("w%d|chat|default|-|true", index), State: state})
		}
		return out
	}
	upstream := func(count int) []UpstreamPrefix {
		var out []UpstreamPrefix
		for index := 0; index < count; index++ {
			out = WithUpstreamPrefix(out, UpstreamPrefix{Worker: fmt.Sprintf("w%d", index), Messages: index, Digest: strings.Repeat("0", 32)})
		}
		return out
	}
	task := []json.RawMessage{json.RawMessage(`{"role":"user","content":"fix the bug"}`)}
	type record struct {
		controls, upstream int
		refusal, exclusion bool
	}
	want := func(r record) string {
		switch {
		case r.controls > legacyMaxControlPlacements || r.upstream > legacyMaxUpstreamPrefixes:
			return "v6"
		case r.refusal:
			return "v5"
		case r.exclusion:
			return "v4"
		}
		return "v3"
	}
	for _, controls := range []int{1, legacyMaxControlPlacements, legacyMaxControlPlacements + 1, MaxControlPlacements} {
		for _, prefixes := range []int{0, legacyMaxUpstreamPrefixes, legacyMaxUpstreamPrefixes + 1, MaxUpstreamPrefixes} {
			for _, refusal := range []bool{false, true} {
				for _, exclusion := range []bool{false, true} {
					r := record{controls, prefixes, refusal, exclusion}
					name := fmt.Sprintf("%+v", r)
					state, err := NewEpisodeState(1)
					if err != nil {
						t.Fatal(err)
					}
					state.Policy = (&PolicyEpisodeState{}).Next(task, strings.Repeat("a", 64), "arm-a")
					if exclusion {
						state.Policy = state.Policy.WithExclusion("vendor/off", "refusal")
					}
					var refusalText *string
					if refusal {
						refusalText = &refused
					}
					state.Controls = placements(controls, refusalText)
					state.Upstream = upstream(prefixes)
					payload, err := marshalEpisodeState(state, 1, now)
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					schema := want(r)
					if !strings.Contains(string(payload), `"schema_version":"rayline.arc.episode-state.`+schema+`"`) {
						t.Fatalf("%s: written as %.80s, want %s", name, payload, schema)
					}
					decoded, _, err := unmarshalEpisodeState(payload, 1, now)
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					if len(decoded.Controls) != controls || len(decoded.Upstream) != prefixes {
						t.Fatalf("%s: round-tripped %d placers and %d records", name, len(decoded.Controls), len(decoded.Upstream))
					}
					for _, other := range []string{"v3", "v4", "v5", "v6"} {
						if other == schema {
							continue
						}
						relabelled := strings.Replace(string(payload), "episode-state."+schema, "episode-state."+other, 1)
						if _, _, err = unmarshalEpisodeState([]byte(relabelled), 1, now); err == nil {
							t.Errorf("%s: a %s record relabelled %s was accepted", name, schema, other)
						}
					}
				}
			}
		}
	}
}

// Past MaxControlPlacements or MaxUpstreamPrefixes a stored record is refused,
// v6 or not.
func TestEpisodeStateWireRefusesARecordPastTheBounds(t *testing.T) {
	now := time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < MaxControlPlacements; index++ {
		state.Controls = append(state.Controls, ControlPlacement{
			Key:   fmt.Sprintf("w%d|chat|default|-|false", index),
			State: thinkingcontrol.PlacerState{Format: thinkingcontrol.FormatChat, Ledger: []thinkingcontrol.LedgerItem{}},
		})
	}
	payload, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	extra := `{"key":"extra","format":"chat","texts":[],"ledger":[],"epoch":0,"calls":0}`
	over := strings.Replace(string(payload), `"controls":[`, `"controls":[`+extra+`,`, 1)
	if _, _, err = unmarshalEpisodeState([]byte(over), 1, now); err == nil {
		t.Fatal("a record holding more than MaxControlPlacements placers decoded")
	}
	state.Controls = append(state.Controls, ControlPlacement{
		Key:   "extra",
		State: thinkingcontrol.PlacerState{Format: thinkingcontrol.FormatChat, Ledger: []thinkingcontrol.LedgerItem{}},
	})
	if _, err = marshalEpisodeState(state, 1, now); err == nil {
		t.Fatal("a state holding more than MaxControlPlacements placers was written")
	}
}

// A record at MaxControlPlacements placers, each with a rendered ledger,
// outgrows a v5 record's 64 KiB; v6 stores it, and a record that size under
// an older schema is refused.
func TestEpisodeStateWireStoresFullPlacersUnderV6(t *testing.T) {
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	cell, err := registry.Cell("z-ai/glm-5.3-flash", "openrouter", thinkingcontrol.FormatMessages)
	if err != nil {
		t.Fatal(err)
	}
	controls := map[string]*thinkingcontrol.Control{}
	for _, id := range []string{
		"605a61ada934f704248cb99cc84179bf15747302120d11e67ba723db0d8d955b", // default, up
		"17ce2fd14e762777a95db605228cc8f731d2d3e200c4c5d07a735f85fca2eae1", // default, none
	} {
		c, _ := registry.Control(id)
		controls[c.Instruction.Level] = &c
	}
	live, _ := thinkingcontrol.NewPlacer(thinkingcontrol.FormatMessages)
	messages := `{"role":"user","content":"task"}`
	for turn := 0; turn < 12; turn++ {
		level := "up"
		if turn%2 == 1 {
			level = "none"
		}
		body := fmt.Sprintf(`{"model":"m","messages":[%s]}`, messages)
		if _, _, err = live.Render([]byte(body), controls[level], cell, "w"); err != nil {
			t.Fatal(err)
		}
		messages += fmt.Sprintf(`,{"role":"assistant","content":"a%d"},{"role":"user","content":"u%d"}`, turn, turn)
	}
	now := time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < MaxControlPlacements; index++ {
		state.Controls = WithControlPlacement(state.Controls, ControlPlacement{
			Key: fmt.Sprintf("w%d|messages|default|-|true", index), State: live.State(),
		})
	}
	payload, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if len(payload) <= maxEpisodeStateBytes {
		t.Fatalf("a full record takes %d bytes, within a v5 record's limit; the test no longer exercises v6's", len(payload))
	}
	if !strings.Contains(string(payload), `"schema_version":"rayline.arc.episode-state.v6"`) {
		t.Fatalf("written as %.80s, want v6", payload)
	}
	decoded, _, err := unmarshalEpisodeState(payload, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded.Controls) != MaxControlPlacements {
		t.Fatalf("round-tripped %d placers", len(decoded.Controls))
	}

	// Within the legacy bound, a record past 64 KiB is still refused.
	state.Controls = state.Controls[:legacyMaxControlPlacements]
	small, err := marshalEpisodeState(state, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	padded := strings.Replace(string(small), `"turn_index":0`, `"turn_index":0`+strings.Repeat(" ", maxEpisodeStateBytes), 1)
	if _, _, err = unmarshalEpisodeState([]byte(padded), 1, now); err == nil || !strings.Contains(err.Error(), "size limit") {
		t.Fatalf("a v3 record past 64 KiB decoded: %v", err)
	}
}
