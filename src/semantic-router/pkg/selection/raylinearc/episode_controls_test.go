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
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkingcontrol"
)

// A placer stored in the episode resumes the same episode: after a steer and
// a return to none, the stored placer renders the next call exactly as the
// one that never left memory.
func TestEpisodeStateCarriesControlPlacementsUnderV3(t *testing.T) {
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	cell, err := registry.Cell("z-ai/glm-5.3-flash", "openrouter", thinkingcontrol.FormatMessages)
	if err != nil {
		t.Fatal(err)
	}
	control := func(level string) *thinkingcontrol.Control {
		for _, id := range []string{
			"605a61ada934f704248cb99cc84179bf15747302120d11e67ba723db0d8d955b", // default, up
			"17ce2fd14e762777a95db605228cc8f731d2d3e200c4c5d07a735f85fca2eae1", // default, none
		} {
			c, _ := registry.Control(id)
			if c.Instruction != nil && c.Instruction.Level == level {
				return &c
			}
		}
		t.Fatalf("no control at level %s", level)
		return nil
	}
	turns := []string{
		`{"model":"m","messages":[{"role":"user","content":"one"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"one"},{"role":"assistant","content":"a"},{"role":"user","content":"two"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"one"},{"role":"assistant","content":"a"},{"role":"user","content":"two"},{"role":"assistant","content":"b"},{"role":"user","content":"three"}]}`,
	}
	live, _ := thinkingcontrol.NewPlacer(thinkingcontrol.FormatMessages)
	for i, level := range []string{"up", "none"} {
		if _, _, err = live.Render([]byte(turns[i]), control(level), cell, "w"); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	state, err := NewEpisodeState(1)
	if err != nil {
		t.Fatal(err)
	}
	state.Controls = WithControlPlacement(nil, ControlPlacement{Key: "glm|messages|default|-|true", State: live.State()})
	payload, err := marshalEpisodeState(state, 3, now)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(payload), `"schema_version":"rayline.arc.episode-state.v3"`) {
		t.Fatalf("control placements written without v3: %.100s", payload)
	}
	if strings.Count(string(payload), "reason more thoroughly") != 1 {
		t.Fatal("each instruction text is stored once")
	}
	decoded, _, err := unmarshalEpisodeState(payload, 1, now)
	if err != nil {
		t.Fatal(err)
	}
	stored, ok := decoded.ControlPlacementFor("glm|messages|default|-|true")
	if !ok {
		t.Fatal("the placement did not round-trip")
	}
	resumed, err := thinkingcontrol.ResumePlacer(stored)
	if err != nil {
		t.Fatal(err)
	}
	want, wantReceipt, err := live.Render([]byte(turns[2]), control("up"), cell, "w")
	if err != nil {
		t.Fatal(err)
	}
	got, gotReceipt, err := resumed.Render([]byte(turns[2]), control("up"), cell, "w")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) || *gotReceipt.Written != *wantReceipt.Written {
		t.Fatalf("the stored placer renders\n%s\nthe live one\n%s", got, want)
	}

	// A malformed stored placer refuses the record rather than steering by it.
	tampered := bytes.Replace(payload, []byte(`"calls":2`), []byte(`"calls":0`), 1)
	if bytes.Equal(tampered, payload) {
		t.Fatalf("the record no longer holds its call count: %s", payload)
	}
	if _, _, err := unmarshalEpisodeState(tampered, 1, now); err == nil {
		t.Fatal("a record whose placer has a ledger but no calls decoded")
	}
}

func TestControlPlacementsAreBounded(t *testing.T) {
	var placements []ControlPlacement
	for index := 0; index < MaxControlPlacements+3; index++ {
		state := thinkingcontrol.PlacerState{Format: thinkingcontrol.FormatChat, Ledger: []thinkingcontrol.LedgerItem{}}
		placements = WithControlPlacement(placements, ControlPlacement{Key: string(rune('a' + index)), State: state})
	}
	if len(placements) != MaxControlPlacements || placements[0].Key != "d" {
		t.Fatalf("placements not bounded to the most recent: %d, oldest %q", len(placements), placements[0].Key)
	}
}
