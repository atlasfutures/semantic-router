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

package thinkingcontrol

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRegistryRefusesAnArtifactThatIsNotCanonical(t *testing.T) {
	var indented bytes.Buffer
	if err := json.Indent(&indented, embeddedArtifact, "", " "); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(indented.Bytes()); err == nil || !strings.Contains(err.Error(), "canonical JCS") {
		t.Fatalf("an indented artifact loaded: %v", err)
	}
}

func TestRegistryRefusesAControlThatDoesNotHashToItsID(t *testing.T) {
	// Change one control's text in place: the bytes stay canonical, the id
	// no longer recomputes.
	tampered := bytes.Replace(embeddedArtifact, []byte("use your normal judgement."), []byte("use your normal judgement!"), 1)
	if bytes.Equal(tampered, embeddedArtifact) {
		t.Fatal("the fixture text moved; pick another")
	}
	if _, err := Load(tampered); err == nil || !strings.Contains(err.Error(), "does not hash to its id") {
		t.Fatalf("a tampered control loaded: %v", err)
	}
}

// An admission level other than certified, experimental or refused would
// slip past the experimental opt-in, so the artifact is refused.
func TestRegistryRefusesAnUnknownAdmissionLevel(t *testing.T) {
	tampered := bytes.Replace(embeddedArtifact, []byte(`"instruction":"experimental"`), []byte(`"instruction":"experimenta1"`), 1)
	if bytes.Equal(tampered, embeddedArtifact) {
		t.Fatal("the artifact holds no experimental cell; pick another")
	}
	if _, err := Load(tampered); err == nil || !strings.Contains(err.Error(), "is not certified, experimental or refused") {
		t.Fatalf("an unknown admission level loaded: %v", err)
	}
}

// A repeated (model, provider, format) cell would let the later one decide
// admission, so the artifact is refused.
func TestRegistryRefusesARepeatedCell(t *testing.T) {
	parsed, err := parseJSON(embeddedArtifact)
	if err != nil {
		t.Fatal(err)
	}
	cells := parsed.get("cells")
	cells.items = append(cells.items, cells.items[0].clone())
	repeated, err := jcs(parsed)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Load(repeated); err == nil || !strings.Contains(err.Error(), "appears twice") {
		t.Fatalf("a repeated cell loaded: %v", err)
	}
}

func TestAdmissionIsPerModelProviderAndFormat(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	// An instruction control admitted on openrouter x messages for Kimi.
	cell, err := reg.Cell("moonshotai/kimi-k3", "openrouter", FormatMessages)
	if err != nil {
		t.Fatal(err)
	}
	var steer *Control
	for id := range cell.controls {
		control, _ := reg.Control(id)
		if control.Instruction != nil && control.Instruction.Level == "up" {
			steer = &control
			break
		}
	}
	if steer == nil {
		t.Fatal("no steered control on the Kimi messages cell")
	}
	if _, err := reg.Admit("moonshotai/kimi-k3", "openrouter", FormatMessages, *steer, true); err != nil {
		t.Fatalf("admitted cell refused: %v", err)
	}
	if _, err := reg.Admit("moonshotai/kimi-k3", "openrouter", FormatMessages, *steer, false); err == nil ||
		!strings.Contains(err.Error(), "experimental only") {
		t.Fatalf("experimental instruction admitted without allow_experimental: %v", err)
	}
	if _, err := reg.Admit("moonshotai/kimi-k3", "openrouter", FormatResponses, *steer, true); err == nil ||
		!strings.Contains(err.Error(), "is not admitted") {
		t.Fatalf("an instruction admitted on a refused cell: %v", err)
	}
	if _, err := reg.Admit("vendor/unknown", "openrouter", FormatMessages, *steer, true); err == nil ||
		!strings.Contains(err.Error(), "no admission cell") {
		t.Fatalf("an unknown model admitted: %v", err)
	}
}

func TestControlIDIsTheJCSDigestOfTheControlObject(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for id, control := range reg.controls {
		canonical, err := jcs(control.json())
		if err != nil {
			t.Fatal(err)
		}
		got, err := ControlIDOf(canonical)
		if err != nil || got != id {
			t.Fatalf("ControlIDOf = %s, %v; want %s", got, err, id)
		}
		parsed, err := ParseControl(canonical)
		if err != nil || parsed.ID() != id {
			t.Fatalf("ParseControl(%s) = %v", canonical, err)
		}
	}
	for _, bad := range []string{
		`{"base":{"native":"high"},"budget_tokens":null}`,
		`{"base":{"native":"high"},"budget_tokens":0,"instruction":null}`,
		`{"base":{"native":"high"},"budget_tokens":null,"instruction":null,"extra":1}`,
		`{"base":{"native":"high"},"budget_tokens":null,"instruction":{"level":"up","text":"x","neutral_text":"y","rules":{"placement":"turn_tail_v1","emit":"on_change_v1","replay":"ledger_v1"}}}`,
	} {
		if _, err := ParseControl([]byte(bad)); err == nil {
			t.Errorf("ParseControl accepted %s", bad)
		}
	}
}

func TestPythonFloatRepr(t *testing.T) {
	for in, want := range map[float64]string{
		1: "1.0", 0.5: "0.5", 100000: "100000.0", 1e16: "1e+16", 1e-5: "1e-05", 1.5e-7: "1.5e-07",
		0.0001: "0.0001", 123456789012345678: "1.2345678901234568e+17", -2.25: "-2.25", 1e15: "1000000000000000.0",
	} {
		if got, err := pythonFloatRepr(in); err != nil || got != want {
			t.Errorf("repr(%v) = %q, want %q", in, got, want)
		}
	}
}

// A resumed placer continues the episode: the hold replays the steer and
// writes nothing.
func TestPlacerStateResumesTheEpisode(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	cell, err := reg.Cell("moonshotai/kimi-k3", "openrouter", FormatMessages)
	if err != nil {
		t.Fatal(err)
	}
	var steer *Control
	for id := range cell.controls {
		control, _ := reg.Control(id)
		if control.Instruction != nil && control.Instruction.Level == "up" && control.Native == "default" {
			steer = &control
		}
	}
	first := []byte(`{"model":"m","messages":[{"role":"user","content":"Start."}]}`)
	second := []byte(`{"model":"m","messages":[{"role":"user","content":"Start."},{"role":"assistant","content":[{"type":"text","text":"ok"}]},{"role":"user","content":"Next."}]}`)
	placer, _ := NewPlacer(FormatMessages)
	if _, _, err := placer.Render(first, steer, cell, "w"); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(placer.State())
	if err != nil {
		t.Fatal(err)
	}
	var state PlacerState
	if err := json.Unmarshal(encoded, &state); err != nil {
		t.Fatal(err)
	}
	resumed, err := ResumePlacer(state)
	if err != nil {
		t.Fatal(err)
	}
	body, receipt, err := resumed.Render(second, steer, cell, "w")
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Written != nil || receipt.InstructionState == nil || *receipt.InstructionState != StateSteered {
		t.Fatalf("receipt = %+v", receipt)
	}
	if !bytes.Contains(body, []byte(`{"role":"user","content":[{"type":"text","text":"Start."},{"type":"text","text":"Until the next steering instruction, reason more thoroughly`)) {
		t.Fatalf("the steer was not replayed at its anchor: %s", body)
	}

	// A state with calls but no first control, or a fresh state that already
	// holds a ledger, would let the next call re-found the episode.
	headless := resumed.State()
	headless.First = nil
	if _, err := ResumePlacer(headless); err == nil {
		t.Fatal("a state with calls and no first control resumed")
	}
	fresh := resumed.State()
	fresh.Calls, fresh.First, fresh.PreviousAnchor, fresh.PreviousControl, fresh.InForce = 0, nil, nil, nil, nil
	if _, err := ResumePlacer(fresh); err == nil {
		t.Fatal("a state with no calls and a ledger resumed")
	}
	nativeOnly := resumed.State()
	native := *nativeOnly.First
	native.Instruction = nil
	nativeOnly.First, nativeOnly.InForce = &native, nil
	if _, err := ResumePlacer(nativeOnly); err == nil {
		t.Fatal("a native-only episode with a ledger resumed")
	}
	// A level in force must be the last written instruction.
	unbacked := resumed.State()
	unbacked.Ledger = nil
	if _, err := ResumePlacer(unbacked); err == nil {
		t.Fatal("a level in force with no instruction in the ledger resumed")
	}
	cleared := resumed.State()
	cleared.InForce = nil
	if _, err := ResumePlacer(cleared); err == nil {
		t.Fatal("a written instruction with no level in force resumed")
	}
	// [instruction@0, neutral_marker@2] stored reversed: the instruction is
	// last in the slice, so it would read as in force; the order is refused.
	reversed := resumed.State()
	item := reversed.Ledger[0]
	reversed.Ledger = []LedgerItem{
		{Anchor: 2, PrefixDigest: item.PrefixDigest, Placement: item.Placement, Text: "neutral", Kind: WrittenNeutralMarker},
		{Anchor: 0, PrefixDigest: item.PrefixDigest, Placement: item.Placement, Text: item.Text, Kind: WrittenInstruction},
	}
	if _, err := ResumePlacer(reversed); err == nil {
		t.Fatal("a ledger with decreasing anchors resumed")
	}
}

// A cell whose instruction admission is refused lists native controls only.
func TestRegistryRefusesAnInstructionOnARefusedCell(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parseJSON(embeddedArtifact)
	if err != nil {
		t.Fatal(err)
	}
	var steered string
	for _, raw := range parsed.get("cells").items {
		if raw.get("instruction").str != AdmissionRefused {
			for _, id := range raw.get("controls").items {
				if c, _ := reg.Control(id.str); c.Instruction != nil {
					steered = id.str
				}
			}
		}
	}
	for _, raw := range parsed.get("cells").items {
		if raw.get("instruction").str == AdmissionRefused {
			ids := raw.get("controls")
			ids.items = append(ids.items, stringValue(steered))
			break
		}
	}
	tampered, err := jcs(parsed)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Load(tampered); err == nil || !strings.Contains(err.Error(), "refuses instructions") {
		t.Fatalf("an instruction on a refused cell loaded: %v", err)
	}
}

// pathfinder ADR 0114: a configuration_update_v1 control is read, so the
// registry loads and its id is the registry's, and is refused at admission,
// because VSR does not implement the unit until it serves a Responses cell.
func TestAConfigurationUpdateControlLoadsAndIsNotAdmitted(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	var unitControl *Control
	var unitID string
	for id, control := range reg.controls {
		if control.Instruction != nil && control.Instruction.Unit == UnitConfigurationUpdate {
			c := control
			unitControl, unitID = &c, id
			break
		}
	}
	if unitControl == nil {
		t.Fatal("the embedded registry has no configuration_update control")
	}
	if got := unitControl.ID(); got != unitID {
		t.Fatalf("the unit is not part of the control's identity: id %s, registry key %s", got, unitID)
	}
	for key, cell := range reg.cells {
		if !cell.controls[unitID] {
			continue
		}
		_, err := reg.Admit(key.model, key.provider, key.format, *unitControl, true)
		if err == nil || !strings.Contains(err.Error(), "does not implement") {
			t.Fatalf("a configuration_update control was admitted on %v: %v", key, err)
		}
		return
	}
	t.Fatal("no cell lists the configuration_update control")
}

// The registry at 50ea71ba admits steering for two new models on OpenRouter
// Messages, experimentally, and nowhere else: a worker serving either is
// admitted only with allow_experimental, and its steer renders on the
// request's tail user message.
func TestNewModelsAdmitSteeringOnlyOnOpenRouterMessages(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, model := range []string{"mistralai/mistral-large-4-0", "meta/muse-spark-1.3-contributor"} {
		t.Run(model, func(t *testing.T) {
			cell, err := reg.Cell(model, "openrouter", FormatMessages)
			if err != nil {
				t.Fatal(err)
			}
			var steer *Control
			for id := range cell.controls {
				control, _ := reg.Control(id)
				if control.Instruction != nil && control.Instruction.Level == "up" {
					steer = &control
					break
				}
			}
			if steer == nil {
				t.Fatal("no steered control on the OpenRouter Messages cell")
			}
			if _, gated := reg.Admit(model, "openrouter", FormatMessages, *steer, false); gated == nil ||
				!strings.Contains(gated.Error(), "experimental only") {
				t.Fatalf("an experimental steer admitted without allow_experimental: %v", gated)
			}
			for _, refused := range []struct{ provider, format string }{
				{"openrouter", FormatChat}, {"openrouter", FormatResponses}, {"anthropic", FormatMessages},
			} {
				if _, admitted := reg.Admit(model, refused.provider, refused.format, *steer, true); admitted == nil {
					t.Fatalf("a steer admitted on %s x %s", refused.provider, refused.format)
				}
			}
			client := []byte(`{"model":"auto","max_tokens":256,"messages":[{"role":"user","content":"fix the failing test"}]}`)
			bodies, receipts, err := RenderAdmitted(reg, [][]byte{client}, []*Control{steer},
				model, "openrouter", FormatMessages, model, true)
			if err != nil {
				t.Fatal(err)
			}
			if receipts[0].Written == nil || *receipts[0].Written != WrittenInstruction {
				t.Fatalf("receipt = %+v, want the instruction written", receipts[0])
			}
			if !strings.Contains(string(bodies[0]), `"model":"`+model+`"`) ||
				!strings.Contains(string(bodies[0]), steer.Instruction.Text) {
				t.Fatalf("the steer did not render on the tail:\n%s", bodies[0])
			}
		})
	}
}

// pathfinder FOLD_RULES (inband_system fold_before_unit_v1): the fold
// controls load under pathfinder's ids, are admitted on Mistral Large 4's
// OpenRouter Messages cell beside its steer3 controls, which still admit,
// and never on a cell that does not list them.
func TestFoldControlsLoadUnderPathfindersIDsAndAdmitWhereListed(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	var fold, steer []string
	for id, control := range reg.controls {
		if control.Instruction == nil {
			continue
		}
		if control.Instruction.InbandSystem == InbandSystemFold {
			if got := control.ID(); got != id {
				t.Fatalf("the fold rule is not part of the control's identity: id %s, registry key %s", got, id)
			}
			fold = append(fold, id)
		}
	}
	for _, prefix := range []string{"65832e190480", "a75fcc5be6ba", "5a811ebd822e"} {
		found := false
		for _, id := range fold {
			found = found || strings.HasPrefix(id, prefix)
		}
		if !found {
			t.Fatalf("fold control %s is not in the registry (fold ids %v)", prefix, fold)
		}
	}
	const model, provider = "mistralai/mistral-large-4-0", "openrouter"
	cell, err := reg.Cell(model, provider, FormatMessages)
	if err != nil {
		t.Fatal(err)
	}
	for id := range cell.controls {
		control, _ := reg.Control(id)
		if control.Instruction != nil && control.Instruction.InbandSystem == "" {
			steer = append(steer, id)
		}
	}
	if len(steer) == 0 {
		t.Fatal("Mistral's Messages cell lost its steer3 controls")
	}
	for _, id := range append(append([]string{}, fold...), steer...) {
		control, _ := reg.Control(id)
		if _, err := reg.Admit(model, provider, FormatMessages, control, true); err != nil {
			t.Fatalf("control %s refused on Mistral's Messages cell: %v", id[:12], err)
		}
	}
	for _, id := range fold {
		control, _ := reg.Control(id)
		if _, err := reg.Admit(model, provider, FormatChat, control, true); err == nil {
			t.Fatalf("fold control %s admitted on a cell that does not list it", id[:12])
		}
	}
}

// A rule set is fixed per episode: a fold control after a steer3 one (or the
// reverse) is refused, as pathfinder refuses it.
func TestARuleSetSwitchMidEpisodeIsRefused(t *testing.T) {
	steer := Control{Native: "default", Instruction: &Instruction{Level: "up", Text: "t", NeutralText: "n"}}
	fold := steer
	foldInstruction := *steer.Instruction
	foldInstruction.InbandSystem = InbandSystemFold
	fold.Instruction = &foldInstruction
	placer, err := NewPlacer(FormatMessages)
	if err != nil {
		t.Fatal(err)
	}
	if err = placer.check(&steer); err != nil {
		t.Fatal(err)
	}
	placer.state.First = &steer
	err = placer.check(&fold)
	if err == nil || !strings.Contains(err.Error(), "base, budget, lever presence and rules are fixed per episode") {
		t.Fatalf("a mid-episode rule switch = %v, want the per-episode refusal", err)
	}
}
