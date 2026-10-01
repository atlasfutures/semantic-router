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
}
