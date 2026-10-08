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
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ADR 0129 decision 5, as pathfinder#3996's test_thinking_control.py states
// it: a write after an image tool result needs the cell's task-fidelity
// evidence.

const (
	glmChatNone = "17ce2fd14e762777a95db605228cc8f731d2d3e200c4c5d07a735f85fca2eae1"
	glmChatUp   = "605a61ada934f704248cb99cc84179bf15747302120d11e67ba723db0d8d955b"
	terraNone   = "ed143fc8288b25589866829f3a7f4caa1382220286b0100383f49f87a3722e79"
	terraUp     = "81d08d646c9db5bbb2b10590c16064f63d8b39c7ce25a91744925a78d34855dd"
	tailPNG     = "data:image/png;base64,iVBORw0KGgo="
)

const (
	chatImageRun = `{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"Read","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"shot.png"}]},` +
		`{"role":"user","content":[{"type":"text","text":"[images returned by tool call c1]"},{"type":"image_url","image_url":{"url":"` + tailPNG + `"}}]}`
	chatTextRun = `{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"Read","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":"README.md"}`
	// A Responses codec that hoists the image into a user item after a
	// text-only output (pathfinder #3996 round 2), and a text-only run.
	responsesHoistedImageRun = `{"type":"function_call","call_id":"c1","name":"shell","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"c1","output":"shot.png"},` +
		`{"type":"message","role":"user","content":[{"type":"input_image","image_url":"` + tailPNG + `"}]}`
	responsesTextRun = `{"type":"function_call","call_id":"c1","name":"shell","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"c1","output":"ok"}`
	responsesImageRun = `{"type":"function_call","call_id":"c1","name":"shell","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"c1","output":[{"type":"input_text","text":"shot.png"},{"type":"input_image","image_url":"` + tailPNG + `"}]}`
)

func registryControl(t *testing.T, id string) *Control {
	t.Helper()
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	control, ok := reg.Control(id)
	if !ok {
		t.Fatalf("control %s is not in the registry", prefix12(id))
	}
	return &control
}

func placeBody(t *testing.T, placer *Placer, raw string, control *Control, admitted bool) (string, Receipt) {
	t.Helper()
	body, err := parseJSON([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	placed, receipt, err := placer.place(body, control, admitted)
	if err != nil {
		t.Fatal(err)
	}
	out, err := dump(placed)
	if err != nil {
		t.Fatal(err)
	}
	return string(out), receipt
}

func TestASteerAfterAnImageToolResultIsRefusedWithoutTaskFidelityEvidence(t *testing.T) {
	for _, tc := range []struct {
		name, format, run string
		refuses           bool
	}{
		{"chat image run", FormatChat, chatImageRun, true},
		{"chat image run then user text", FormatChat, chatImageRun + `,{"role":"user","content":"What colour?"}`, true},
		{"chat text run", FormatChat, chatTextRun, false},
		{"responses image run", FormatResponses, responsesImageRun, true},
		{"responses image hoisted into a user item", FormatResponses, responsesHoistedImageRun, true},
		{"responses text run", FormatResponses, responsesTextRun, false},
	} {
		for _, admitted := range []bool{false, true} {
			key, first, none, up := "messages", `{"role":"user","content":"Read it."}`, glmChatNone, glmChatUp
			if tc.format == FormatResponses {
				key, first, none, up = "input", `{"type":"message","role":"user","content":"Read it."}`, terraNone, terraUp
			}
			upControl := registryControl(t, up)
			placer, err := NewPlacer(tc.format)
			if err != nil {
				t.Fatal(err)
			}
			placeBody(t, placer, `{"`+key+`":[`+first+`]}`, registryControl(t, none), admitted)
			body, receipt := placeBody(t, placer, `{"`+key+`":[`+first+`,`+tc.run+`]}`, upControl, admitted)
			refused := tc.refuses && !admitted
			name := tc.name
			if admitted {
				name += ", evidenced cell"
			}
			if got := receipt.Refused != nil && *receipt.Refused == RefusedImageToolTail; got != refused {
				t.Errorf("%s: refused = %v, want %v", name, got, refused)
			}
			wantLevel, wantWritten := "up", WrittenInstruction
			if refused {
				wantLevel, wantWritten = InstructionNone, ""
			}
			if *receipt.LevelInForce != wantLevel || (receipt.Written == nil) != (wantWritten == "") {
				t.Errorf("%s: level_in_force %s written %v, want %s %q", name, *receipt.LevelInForce, receipt.Written, wantLevel, wantWritten)
			}
			raw, _ := json.Marshal(receipt)
			if strings.Contains(string(raw), `"refused"`) != refused {
				t.Errorf("%s: receipt %s; refused appears only when set", name, raw)
			}
			text, _ := json.Marshal(upControl.Instruction.Text)
			steered := strings.Count(body, strings.Trim(string(text), `"`))
			if (steered == 0) != refused || steered > 1 {
				t.Errorf("%s: the steer appears %d times in %s", name, steered, body)
			}
		}
	}
}

func TestARetryOfARefusedSteerRepeatsTheRefusalAndALaterTextTurnWritesIt(t *testing.T) {
	first := `{"role":"user","content":"Read it."}`
	up := registryControl(t, glmChatUp)
	placer, err := NewPlacer(FormatChat)
	if err != nil {
		t.Fatal(err)
	}
	placeBody(t, placer, `{"messages":[`+first+`]}`, registryControl(t, glmChatNone), false)
	refusedBody := `{"messages":[` + first + `,` + chatImageRun + `]}`
	_, receipt := placeBody(t, placer, refusedBody, up, false)
	if receipt.Refused == nil || *receipt.LevelInForce != InstructionNone {
		t.Fatalf("the image tail was not refused: %+v", receipt)
	}
	raw, err := json.Marshal(placer.State())
	if err != nil {
		t.Fatal(err)
	}
	var state PlacerState
	if err = json.Unmarshal(raw, &state); err != nil {
		t.Fatal(err)
	}
	if placer, err = ResumePlacer(state); err != nil {
		t.Fatal(err)
	}
	_, retry := placeBody(t, placer, refusedBody, up, false)
	if !retry.Retry || retry.Refused == nil || *retry.Refused != RefusedImageToolTail ||
		*retry.LevelInForce != InstructionNone || retry.Written != nil {
		t.Fatalf("retry receipt %+v, want the first send's refusal and level none", retry)
	}
	later := `{"messages":[` + first + `,` + chatImageRun + `,{"role":"assistant","content":"Seen."},{"role":"user","content":"Next."}]}`
	_, written := placeBody(t, placer, later, up, false)
	if written.Refused != nil || *written.LevelInForce != "up" || written.Written == nil || *written.Written != WrittenInstruction {
		t.Fatalf("later text turn receipt %+v, want the steer written", written)
	}
}

func TestAStateNamesOnlyTheImageToolTailRefusal(t *testing.T) {
	placer, err := NewPlacer(FormatChat)
	if err != nil {
		t.Fatal(err)
	}
	placeBody(t, placer, `{"messages":[{"role":"user","content":"Read it."}]}`, registryControl(t, glmChatNone), false)
	state := placer.State()
	other := "something_else"
	state.PreviousRefused = &other
	if _, err := ResumePlacer(state); err == nil {
		t.Fatal("a state with an unknown refusal resumed")
	}
	state.PreviousRefused = nil
	if _, err := ResumePlacer(state); err != nil {
		t.Fatalf("a state from before ADR 0129 did not resume: %v", err)
	}
}

// An evidenced row compiles into its cell; anywhere else the field is
// refused, never ignored.
func TestTheRegistryAdmitsImageToolTailEvidenceOnlyOnAChatOrResponsesCell(t *testing.T) {
	for _, tc := range []struct {
		name, format, evidence, basis string
		ok, fold                      bool
	}{
		{"chat", FormatChat, `{"task_fidelity":["doc:docs/history/x.md"]}`, "live", true, false},
		// Pathfinder #3933 compiles fold_ladder onto the same cell beside it.
		{"beside a fold ladder", FormatChat, `{"task_fidelity":["doc:docs/history/x.md"]}`, "live", true, true},
		{"messages", FormatMessages, `{"task_fidelity":["doc:docs/history/x.md"]}`, "live", false, false},
		{"empty", FormatChat, `{"task_fidelity":[]}`, "live", false, false},
		{"another key", FormatChat, `{"verifier":["x"]}`, "live", false, false},
		// ADR 0129 task fidelity is a live verifier result (pathfinder #3996).
		{"replay row", FormatChat, `{"task_fidelity":["doc:docs/history/x.md"]}`, "replay", false, false},
		{"no basis", FormatChat, `{"task_fidelity":["doc:docs/history/x.md"]}`, "", false, false},
	} {
		parsed, err := parseJSON(embeddedArtifact)
		if err != nil {
			t.Fatal(err)
		}
		evidence, err := parseJSON([]byte(tc.evidence))
		if err != nil {
			t.Fatal(err)
		}
		var target *value
		for _, cell := range parsed.get("cells").items {
			if cell.get("model").str == "z-ai/glm-5.3-flash" && cell.get("provider").str == "openrouter" &&
				cell.get("format").str == tc.format {
				target = cell
			}
		}
		if target == nil {
			t.Fatalf("%s: no glm-5.3-flash openrouter cell", tc.name)
		}
		target.set("image_tool_tail", evidence)
		if tc.fold {
			fold, foldErr := parseJSON([]byte(`{"ladder":"steer3-periodic-fold-v1","lever_separation":"not_demonstrated"}`))
			if foldErr != nil {
				t.Fatal(foldErr)
			}
			target.set("fold_ladder", fold)
		}
		if tc.basis == "" {
			target.set("basis", nullValue())
		} else {
			target.set("basis", stringValue(tc.basis))
		}
		tampered, err := jcs(parsed)
		if err != nil {
			t.Fatal(err)
		}
		reg, err := Load(tampered)
		if !tc.ok {
			if err == nil || !strings.Contains(err.Error(), "image_tool_tail") {
				t.Errorf("%s: loaded: %v", tc.name, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		cell, err := reg.Admit("z-ai/glm-5.3-flash", "openrouter", FormatChat, *registryControl(t, glmChatUp), true)
		if err != nil || !cell.ImageToolTailAdmitted {
			t.Errorf("%s: cell admitted %v, err %v", tc.name, cell != nil && cell.ImageToolTailAdmitted, err)
		}
	}
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	cell, err := reg.Admit("z-ai/glm-5.3-flash", "openrouter", FormatChat, *registryControl(t, glmChatUp), true)
	if err != nil || cell.ImageToolTailAdmitted {
		t.Fatalf("the embedded registry states no evidence; cell admitted %v, err %v", cell != nil && cell.ImageToolTailAdmitted, err)
	}
}

// Every image tool-tail golden that records the refusal has its evidenced
// twin in the placement corpus, so the spelling keeps its byte coverage.
func TestEveryRefusedImageToolTailGoldenHasItsEvidencedTwin(t *testing.T) {
	twins := 0
	for _, format := range goldenFormats {
		dir := filepath.Join("testdata", "golden", format)
		for _, name := range caseNames(t, dir) {
			c := readCase(t, filepath.Join(dir, name))
			refused := false
			for _, call := range c.Calls {
				refused = refused || strings.Contains(string(call.Receipt), `"refused"`)
			}
			if !refused {
				continue
			}
			twin := filepath.Join("testdata", "placement", format, name)
			if _, err := os.Stat(filepath.Join(twin, "case.json")); err != nil || readCase(t, twin).Admission != imageToolTailAdmitted {
				t.Errorf("golden %s/%s refuses an image tool tail and has no %s twin", format, name, imageToolTailAdmitted)
			}
			twins++
		}
	}
	if twins != 8 {
		t.Errorf("%d refused image tool-tail goldens, want 8 (four cases on chat and responses)", twins)
	}
}

// A refused call keeps the drawn control in its receipt: the label is the
// policy's action, and what the provider saw is the level in force beside
// the refusal (pathfinder ADR 0129). A byte-identical retry repeats both.
func TestARefusedCallKeepsTheDrawnControl(t *testing.T) {
	placer, err := NewPlacer(FormatChat)
	if err != nil {
		t.Fatal(err)
	}
	first := `{"role":"user","content":"Read it."}`
	placeBody(t, placer, `{"messages":[`+first+`]}`, registryControl(t, glmChatNone), false)
	refusedBody := `{"messages":[` + first + `,` + chatImageRun + `]}`
	_, receipt := placeBody(t, placer, refusedBody, registryControl(t, glmChatUp), false)
	if receipt.Refused == nil || receipt.ControlID != glmChatUp || *receipt.LevelInForce != InstructionNone || receipt.Written != nil {
		t.Fatalf("refused receipt %+v, want the drawn control %s at level none, nothing written", receipt, prefix12(glmChatUp))
	}
	_, retry := placeBody(t, placer, refusedBody, registryControl(t, glmChatUp), false)
	if !retry.Retry || retry.Refused == nil || retry.ControlID != glmChatUp {
		t.Fatalf("retry receipt %+v, want the drawn control and the refusal repeated", retry)
	}
}
