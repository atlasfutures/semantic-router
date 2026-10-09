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

package thinkinglever

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc/thinkingcontrol"
)

type goldenCall struct {
	ClientBody   string `json:"client_body"`
	ControlID    string `json:"control_id"`
	ExpectedBody string `json:"expected_body"`
	Receipt      struct {
		LevelInForce string  `json:"level_in_force"`
		Refused      *string `json:"refused"`
	} `json:"receipt"`
}

type goldenCase struct {
	Model    string       `json:"model"`
	Provider string       `json:"provider"`
	Format   string       `json:"format"`
	Calls    []goldenCall `json:"calls"`
}

// leverBindingFor is a steering-suffix binding whose levels write the
// golden's controls' instruction text, as a binding compiled from the
// registry export does.
func leverBindingFor(t *testing.T, registry *thinkingcontrol.Registry, calls []goldenCall) Binding {
	t.Helper()
	binding := Binding{
		Lever: LeverSteeringSuffix, Emit: EmitOnChange, Neutral: "none",
		Placements: []Placement{PlaceAppendTailUserText, PlaceUserAfterToolRun},
		Levels:     []Level{{Name: "none", Rank: 0}},
	}
	for _, call := range calls {
		control, ok := registry.Control(call.ControlID)
		if !ok {
			t.Fatalf("control %s is not in the embedded registry", call.ControlID)
		}
		if control.Instruction == nil || control.Instruction.Level == thinkingcontrol.InstructionNone {
			continue
		}
		binding.Levels = append(binding.Levels, Level{Name: control.Instruction.Level, Rank: 1, Suffix: control.Instruction.Text})
	}
	return binding
}

func goldenMessages(t *testing.T, raw []byte) []json.RawMessage {
	t.Helper()
	var body struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	return body.Messages
}

// ADR 0129 decision 5 through the thinking lever, as the policy-mode cell
// serves it: a steer after an image tool result is written only where the
// worker's admission cell states task-fidelity evidence, and then as the
// v5 placer writes it, a text part of the hoisted image message. The
// pathfinder goldens are the parity target: the evidenced kimi-k3 Chat cell
// steers, and an unevidenced cell sends the client's messages unchanged.
func TestTheLeverSteersAnImageToolTailOnlyOnAnEvidencedCell(t *testing.T) {
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"image_tool_tail_evidenced_cell", "image_tool_tail"} {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join("..", "thinkingcontrol", "testdata", "golden", "chat", name)
			raw, err := os.ReadFile(filepath.Join(dir, "case.json"))
			if err != nil {
				t.Fatal(err)
			}
			var golden goldenCase
			if err = json.Unmarshal(raw, &golden); err != nil {
				t.Fatal(err)
			}
			cell, err := registry.Cell(golden.Model, golden.Provider, golden.Format)
			if err != nil {
				t.Fatal(err)
			}
			binding := leverBindingFor(t, registry, golden.Calls)
			var ledger *Ledger
			var turnIndex uint64
			for index, call := range golden.Calls {
				client, err := os.ReadFile(filepath.Join(dir, call.ClientBody))
				if err != nil {
					t.Fatal(err)
				}
				expected, err := os.ReadFile(filepath.Join(dir, call.ExpectedBody))
				if err != nil {
					t.Fatal(err)
				}
				request, _, _, err := (protocolcodec.OpenAIChatCodec{}).DecodeRequest(client, llmprotocol.DefaultPolicy())
				if err != nil {
					t.Fatal(err)
				}
				control, _ := registry.Control(call.ControlID)
				requested := "none"
				if control.Instruction != nil {
					requested = control.Instruction.Level
				}
				imageToolTail := ImageToolTail(request, llmprotocol.OpenAIChatV1)
				plan, err := PlanTurn(Turn{
					Binding: binding, Ledger: ledger, Messages: Messages(request.Messages), TurnIndex: turnIndex,
					Requested: requested, ImageToolTail: imageToolTail,
					ImageToolTailAdmitted: imageToolTail && cell.ImageToolTailAdmitted,
				})
				if err != nil {
					t.Fatal(err)
				}
				wantRefused := ""
				if call.Receipt.Refused != nil {
					wantRefused = *call.Receipt.Refused
				}
				if plan.Refused != wantRefused || plan.LevelInForce != call.Receipt.LevelInForce {
					t.Fatalf("call %d: refused %q level %q, want refused %q level %q",
						index, plan.Refused, plan.LevelInForce, wantRefused, call.Receipt.LevelInForce)
				}
				request.Messages, err = ApplyLedger(request.Messages, binding.Lever, plan.Next)
				if err != nil {
					t.Fatal(err)
				}
				body, _, err := (protocolcodec.OpenAIChatCodec{}).EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
				if err != nil {
					t.Fatal(err)
				}
				sent := goldenMessages(t, body)
				// The golden's bytes pass through the same codec, so only the
				// lever's write can differ, not the codec's spelling.
				golden, _, _, err := (protocolcodec.OpenAIChatCodec{}).DecodeRequest(expected, llmprotocol.DefaultPolicy())
				if err != nil {
					t.Fatal(err)
				}
				goldenBody, _, err := (protocolcodec.OpenAIChatCodec{}).EncodeRequest(golden, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
				if err != nil {
					t.Fatal(err)
				}
				want := goldenMessages(t, goldenBody)
				if len(sent) != len(want) {
					t.Fatalf("call %d: sent %d messages, golden has %d: %s", index, len(sent), len(want), sent)
				}
				for i := range want {
					var a, b bytes.Buffer
					if err := json.Compact(&a, sent[i]); err != nil {
						t.Fatal(err)
					}
					if err := json.Compact(&b, want[i]); err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(a.Bytes(), b.Bytes()) {
						t.Fatalf("call %d message %d:\n got  %s\n want %s", index, i, a.Bytes(), b.Bytes())
					}
				}
				next := plan.Next
				ledger = &next
				turnIndex++
			}
		})
	}
}

// The evidence admits the write, not the tail: with evidence a steer after
// an image tool result is written, and so is the neutral marker that
// cancels a steer the worker was sent (pathfinder #4034's evidence covers
// it). Without evidence both are refused.
func TestEvidenceAdmitsTheSteerAndTheNeutralMarkerAfterAnImageToolTail(t *testing.T) {
	binding := suffixBinding(EmitOnChangeV1, "none")
	binding.NeutralText = "Until the next steering instruction, use your normal judgement."
	imageAt := func(messages []llmprotocol.Message) func(uint32) bool {
		return func(index uint32) bool {
			return ImageToolTail(llmprotocol.Request{Messages: messages[:index+1]}, llmprotocol.OpenAIChatV1)
		}
	}
	ask := text(llmprotocol.RoleUser, "Read it.")
	image := []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1")}
	for _, admitted := range []bool{true, false} {
		steer, err := PlanTurn(Turn{
			Binding: binding, Messages: Messages(image), TurnIndex: 1, Requested: "up",
			ImageToolTail: true, ImageToolTailAdmitted: admitted, ImageToolTailAt: imageAt(image),
		})
		if err != nil {
			t.Fatal(err)
		}
		if admitted && (steer.Refused != "" || steer.Written != WrittenInstruction || steer.LevelInForce != "up") {
			t.Fatalf("evidenced steer: refused %q written %q level %q", steer.Refused, steer.Written, steer.LevelInForce)
		}
		if !admitted && (steer.Refused != RefusedImageToolTail || steer.Emitted) {
			t.Fatalf("unevidenced steer: refused %q emitted %v, want the refusal", steer.Refused, steer.Emitted)
		}
	}

	// A steer on a text tail, sent to every worker; then a return to
	// neutral after an image tool result.
	text1 := []llmprotocol.Message{ask, toolCall("c1"), textResult("c1")}
	steer, err := PlanTurn(Turn{Binding: binding, Messages: Messages(text1), TurnIndex: 1, Requested: "up", ImageToolTailAt: imageAt(text1)})
	if err != nil || !steer.Emitted {
		t.Fatalf("text-tail steer: emitted %v, err %v", steer.Emitted, err)
	}
	later := append(append([]llmprotocol.Message(nil), text1...),
		text(llmprotocol.RoleAssistant, ""), toolCall("c2"), imageResult("c2"))
	for _, admitted := range []bool{true, false} {
		ledger := steer.Next
		marker, err := PlanTurn(Turn{
			Binding: binding, Ledger: &ledger, Messages: Messages(later), TurnIndex: 3, Requested: "none",
			ImageToolTail: true, ImageToolTailAdmitted: admitted, ImageToolTailAt: imageAt(later),
		})
		if err != nil {
			t.Fatal(err)
		}
		if admitted && (marker.Refused != "" || marker.Written != WrittenNeutralMarker) {
			t.Fatalf("evidenced marker: refused %q written %q, want the neutral marker", marker.Refused, marker.Written)
		}
		if !admitted && (marker.Refused != RefusedImageToolTail || marker.Emitted || marker.LevelInForce != "up") {
			t.Fatalf("unevidenced marker: refused %q emitted %v level %q, want the refusal with up in force",
				marker.Refused, marker.Emitted, marker.LevelInForce)
		}
	}
}

// A worker that withholds an evidenced image-tail steer decides its own
// writes against what it is sent: asked for the same level on a later turn,
// it is steered there; and a return to neutral over a steer it was sent is
// still marked, though the item in force elsewhere is withheld from it.
func TestAWorkerWithoutEvidenceDecidesAgainstWhatItIsSent(t *testing.T) {
	binding := suffixBinding(EmitOnChangeV1, "none")
	binding.NeutralText = "Until the next steering instruction, use your normal judgement."
	ask := text(llmprotocol.RoleUser, "Read it.")
	imageAt := func(messages []llmprotocol.Message) func(uint32) bool {
		return func(index uint32) bool {
			return ImageToolTail(llmprotocol.Request{Messages: messages[:index+1]}, llmprotocol.OpenAIChatV1)
		}
	}
	tail := []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1")}
	evidenced, err := PlanTurn(Turn{
		Binding: binding, Messages: Messages(tail), TurnIndex: 1, Requested: "up",
		ImageToolTail: true, ImageToolTailAdmitted: true, ImageToolTailAt: imageAt(tail),
	})
	if err != nil || !evidenced.Emitted {
		t.Fatalf("the evidenced worker was not steered: emitted %v, err %v", evidenced.Emitted, err)
	}
	later := append(append([]llmprotocol.Message(nil), tail...),
		text(llmprotocol.RoleAssistant, "Seen."), text(llmprotocol.RoleUser, "Next."))
	plan, err := PlanTurn(Turn{
		Binding: binding, Ledger: &evidenced.Next, Messages: Messages(later), TurnIndex: 2, Requested: "up",
		ImageToolTailAt: imageAt(later),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Emitted || plan.LevelInForce != "up" || len(plan.Applied().Entries) != 1 || len(plan.Next.Entries) != 2 {
		t.Fatalf("later turn: emitted %v level %q, sent %d of %d items; want its own up steer",
			plan.Emitted, plan.LevelInForce, len(plan.Applied().Entries), len(plan.Next.Entries))
	}

	// Steered on text first, then up after an image on the evidenced worker:
	// the unevidenced worker returning to neutral is sent the marker over the
	// down it was sent.
	first, err := PlanTurn(Turn{Binding: binding, Messages: Messages([]llmprotocol.Message{ask}), TurnIndex: 1, Requested: "down"})
	if err != nil || !first.Emitted {
		t.Fatalf("first steer: emitted %v, err %v", first.Emitted, err)
	}
	run := []llmprotocol.Message{ask, text(llmprotocol.RoleAssistant, ""), toolCall("c1"), imageResult("c1")}
	second, err := PlanTurn(Turn{
		Binding: binding, Ledger: &first.Next, Messages: Messages(run), TurnIndex: 2, Requested: "up",
		ImageToolTail: true, ImageToolTailAdmitted: true, ImageToolTailAt: imageAt(run),
	})
	if err != nil || !second.Emitted {
		t.Fatalf("evidenced up: emitted %v, err %v", second.Emitted, err)
	}
	back := append(append([]llmprotocol.Message(nil), run...),
		text(llmprotocol.RoleAssistant, "Seen."), text(llmprotocol.RoleUser, "Next."))
	plan, err = PlanTurn(Turn{
		Binding: binding, Ledger: &second.Next, Messages: Messages(back), TurnIndex: 3, Requested: "none",
		ImageToolTailAt: imageAt(back),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Emitted || plan.Written != WrittenNeutralMarker || plan.InstructionState != InstructionNeutralMarker {
		t.Fatalf("return to neutral: emitted %v written %q state %q; want the neutral marker", plan.Emitted, plan.Written, plan.InstructionState)
	}
}
