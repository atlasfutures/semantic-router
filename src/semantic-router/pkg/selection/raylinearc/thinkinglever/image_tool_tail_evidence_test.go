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
// an image tool result is written, and the neutral marker that follows a
// steered level is written too (pathfinder #4034's evidence covers it).
func TestEvidenceAdmitsTheSteerAndTheNeutralMarkerAfterAnImageToolTail(t *testing.T) {
	binding := suffixBinding(EmitOnChangeV1, "none")
	binding.NeutralText = "Until the next steering instruction, use your normal judgement."
	ask := text(llmprotocol.RoleUser, "Read it.")
	messages := []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1")}
	plan, err := PlanTurn(Turn{
		Binding: binding, Messages: Messages(messages), TurnIndex: 1, Requested: "up",
		ImageToolTail: true, ImageToolTailAdmitted: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Refused != "" || !plan.Emitted || plan.Written != WrittenInstruction || plan.LevelInForce != "up" {
		t.Fatalf("steer: refused %q emitted %v written %q level %q", plan.Refused, plan.Emitted, plan.Written, plan.LevelInForce)
	}
	later := append(append([]llmprotocol.Message(nil), messages...),
		text(llmprotocol.RoleAssistant, ""), toolCall("c2"), imageResult("c2"))
	for _, admitted := range []bool{true, false} {
		ledger := plan.Next
		marker, err := PlanTurn(Turn{
			Binding: binding, Ledger: &ledger, Messages: Messages(later), TurnIndex: 3, Requested: "none",
			ImageToolTail: true, ImageToolTailAdmitted: admitted,
		})
		if err != nil {
			t.Fatal(err)
		}
		if admitted && (marker.Refused != "" || marker.Written != WrittenNeutralMarker) {
			t.Fatalf("evidenced marker: refused %q written %q, want the neutral marker", marker.Refused, marker.Written)
		}
		if !admitted && (marker.Refused != RefusedImageToolTail || marker.Emitted) {
			t.Fatalf("unevidenced marker: refused %q emitted %v, want the refusal", marker.Refused, marker.Emitted)
		}
	}
}
