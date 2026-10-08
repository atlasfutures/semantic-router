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
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func imageResult(id string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: id, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "shot.png"},
			{Kind: llmprotocol.ContentImage, MediaType: "image/png", Data: tinyPNG},
		}},
	}}}
}

func textResult(id string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: id, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "README.md"},
		}},
	}}}
}

func userImage() llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentImage, MediaType: "image/png", Data: tinyPNG},
	}}
}

// ImageToolTail reads the neutral turn as pathfinder's placer reads the
// encoded body (ADR 0129): Chat's images follow the run in user messages,
// Responses keeps them in the tool output, and Messages is not governed.
func TestImageToolTailReadsTheTurnAsTheWorkerWireCarriesIt(t *testing.T) {
	ask := text(llmprotocol.RoleUser, "Read it.")
	for _, tc := range []struct {
		name            string
		messages        []llmprotocol.Message
		chat, responses bool
	}{
		{"image result", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1")}, true, true},
		{"image result then user text", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1"), text(llmprotocol.RoleUser, "Colour?")}, true, true},
		{"mixed run", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1"), textResult("c2")}, true, true},
		{"text result", []llmprotocol.Message{ask, toolCall("c1"), textResult("c1")}, false, false},
		// On Chat a user's own image after a tool run is indistinguishable
		// from a hoisted one; Responses reads only the tool output.
		{"user image after a text result", []llmprotocol.Message{ask, toolCall("c1"), textResult("c1"), userImage()}, true, false},
		{"user image, no tool run", []llmprotocol.Message{ask, userImage()}, false, false},
		{"image result answered, then a new ask", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1"), text(llmprotocol.RoleAssistant, "Seen."), ask}, false, false},
	} {
		if got := ImageToolTail(tc.messages, llmprotocol.OpenAIChatV1); got != tc.chat {
			t.Errorf("%s on chat = %v, want %v", tc.name, got, tc.chat)
		}
		if got := ImageToolTail(tc.messages, llmprotocol.OpenAIResponsesV1); got != tc.responses {
			t.Errorf("%s on responses = %v, want %v", tc.name, got, tc.responses)
		}
		if ImageToolTail(tc.messages, llmprotocol.AnthropicMessagesV1) {
			t.Errorf("%s on messages is governed", tc.name)
		}
	}
}

// ADR 0129 decision 5: a steering suffix after an image tool result is
// refused. Nothing is written, the level in force holds, and the plan names
// the refusal. A later turn without an image tail writes the steer.
func TestASteeringSuffixAfterAnImageToolTailIsRefused(t *testing.T) {
	binding := suffixBinding(EmitOnChange, "none")
	ask := text(llmprotocol.RoleUser, "Read it.")
	messages := []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1")}
	plan, err := PlanTurn(Turn{
		Binding: binding, Messages: Messages(messages), TurnIndex: 1, Requested: "up", ImageToolTail: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Refused != RefusedImageToolTail || plan.Emitted || len(plan.Next.Entries) != 0 || plan.LevelInForce != "none" {
		t.Fatalf("plan = refused %q emitted %v entries %d level %q; want the refusal, nothing written, none in force",
			plan.Refused, plan.Emitted, len(plan.Next.Entries), plan.LevelInForce)
	}
	later := append(append([]llmprotocol.Message(nil), messages...),
		text(llmprotocol.RoleAssistant, "Seen."), text(llmprotocol.RoleUser, "Next."))
	plan, err = PlanTurn(Turn{Binding: binding, Ledger: &plan.Next, Messages: Messages(later), TurnIndex: 2, Requested: "up"})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Refused != "" || !plan.Emitted || plan.LevelInForce != "up" {
		t.Fatalf("later plan = refused %q emitted %v level %q; want the steer written", plan.Refused, plan.Emitted, plan.LevelInForce)
	}
}

// A hold writes nothing, so it has nothing to refuse; and the per-turn effort
// lever writes a configuration, not text, which ADR 0129 leaves out.
func TestOnlyAWriteOfTheSteeringSuffixIsRefused(t *testing.T) {
	ask := text(llmprotocol.RoleUser, "Read it.")
	messages := []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1")}
	plan, err := PlanTurn(Turn{
		Binding: suffixBinding(EmitOnChange, "none"), Messages: Messages(messages), TurnIndex: 1, Requested: "none",
		ImageToolTail: true,
	})
	if err != nil || plan.Refused != "" {
		t.Fatalf("a hold at none: refused %q, err %v", plan.Refused, err)
	}
	plan, err = PlanTurn(Turn{
		Binding: effortBinding(), Messages: Messages(messages), TurnIndex: 1, Requested: "max", ImageToolTail: true,
	})
	if err != nil || plan.Refused != "" || !plan.Emitted {
		t.Fatalf("the effort lever: refused %q emitted %v, err %v", plan.Refused, plan.Emitted, err)
	}
}
