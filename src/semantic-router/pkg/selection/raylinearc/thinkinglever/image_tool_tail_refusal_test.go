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

func reasoningOnly() llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentReasoning, Text: "thinking"},
	}}
}

func redactedOnly() llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentUnmodeled, Unmodeled: &llmprotocol.UnmodeledBlock{
			Format: llmprotocol.AnthropicMessagesV1, Type: "redacted_thinking", Raw: []byte(`{"type":"redacted_thinking","data":"x"}`),
		},
	}}}
}

func serverToolUseOnly() llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentUnmodeled, Unmodeled: &llmprotocol.UnmodeledBlock{
			Format: llmprotocol.AnthropicMessagesV1, Type: "server_tool_use",
			Raw: []byte(`{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"x"}}`),
		},
	}}}
}

func webSearchResultOnly() llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentUnmodeled, Unmodeled: &llmprotocol.UnmodeledBlock{
			Format: llmprotocol.AnthropicMessagesV1, Type: "web_search_tool_result",
			Raw: []byte(`{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]}`),
		},
	}}}
}

func encryptedReasoningOnly() llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentUnmodeled, Unmodeled: &llmprotocol.UnmodeledBlock{
			Format: llmprotocol.OpenAIResponsesV1, Type: "reasoning",
			Raw: []byte(`{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAA"}`),
		},
	}}}
}

func refusal() llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentRefusal, Text: "I can't help with that."},
	}}
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
		// A user image after a tool run counts on both wires: on Chat it is
		// indistinguishable from a hoisted one, and a Responses codec may
		// hoist one into a user item after the outputs (pathfinder #3996
		// round 2).
		{"user image after a text result", []llmprotocol.Message{ask, toolCall("c1"), textResult("c1"), userImage()}, true, true},
		{"user image, no tool run", []llmprotocol.Message{ask, userImage()}, false, false},
		// Reasoning alone, which the adapter may drop, does not end the run.
		{"image result, reasoning, then user text", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1"), reasoningOnly(), text(llmprotocol.RoleUser, "Colour?")}, true, true},
		{"image result, redacted thinking, then user text", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1"), redactedOnly(), text(llmprotocol.RoleUser, "Colour?")}, true, true},
		// A message the target encoder omits whole (an Anthropic
		// server_tool_use alone) never reaches the worker either.
		{"image result, server tool use, then user text", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1"), serverToolUseOnly(), text(llmprotocol.RoleUser, "Colour?")}, true, true},
		// Nor does a user message the encoder omits, inside the run.
		{"image result, omitted user message, text result", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1"), webSearchResultOnly(), textResult("c2")}, true, true},
		// An encrypted reasoning item the request does not forward.
		{"image result, unforwarded encrypted reasoning, then user text", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1"), encryptedReasoningOnly(), text(llmprotocol.RoleUser, "Colour?")}, true, true},
		// A refusal is the assistant's answer: the image turn is over.
		{"image result answered with a refusal, then a new ask", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1"), refusal(), ask}, false, false},
		{"image result answered, then a new ask", []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1"), text(llmprotocol.RoleAssistant, "Seen."), ask}, false, false},
	} {
		if got := ImageToolTail(llmprotocol.Request{Messages: tc.messages}, llmprotocol.OpenAIChatV1); got != tc.chat {
			t.Errorf("%s on chat = %v, want %v", tc.name, got, tc.chat)
		}
		if got := ImageToolTail(llmprotocol.Request{Messages: tc.messages}, llmprotocol.OpenAIResponsesV1); got != tc.responses {
			t.Errorf("%s on responses = %v, want %v", tc.name, got, tc.responses)
		}
		if ImageToolTail(llmprotocol.Request{Messages: tc.messages}, llmprotocol.AnthropicMessagesV1) {
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
	// The refused turn stays attributed to the drawn control, the policy's
	// action (pathfinder ADR 0129).
	up, _ := binding.Level("up")
	if drawn := binding.ControlSHA256(up); plan.ControlInForce != drawn {
		t.Fatalf("refused turn attributed to %q, want the drawn control %q", plan.ControlInForce, drawn)
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

// A full ledger cannot write either way; it reports itself, not the image
// refusal.
func TestAFullLedgerIsReportedBeforeTheImageRefusal(t *testing.T) {
	binding := suffixBinding(EmitOnChange, "none")
	ask := text(llmprotocol.RoleUser, "Read it.")
	first, err := PlanTurn(Turn{Binding: binding, Messages: Messages([]llmprotocol.Message{ask}), TurnIndex: 1, Requested: "down", MaxEntries: 1})
	if err != nil || !first.Emitted {
		t.Fatalf("first turn emitted %v, err %v", first.Emitted, err)
	}
	messages := []llmprotocol.Message{ask, toolCall("c1"), imageResult("c1")}
	plan, err := PlanTurn(Turn{
		Binding: binding, Ledger: &first.Next, Messages: Messages(messages), TurnIndex: 2, Requested: "up",
		MaxEntries: 1, ImageToolTail: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Skipped != SkipLedgerFull || plan.Refused != "" {
		t.Fatalf("skipped %q refused %q, want ledger_full and no refusal", plan.Skipped, plan.Refused)
	}
}
