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
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// llmprotocolContentUnion is the whole content union at pkg/llmprotocol.
// Every kind must be rendered or dropped on purpose. Widen this list from that
// union when it grows, and decide the new kind's fate in the same change.
var llmprotocolContentUnion = []llmprotocol.ContentKind{
	llmprotocol.ContentAudio,
	llmprotocol.ContentFile,
	llmprotocol.ContentGeneratedImage,
	llmprotocol.ContentImage,
	llmprotocol.ContentReasoning,
	llmprotocol.ContentRefusal,
	llmprotocol.ContentText,
	llmprotocol.ContentToolCall,
	llmprotocol.ContentToolResult,
	llmprotocol.ContentUnmodeled,
	llmprotocol.ContentVideo,
}

func userRequest(blocks ...llmprotocol.Content) *llmprotocol.Request {
	return &llmprotocol.Request{
		Generation: 1,
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: blocks},
		},
	}
}

// A content kind the projection does not know is the one case that must never
// degrade quietly. Silently stripping it would show the trained selector a
// truncated conversation and route on it; failing closed answers 503 instead.
func TestProjectTurnsFailsClosedOnUnknownContentKind(t *testing.T) {
	unknown := llmprotocol.ContentKind("future_kind_from_a_later_release")
	request := userRequest(
		llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "keep"},
		llmprotocol.Content{Kind: unknown, Text: "drop me silently"},
	)

	turns, err := ProjectTurns(request, TurnOptions{})
	if err == nil {
		t.Fatalf("ProjectTurns() accepted an unknown content kind: %v", turns)
	}
	if turns != nil {
		t.Fatalf("a failed projection returned turns: %v", turns)
	}
	if code := TurnNormalizationErrorCode(err); code != "unknown_item" {
		t.Fatalf("error code = %q, want %q", code, "unknown_item")
	}
	if detail := TurnNormalizationErrorDetail(err); detail != string(unknown) {
		t.Fatalf("error detail = %q, want %q", detail, unknown)
	}
	if path := TurnNormalizationErrorPath(err); path != "messages[0].content[1].kind" {
		t.Fatalf("error path = %q", path)
	}
}

func TestProjectTurnsFailsClosedOnUnknownRole(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 1,
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.Role("auditor"),
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "x"}},
		}},
	}

	_, err := ProjectTurns(request, TurnOptions{})
	if code := TurnNormalizationErrorCode(err); code != "unknown_item" {
		t.Fatalf("error code = %q, want %q", code, "unknown_item")
	}
	if detail := TurnNormalizationErrorDetail(err); detail != "auditor" {
		t.Fatalf("error detail = %q, want %q", detail, "auditor")
	}
}

// Every kind in the union has exactly one fate, and the two sets never
// overlap. That is what makes the accepted content auditable.
func TestContentKindsAreRenderedOrDroppedExactlyOnce(t *testing.T) {
	rendered := map[llmprotocol.ContentKind]bool{
		llmprotocol.ContentText:       true,
		llmprotocol.ContentToolCall:   true,
		llmprotocol.ContentToolResult: true,
	}
	for _, kind := range llmprotocolContentUnion {
		switch {
		case rendered[kind] && droppedContentKinds[kind]:
			t.Errorf("content kind %q is both rendered and dropped", kind)
		case !rendered[kind] && !droppedContentKinds[kind]:
			t.Errorf("content kind %q has no recorded fate", kind)
		}
	}
	for kind := range droppedContentKinds {
		if !containsKind(llmprotocolContentUnion, kind) {
			t.Errorf("drop table names %q, which is not in the union", kind)
		}
	}
}

// A dropped kind contributes no text, and the turn that carried it survives.
// Dropping the turn instead would change the turn count the encoder sees.
func TestDroppedContentKindsContributeNoText(t *testing.T) {
	for kind := range droppedContentKinds {
		turns, err := ProjectTurns(
			userRequest(llmprotocol.Content{Kind: kind, Text: "ignored"}),
			TurnOptions{},
		)
		if err != nil {
			t.Fatalf("kind %q: ProjectTurns() error = %v", kind, err)
		}
		want := []Turn{{Role: "user", Text: ""}}
		assertTurnsIdentical(t, turns, want)
	}
}

// A tool result whose call was never announced cannot be named, and a
// [tool_result ] with an empty name is a token shape the checkpoint has never
// been shown. The codec rejects orphan results at ingress; this keeps the
// projection refusing them too.
func TestProjectTurnsFailsClosedOnUnresolvedToolResult(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 1,
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleTool,
			Content: []llmprotocol.Content{{
				Kind:       llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{CallID: "never-announced"},
			}},
		}},
	}

	_, err := ProjectTurns(request, TurnOptions{})
	if code := TurnNormalizationErrorCode(err); code != "unresolved_tool_id" {
		t.Fatalf("error code = %q, want %q", code, "unresolved_tool_id")
	}
}

// The codec hoists every system message into Instructions, so that is the only
// system text a decoded request carries. It is folded into the first user turn
// only when the deployment asks for it, and it never becomes a turn of its
// own: the wire contract has two roles.
func TestProjectTurnsFoldsInstructionsOnlyWhenAsked(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 1,
		Instructions: []llmprotocol.InstructionBlock{{
			Role:    llmprotocol.RoleSystem,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "standing brief"}},
		}},
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "ask"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "answer"}}},
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "again"}}},
		},
	}

	dropped, err := ProjectTurns(request, TurnOptions{})
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	assertTurnsIdentical(t, dropped, []Turn{
		{Role: "user", Text: "ask"},
		{Role: "assistant", Text: "answer"},
		{Role: "user", Text: "again"},
	})

	folded, err := ProjectTurns(request, TurnOptions{IncludeSystemText: true})
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	// The brief lands on the first user turn only. That turn is what the
	// serializer renders as the episode's [Task] block.
	assertTurnsIdentical(t, folded, []Turn{
		{Role: "user", Text: "standing brief\n\nask"},
		{Role: "assistant", Text: "answer"},
		{Role: "user", Text: "again"},
	})
}

// Neither system-text option may add a turn, remove one, or change a role.
// Only the turn text gets longer. The encoder wire schema pins the role set,
// so a shape change would emit tokens the checkpoint has never been shown.
func TestProjectTurnsKeepsTheTurnShapeInEverySystemTextState(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 1,
		Instructions: []llmprotocol.InstructionBlock{{
			Role:    llmprotocol.RoleDeveloper,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "brief"}},
		}},
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "one"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "two"}}},
		},
	}
	wantRoles := []string{"user", "assistant"}

	for _, options := range []TurnOptions{
		{},
		{IncludeSystemText: true},
		{DropMidConversationSystemText: true},
		{IncludeSystemText: true, DropMidConversationSystemText: true},
	} {
		turns, err := ProjectTurns(request, options)
		if err != nil {
			t.Fatalf("options %+v: ProjectTurns() error = %v", options, err)
		}
		if len(turns) != len(wantRoles) {
			t.Fatalf("options %+v: %d turns, want %d", options, len(turns), len(wantRoles))
		}
		for index, role := range wantRoles {
			if turns[index].Role != role {
				t.Fatalf(
					"options %+v: turn %d role = %q, want %q",
					options, index, turns[index].Role, role,
				)
			}
		}
	}
}

// A system message that reaches the projection inside the message sequence
// still has a known position, so the mid-conversation scope can be honoured
// there. No public wire format produces one today, because the codec hoists
// system text into Instructions.
func TestProjectTurnsHonoursTheMidConversationScopeWhenPositionSurvives(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 1,
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "one"}}},
			{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "answer in JSON"}}},
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "two"}}},
		},
	}

	included, err := ProjectTurns(request, TurnOptions{})
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	assertTurnsIdentical(t, included, []Turn{
		{Role: "user", Text: "one"},
		{Role: "user", Text: "answer in JSON\n\ntwo"},
	})

	kept, err := ProjectTurns(request, TurnOptions{DropMidConversationSystemText: true})
	if err != nil {
		t.Fatalf("ProjectTurns() error = %v", err)
	}
	assertTurnsIdentical(t, kept, []Turn{
		{Role: "user", Text: "one"},
		{Role: "user", Text: "two"},
	})
}

func containsKind(kinds []llmprotocol.ContentKind, want llmprotocol.ContentKind) bool {
	for _, kind := range kinds {
		if kind == want {
			return true
		}
	}
	return false
}

// stringCoerce belongs to the calibrated renderer, which must stay byte for
// byte what the encoder was trained against. The codec now refuses tool call
// arguments that are not a strict JSON object, so no decoded request can reach
// it; the contract is pinned here rather than deleted, because deleting it
// would edit the renderer.
func TestStringCoerceKeepsThePythonRenderingItWasCalibratedOn(t *testing.T) {
	for _, test := range []struct {
		value any
		want  string
	}{
		{value: nil, want: ""},
		{value: true, want: "True"},
		{value: false, want: "False"},
		{value: "plain", want: "plain"},
		{value: []any{1, "two"}, want: `[1,"two"]`},
	} {
		if got := stringCoerce(test.value); got != test.want {
			t.Fatalf("stringCoerce(%v) = %q, want %q", test.value, got, test.want)
		}
	}
}
