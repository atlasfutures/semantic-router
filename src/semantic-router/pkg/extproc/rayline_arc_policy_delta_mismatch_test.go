package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func mismatchMessage(t *testing.T, value any) json.RawMessage {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

// mismatchReply is a staged assistant reply with a thinking, a text and a
// tool_use block; edit changes a fresh copy of it.
func mismatchReply(edit func(message map[string]any, blocks []any) []any) map[string]any {
	blocks := []any{
		map[string]any{"type": "thinking", "thinking": "plan it", "signature": "sig-1"},
		map[string]any{"type": "text", "text": "Listing them.", "citations": []any{}},
		map[string]any{"type": "redacted_thinking", "data": "opaque"},
		map[string]any{"type": "tool_use", "id": "toolu_1", "caller": map[string]any{"type": "direct"}, "name": "Bash", "input": map[string]any{"command": "ls"}},
	}
	message := map[string]any{"role": "assistant"}
	if edit != nil {
		blocks = edit(message, blocks)
	}
	if _, set := message["content"]; !set {
		message["content"] = blocks
	}
	return message
}

func block(blocks []any, index int) map[string]any {
	return blocks[index].(map[string]any)
}

// Each part of the appended reply an echo can differ in is named, one of a
// fixed set; what the digest ignores (cache_control, a signature, a direct
// caller, empty citations) differs in nothing and reads "other".
func TestAppendedMismatchPartNamesThePartThatDiffers(t *testing.T) {
	staged := policyAppendedShapeOf([]json.RawMessage{mismatchMessage(t, mismatchReply(nil))})
	for _, tc := range []struct {
		name string
		edit func(map[string]any, []any) []any
		want string
	}{
		{"role", func(message map[string]any, blocks []any) []any { message["role"] = "user"; return blocks }, appendedMismatchRole},
		{"content_form", func(message map[string]any, blocks []any) []any {
			message["content"] = "Listing them."
			return blocks
		}, appendedMismatchContentForm},
		{"content_null", func(message map[string]any, blocks []any) []any { message["content"] = nil; return blocks }, appendedMismatchContentForm},
		{"block_count", func(_ map[string]any, blocks []any) []any { return blocks[:3] }, appendedMismatchBlockCount},
		{"block_order", func(_ map[string]any, blocks []any) []any {
			blocks[1], blocks[3] = blocks[3], blocks[1]
			return blocks
		}, appendedMismatchBlockOrder},
		{"block_type", func(_ map[string]any, blocks []any) []any { block(blocks, 1)["type"] = "output_text"; return blocks }, appendedMismatchBlockType},
		{"text", func(_ map[string]any, blocks []any) []any { block(blocks, 1)["text"] = "Listing."; return blocks }, appendedMismatchText},
		{"thinking_text", func(_ map[string]any, blocks []any) []any { block(blocks, 0)["thinking"] = "plan"; return blocks }, appendedMismatchThinking},
		{"redacted_data", func(_ map[string]any, blocks []any) []any { block(blocks, 2)["data"] = "other"; return blocks }, appendedMismatchThinking},
		{"tool_input", func(_ map[string]any, blocks []any) []any {
			block(blocks, 3)["input"] = map[string]any{"command": "ls -la"}
			return blocks
		}, appendedMismatchToolUse},
		{"tool_id", func(_ map[string]any, blocks []any) []any { block(blocks, 3)["id"] = "toolu_2"; return blocks }, appendedMismatchToolUse},
		{"citations", func(_ map[string]any, blocks []any) []any {
			block(blocks, 1)["citations"] = []any{map[string]any{"type": "web"}}
			return blocks
		}, appendedMismatchBlockMetadata},
		{"block_extra_member", func(_ map[string]any, blocks []any) []any { block(blocks, 3)["extra"] = true; return blocks }, appendedMismatchBlockMetadata},
		{"message_extra_member", func(message map[string]any, blocks []any) []any { message["model"] = "m"; return blocks }, appendedMismatchMessageMetadata},
		{"ignored_members_only", func(message map[string]any, blocks []any) []any {
			message["cache_control"] = map[string]any{"type": "ephemeral"}
			block(blocks, 0)["signature"] = ""
			delete(block(blocks, 1), "citations")
			delete(block(blocks, 3), "caller")
			block(blocks, 3)["cache_control"] = map[string]any{"type": "ephemeral"}
			return blocks
		}, appendedMismatchOther},
	} {
		t.Run(tc.name, func(t *testing.T) {
			echoed := policyAppendedShapeOf([]json.RawMessage{mismatchMessage(t, mismatchReply(tc.edit))})
			if got := policyAppendedShapeMismatch(staged, echoed); got != tc.want {
				t.Fatalf("part %q, want %q", got, tc.want)
			}
		})
	}
	// A string content that differs is its text.
	stringStaged := policyAppendedShapeOf([]json.RawMessage{mismatchMessage(t, map[string]any{"role": "assistant", "content": "a"})})
	stringEchoed := policyAppendedShapeOf([]json.RawMessage{mismatchMessage(t, map[string]any{"role": "assistant", "content": "b"})})
	if got := policyAppendedShapeMismatch(stringStaged, stringEchoed); got != appendedMismatchText {
		t.Fatalf("string content part %q, want text", got)
	}
	// An echo that does not parse is other.
	if got := policyAppendedShapeMismatch(staged, policyAppendedShapeOf([]json.RawMessage{json.RawMessage(`not json`)})); got != appendedMismatchOther {
		t.Fatalf("unparsed echo part %q, want other", got)
	}
}

// A Responses function_call item's arguments are its tool call; its other
// members are the message's metadata.
func TestAppendedMismatchPartReadsResponsesItems(t *testing.T) {
	item := func(arguments, status string) json.RawMessage {
		return mismatchMessage(t, map[string]any{"type": "function_call", "name": "Bash", "call_id": "c1", "arguments": arguments, "status": status})
	}
	staged := policyAppendedShapeOf([]json.RawMessage{item(`{"command":"ls"}`, "completed")})
	if got := policyAppendedShapeMismatch(staged, policyAppendedShapeOf([]json.RawMessage{item(`{"command":"pwd"}`, "completed")})); got != appendedMismatchToolUse {
		t.Fatalf("arguments part %q, want tool_use", got)
	}
	if got := policyAppendedShapeMismatch(staged, policyAppendedShapeOf([]json.RawMessage{item(`{"command":"ls"}`, "in_progress")})); got != appendedMismatchMessageMetadata {
		t.Fatalf("status part %q, want message_metadata", got)
	}
}

// A reply past the bounds keeps no shape, and a staged turn without one
// reads unavailable.
func TestAppendedMismatchPartUnavailableWithoutAShape(t *testing.T) {
	many := make([]json.RawMessage, maxPolicyAppendedShapeMessages+1)
	for index := range many {
		many[index] = mismatchMessage(t, map[string]any{"role": "assistant", "content": "x"})
	}
	if policyAppendedShapeOf(many) != nil {
		t.Fatal("a reply past the message bound kept a shape")
	}
	blocks := make([]any, maxPolicyAppendedShapeBlocks+1)
	for index := range blocks {
		blocks[index] = map[string]any{"type": "text", "text": "x"}
	}
	if policyAppendedShapeOf([]json.RawMessage{mismatchMessage(t, map[string]any{"role": "assistant", "content": blocks})}) != nil {
		t.Fatal("a reply past the block bound kept a shape")
	}
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	scorer := fixture.scorer()
	scorer.staged.put(fixture.episode, policyStagedTurn{turnIndex: 1, count: 2, appendedFrom: 1})
	messages := []json.RawMessage{mismatchMessage(t, deltaUser), mismatchMessage(t, deltaAssistant), mismatchMessage(t, deltaToolResult)}
	if got := policyAppendedMismatchPart(scorer, fixture.episode, messages); got != appendedMismatchUnavailable {
		t.Fatalf("part %q, want unavailable", got)
	}
}

// The held turn's staged_turn event names the part of an altered echo as
// appended_mismatch_part, and the turn still goes in full; a turn that goes
// out as a delta carries no part.
func TestDeltaStagedTurnLogsTheAppendedMismatchPart(t *testing.T) {
	for _, tc := range []struct {
		name string
		echo map[string]any
		mode string
		part string
	}{
		{"unchanged", deltaAssistant, policyDecisionModeDelta, ""},
		{"text_changed", map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "Listing these.", "citations": []any{}},
			map[string]any{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": map[string]any{"command": "ls"}},
		}}, policyDecisionModeSync, appendedMismatchText},
		{"tool_input_changed", map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "Listing them."},
			map[string]any{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": map[string]any{"command": "ls -la"}},
		}}, policyDecisionModeSync, appendedMismatchToolUse},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
			fixture.stagedFirstTurn(t)
			logs := captureLogs(t)
			second, ctx := fixture.turn(t, policyTestRequest(t, deltaUser, tc.echo, deltaToolResult))
			if second.RaylineARC.PolicyDecisionMode != tc.mode {
				t.Fatalf("held turn went %s, want %s", second.RaylineARC.PolicyDecisionMode, tc.mode)
			}
			fields := findLogEvent(t, logs, "rayline_arc_policy_staged_turn")
			part, present := fields["appended_mismatch_part"]
			if tc.part == "" {
				if present || fields["delta_mismatch_reason"] != "none" {
					t.Fatalf("delta turn logged %v", fields)
				}
			} else if fields["delta_mismatch_reason"] != "appended_mismatch" || part != tc.part {
				t.Fatalf("staged_turn logged reason %v part %v, want appended_mismatch %s", fields["delta_mismatch_reason"], part, tc.part)
			}
			for key, value := range fields {
				if text, ok := value.(string); ok && strings.Contains(text, "Listing") {
					t.Fatalf("%s carries message text", key)
				}
			}
			complete(t, ctx, deltaReply(t))
		})
	}
}
