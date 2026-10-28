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

package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The delta digest reads a staged reply and a client's echo of it as the
// service's canonical projection does (canonical_conversation.py _clean):
// caller on tool_use and server_tool_use, and signature on thinking, are
// dropped whatever their value; citations only when null or []. Thinking
// text, non-empty citations, redacted_thinking data and block order count.

// Workshop's agent rebuilds the assistant message it echoes
// (message_converter.py): thinking first with "signature" set ("" when the
// reply had none), text without citations, tool_use without caller.
var (
	workshopSignedReply = []any{
		map[string]any{"type": "thinking", "thinking": "list first", "signature": "sig-1"},
		map[string]any{"type": "text", "text": "Listing them."},
		map[string]any{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": map[string]any{"command": "ls", "cwd": "/"}},
	}
	workshopSignedEcho = map[string]any{"role": "assistant", "content": []any{
		map[string]any{"type": "thinking", "thinking": "list first", "signature": "sig-1"},
		map[string]any{"type": "text", "text": "Listing them."},
		map[string]any{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": map[string]any{"cwd": "/", "command": "ls"}},
	}}
	// An unsigned thinking reply, as a non-Claude arm's reaches the client.
	workshopUnsignedReply = []any{
		map[string]any{"type": "thinking", "thinking": "now read it"},
		map[string]any{"type": "tool_use", "id": "toolu_2", "name": "Read", "input": map[string]any{"path": "README.md"}},
	}
	workshopUnsignedEcho = map[string]any{"role": "assistant", "content": []any{
		map[string]any{"type": "thinking", "thinking": "now read it", "signature": ""},
		map[string]any{"type": "tool_use", "id": "toolu_2", "name": "Read", "input": map[string]any{"path": "README.md"}},
	}}
	workshopToolResult = func(id, output string) map[string]any {
		return map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "tool_result", "tool_use_id": id, "content": output},
		}}
	}
)

func anthropicReply(t *testing.T, content []any) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"id": "msg_1", "type": "message", "role": "assistant", "model": "think",
		"content": content, "stop_reason": "tool_use", "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": 10, "output_tokens": 5},
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func TestDeltaDigestMirrorsTheServiceClean(t *testing.T) {
	text := func(extra map[string]any) map[string]any {
		block := map[string]any{"type": "text", "text": "Listing them."}
		for key, value := range extra {
			block[key] = value
		}
		return block
	}
	tool := func(kind string, extra map[string]any) map[string]any {
		block := map[string]any{"type": kind, "id": "toolu_1", "name": "Bash", "input": map[string]any{"command": "ls"}}
		for key, value := range extra {
			block[key] = value
		}
		return block
	}
	thinking := func(text string, signature any, signed bool) map[string]any {
		block := map[string]any{"type": "thinking", "thinking": text}
		if signed {
			block["signature"] = signature
		}
		return block
	}
	citation := []any{map[string]any{"type": "char_location", "cited_text": "a", "document_index": 0, "start_char_index": 0, "end_char_index": 1}}
	direct := map[string]any{"type": "direct"}
	for _, tc := range []struct {
		name  string
		reply []any
		echo  []any
		match bool
	}{
		// A: Workshop's bare echo of a signed thinking tool turn.
		{"workshop_bare_echo", workshopSignedReply, workshopSignedEcho["content"].([]any), true},
		{"citations_absent_vs_staged_empty", []any{text(nil)}, []any{text(nil)}, true},
		{"citations_null_vs_absent", []any{text(map[string]any{"citations": nil})}, []any{text(nil)}, true},
		{"citations_empty_vs_null", []any{text(map[string]any{"citations": []any{}})}, []any{text(map[string]any{"citations": nil})}, true},
		{"citations_nonempty_dropped", []any{text(map[string]any{"citations": citation})}, []any{text(nil)}, false},
		{"citations_nonempty_kept", []any{text(map[string]any{"citations": citation})}, []any{text(map[string]any{"citations": citation})}, true},
		{"caller_absent_vs_staged_direct", []any{tool("tool_use", nil)}, []any{tool("tool_use", nil)}, true},
		{"caller_other_value", []any{tool("tool_use", map[string]any{"caller": direct})}, []any{tool("tool_use", map[string]any{"caller": map[string]any{"type": "code_execution"}})}, true},
		{"server_tool_use_caller", []any{tool("server_tool_use", map[string]any{"caller": direct})}, []any{tool("server_tool_use", nil)}, true},
		{"tool_input_changed", []any{tool("tool_use", nil)}, []any{tool("tool_use", map[string]any{"input": map[string]any{"command": "ls -a"}})}, false},
		// F: an unsigned reply echoed with "signature": "".
		{"signature_absent_vs_empty", []any{thinking("t", nil, false)}, []any{thinking("t", "", true)}, true},
		{"signature_present_vs_absent", []any{thinking("t", "sig", true)}, []any{thinking("t", nil, false)}, true},
		{"signature_present_vs_other", []any{thinking("t", "sig", true)}, []any{thinking("t", "sig-2", true)}, true},
		// D: the thinking text is encoded.
		{"thinking_text_changed", []any{thinking("t", "sig", true)}, []any{thinking("t2", "sig", true)}, false},
		// C: a client that drops the thinking block.
		{"thinking_dropped", []any{thinking("t", "sig", true), text(nil)}, []any{text(nil)}, false},
		// Reordered blocks.
		{"blocks_reordered", []any{thinking("t", "sig", true), text(nil), tool("tool_use", nil)}, []any{thinking("t", "", true), tool("tool_use", nil), text(nil)}, false},
		// I: two thinking blocks merged into one, as Workshop concatenates them.
		{"thinking_merged", []any{thinking("a", "s1", true), thinking("b", "s2", true), text(nil)}, []any{thinking("ab", "s1s2", true), text(nil)}, false},
		// signature is dropped only on thinking: redacted_thinking data counts.
		{"redacted_data_changed", []any{map[string]any{"type": "redacted_thinking", "data": "x"}}, []any{map[string]any{"type": "redacted_thinking", "data": "y"}}, false},
		{"redacted_signature_kept", []any{map[string]any{"type": "redacted_thinking", "data": "x", "signature": "s"}}, []any{map[string]any{"type": "redacted_thinking", "data": "x"}}, false},
		// caller is dropped only on tool_use and server_tool_use.
		{"caller_on_text_kept", []any{text(map[string]any{"caller": direct})}, []any{text(nil)}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			staged, ok := policyAssistantOutput(policyFormatAnthropic, anthropicReply(t, tc.reply))
			if !ok || len(staged) != 1 {
				t.Fatalf("staged %s", staged)
			}
			echo, err := json.Marshal(map[string]any{"role": "assistant", "content": tc.echo})
			if err != nil {
				t.Fatal(err)
			}
			got := policyDeltaDigest(staged, 1) == policyDeltaDigest([]json.RawMessage{echo}, 1)
			if got != tc.match {
				t.Fatalf("staged %s vs echo %s: match %v, want %v", staged[0], echo, got, tc.match)
			}
		})
	}
}

// The removals apply to every message's top-level content blocks, as _clean
// does in every role and format, and before a system or user message's one
// text block is demoted to its string; data nested deeper (a tool's input,
// a tool_result's content) and Responses parts keep every member, which is
// stricter than the service and so safe.
func TestDeltaDigestCleansEveryRoleAtTheBlockBoundary(t *testing.T) {
	for _, tc := range []struct {
		name        string
		left, right string
		match       bool
	}{
		{"user_citations_empty_demoted", `{"role":"user","content":[{"type":"text","text":"hi","citations":[]}]}`, `{"role":"user","content":"hi"}`, true},
		{"user_citations_null_demoted", `{"role":"user","content":[{"type":"text","text":"hi","citations":null}]}`, `{"role":"user","content":"hi"}`, true},
		{"system_citations_empty_demoted", `{"role":"system","content":[{"type":"text","text":"note","citations":[]}]}`, `{"role":"system","content":"note"}`, true},
		{"user_citations_nonempty", `{"role":"user","content":[{"type":"text","text":"hi","citations":[{"type":"x"}]}]}`, `{"role":"user","content":"hi"}`, false},
		{"user_thinking_signature", `{"role":"user","content":[{"type":"thinking","thinking":"t","signature":"s"}]}`, `{"role":"user","content":[{"type":"thinking","thinking":"t"}]}`, true},
		{"chat_user_parts", `{"role":"user","content":[{"type":"text","text":"a","citations":[]},{"type":"text","text":"b"}]}`, `{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`, true},
		{"citations_false_kept", `{"role":"assistant","content":[{"type":"text","text":"a","citations":false}]}`, `{"role":"assistant","content":[{"type":"text","text":"a"}]}`, false},
		{"tool_input_signature_kept", `{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"x","input":{"type":"thinking","thinking":"a","signature":"s"}}]}`,
			`{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"x","input":{"type":"thinking","thinking":"a"}}]}`, false},
		{"tool_result_nested_kept", `{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"a","citations":[]}]}]}`,
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"a"}]}]}`, false},
		{"responses_output_text_kept", `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"a","citations":[]}]}`,
			`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"a"}]}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := policyDeltaDigest([]json.RawMessage{json.RawMessage(tc.left)}, 1) ==
				policyDeltaDigest([]json.RawMessage{json.RawMessage(tc.right)}, 1)
			if got != tc.match {
				t.Fatalf("%s vs %s: match %v, want %v", tc.left, tc.right, got, tc.match)
			}
		})
	}
}

// The parity oracle cleans only where _clean does: a tool's input stays
// opaque, whatever block shapes it holds.
func TestDeltaParityOracleKeepsToolInputOpaque(t *testing.T) {
	request := func(signature string) raylinearc.PolicyClientRequest {
		thinking := map[string]any{"type": "thinking", "thinking": "t"}
		if signature != "" {
			thinking["signature"] = signature
		}
		messages, _ := json.Marshal([]any{map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "tool_use", "id": "t", "name": "x", "input": map[string]any{
				"type": "tool_result", "content": []any{thinking}, "role": "user"}},
		}}})
		return raylinearc.PolicyClientRequest{Messages: messages}
	}
	if fakeCanonicalRequest(t, request("s")) == fakeCanonicalRequest(t, request("")) {
		t.Fatal("the oracle cleaned a tool's input")
	}
}

// A Workshop-shaped conversation (signed thinking, then unsigned thinking
// from another arm, each echoed as the agent rebuilds it) appends every
// reply and sends every held turn as a delta that decides on the
// conversation the full request carries.
func TestDeltaReplayParityWorkshopEcho(t *testing.T) {
	logs := captureLogs(t)
	turn1 := policyTestRequest(t, deltaUser)
	turn2 := policyTestRequest(t, deltaUser, workshopSignedEcho, workshopToolResult("toolu_1", "README.md"))
	turn3 := policyTestRequest(t, deltaUser, workshopSignedEcho, workshopToolResult("toolu_1", "README.md"),
		workshopUnsignedEcho, workshopToolResult("toolu_2", "# readme"))
	final := anthropicReply(t, []any{map[string]any{"type": "text", "text": "Done."}})
	modes := replayParity(t, []parityTurn{
		{turn1, anthropicReply(t, workshopSignedReply)},
		{turn2, anthropicReply(t, workshopUnsignedReply)},
		{turn3, final},
	})
	if !slices_Equal(modes, []string{policyDecisionModeSync, policyDecisionModeDelta, policyDecisionModeDelta}) {
		t.Fatalf("decision modes %v", modes)
	}
	appended, failed := 0, 0
	for _, entry := range logs.All() {
		switch entry.ContextMap()["event"] {
		case "rayline_arc_policy_append":
			appended++
		case "rayline_arc_policy_append_failed":
			failed++
		}
	}
	if appended != 3 || failed != 0 {
		t.Fatalf("appends %d, append_failed %d", appended, failed)
	}
}

// What the service encodes and a client changed sends the held turn in
// full, decided as the sync cell decides it.
func TestDeltaWorkshopEchoMismatchesGoInFull(t *testing.T) {
	citation := []any{map[string]any{"type": "char_location", "cited_text": "a", "document_index": 0, "start_char_index": 0, "end_char_index": 1}}
	for _, tc := range []struct {
		name  string
		reply []any
		echo  map[string]any
	}{
		{"citations_dropped",
			[]any{map[string]any{"type": "text", "text": "Listing them.", "citations": citation}, workshopSignedReply[2]},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "text", "text": "Listing them."}, workshopSignedEcho["content"].([]any)[2]}}},
		{"thinking_text_changed", workshopSignedReply, map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": "list", "signature": "sig-1"},
			workshopSignedEcho["content"].([]any)[1], workshopSignedEcho["content"].([]any)[2]}}},
		{"blocks_reordered", workshopSignedReply, map[string]any{"role": "assistant", "content": []any{
			workshopSignedEcho["content"].([]any)[0], workshopSignedEcho["content"].([]any)[2], workshopSignedEcho["content"].([]any)[1]}}},
		{"thinking_merged", []any{
			map[string]any{"type": "thinking", "thinking": "a", "signature": "s1"},
			map[string]any{"type": "thinking", "thinking": "b", "signature": "s2"},
			workshopSignedReply[2]},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "thinking", "thinking": "ab", "signature": "s1s2"}, workshopSignedEcho["content"].([]any)[2]}}},
		// E: redacted_thinking is not appended, whatever the echo.
		{"redacted_thinking", []any{
			map[string]any{"type": "thinking", "thinking": "list first", "signature": "sig-1"},
			map[string]any{"type": "redacted_thinking", "data": "x"},
			workshopSignedReply[2]},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "thinking", "thinking": "list first", "signature": "sig-1"},
				map[string]any{"type": "redacted_thinking", "data": "x"},
				workshopSignedEcho["content"].([]any)[2]}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			modes := replayParity(t, []parityTurn{
				{policyTestRequest(t, deltaUser), anthropicReply(t, tc.reply)},
				{policyTestRequest(t, deltaUser, tc.echo, workshopToolResult("toolu_1", "README.md")), deltaReply(t)},
			})
			if modes[1] != policyDecisionModeSync {
				t.Fatalf("held turn went %s, want sync", modes[1])
			}
		})
	}
}
