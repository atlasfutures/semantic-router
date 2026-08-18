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
	"reflect"
	"testing"
)

// These tests pin TurnOptions.IncludeSystemText, the option that folds
// system-prompt text into the user turn it governs.
//
// Every case is parsed twice. Off, the result must equal the same request with
// the system text deleted: the default has to leave behaviour exactly as it
// shipped, and comparing two parser outputs rather than a literal keeps that
// claim independent of the rendering format. On, the result must carry the
// text and change nothing else. The turn count and the roles stay put, because
// the encoder wire contract admits only user and assistant and the serializer
// writes the role string into the token stream.

// systemTextCase is one request in three forms: with the system text, without
// it, and the turns the option is expected to produce.
type systemTextCase struct {
	name     string
	protocol InputProtocol
	request  string
	stripped string
	wantOn   []Turn
}

func systemTextCases() []systemTextCase {
	return []systemTextCase{
		// Channel 1: the Anthropic top-level system field. Nothing read it
		// before, so it was never dropped at the parser -- it was never seen.
		{
			name:     "anthropic_system_string",
			protocol: ProtocolAnthropicMessages,
			request: `{"system":"be terse","messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"yo"},
				{"role":"user","content":"again"}
			]}`,
			stripped: `{"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"yo"},
				{"role":"user","content":"again"}
			]}`,
			// The standing prompt lands on the first user turn only. That turn
			// is what the serializer renders as the episode's [Task] block.
			wantOn: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "again"},
			},
		},
		{
			name:     "anthropic_system_blocks",
			protocol: ProtocolAnthropicMessages,
			request: `{"system":[
				{"type":"text","text":"first"},
				{"type":"text","text":"second"}
			],"messages":[{"role":"user","content":"hi"}]}`,
			stripped: `{"messages":[{"role":"user","content":"hi"}]}`,
			wantOn: []Turn{
				{Role: "user", Text: "first\nsecond\n\nhi"},
			},
		},
		{
			name:     "anthropic_system_empty_string",
			protocol: ProtocolAnthropicMessages,
			request: `{"system":"","messages":[
				{"role":"user","content":"hi"}
			]}`,
			stripped: `{"messages":[{"role":"user","content":"hi"}]}`,
			// Empty system text must not add a separator to the turn it would
			// have governed.
			wantOn: []Turn{{Role: "user", Text: "hi"}},
		},
		// Channel 2: mid_conv_system, the system prompt relocated into the
		// message list. It renders in place, because it already sits at the
		// position it governs.
		{
			name:     "anthropic_mid_conv_system_content",
			protocol: ProtocolAnthropicMessages,
			request: `{"messages":[{"role":"user","content":[
				{"type":"text","text":"before"},
				{"type":"mid_conv_system","content":[
					{"type":"text","text":"rule"}
				]},
				{"type":"text","text":"after"}
			]}]}`,
			stripped: `{"messages":[{"role":"user","content":[
				{"type":"text","text":"before"},
				{"type":"text","text":"after"}
			]}]}`,
			wantOn: []Turn{
				{Role: "user", Text: "before\nrule\nafter"},
			},
		},
		{
			name:     "anthropic_mid_conv_system_text",
			protocol: ProtocolAnthropicMessages,
			request: `{"messages":[{"role":"user","content":[
				{"type":"mid_conv_system","text":"rule"},
				{"type":"text","text":"after"}
			]}]}`,
			stripped: `{"messages":[{"role":"user","content":[
				{"type":"text","text":"after"}
			]}]}`,
			wantOn: []Turn{{Role: "user", Text: "rule\nafter"}},
		},
		{
			name:     "anthropic_mid_conv_system_in_assistant",
			protocol: ProtocolAnthropicMessages,
			request: `{"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":[
					{"type":"mid_conv_system","text":"rule"},
					{"type":"text","text":"yo"}
				]}
			]}`,
			stripped: `{"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":[
					{"type":"text","text":"yo"}
				]}
			]}`,
			wantOn: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "rule\nyo"},
			},
		},
		// Channel 3: the OpenAI Chat system and developer roles.
		{
			name:     "chat_system_role",
			protocol: ProtocolOpenAIChat,
			request: `{"messages":[
				{"role":"system","content":"be terse"},
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"yo"},
				{"role":"user","content":"again"}
			]}`,
			stripped: `{"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"yo"},
				{"role":"user","content":"again"}
			]}`,
			wantOn: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "again"},
			},
		},
		{
			name:     "chat_developer_role",
			protocol: ProtocolOpenAIChat,
			request: `{"messages":[
				{"role":"developer","content":[
					{"type":"text","text":"be terse"}
				]},
				{"role":"user","content":"hi"}
			]}`,
			stripped: `{"messages":[{"role":"user","content":"hi"}]}`,
			wantOn: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
			},
		},
		{
			// Each stretch of system text lands on the next user turn, which
			// is the turn it governs. A later instruction must not be pulled
			// forward onto an earlier request.
			name:     "chat_two_system_stretches",
			protocol: ProtocolOpenAIChat,
			request: `{"messages":[
				{"role":"system","content":"first"},
				{"role":"user","content":"one"},
				{"role":"assistant","content":"ack"},
				{"role":"system","content":"second"},
				{"role":"user","content":"two"}
			]}`,
			stripped: `{"messages":[
				{"role":"user","content":"one"},
				{"role":"assistant","content":"ack"},
				{"role":"user","content":"two"}
			]}`,
			wantOn: []Turn{
				{Role: "user", Text: "first\n\none"},
				{Role: "assistant", Text: "ack"},
				{Role: "user", Text: "second\n\ntwo"},
			},
		},
		{
			// Two system messages before one user turn join into that turn,
			// in the order they were sent.
			name:     "chat_adjacent_system_messages",
			protocol: ProtocolOpenAIChat,
			request: `{"messages":[
				{"role":"system","content":"first"},
				{"role":"developer","content":"second"},
				{"role":"user","content":"hi"}
			]}`,
			stripped: `{"messages":[{"role":"user","content":"hi"}]}`,
			wantOn: []Turn{
				{Role: "user", Text: "first\n\nsecond\n\nhi"},
			},
		},
		{
			// A system message with no user turn after it still governs the
			// reply being routed, so it attaches to the nearest turn instead
			// of being discarded.
			name:     "chat_trailing_system",
			protocol: ProtocolOpenAIChat,
			request: `{"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"yo"},
				{"role":"system","content":"late"}
			]}`,
			stripped: `{"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"yo"}
			]}`,
			wantOn: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "late\n\nyo"},
			},
		},
		// Channel 4: the OpenAI Responses system and developer message roles.
		{
			name:     "responses_system_message",
			protocol: ProtocolOpenAIResponses,
			request: `{"input":[
				{"type":"message","role":"system","content":"be terse"},
				{"type":"message","role":"user","content":"hi"},
				{"type":"message","role":"assistant","content":"yo"}
			]}`,
			stripped: `{"input":[
				{"type":"message","role":"user","content":"hi"},
				{"type":"message","role":"assistant","content":"yo"}
			]}`,
			wantOn: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
			},
		},
		{
			// The system item sits between two user items that coalesce into
			// one turn. Dropping it must not split the turn, and folding it in
			// must not either.
			name:     "responses_developer_between_user_items",
			protocol: ProtocolOpenAIResponses,
			request: `{"input":[
				{"type":"message","role":"user","content":"one"},
				{"type":"message","role":"developer","content":"rule"},
				{"type":"message","role":"user","content":"two"}
			]}`,
			stripped: `{"input":[
				{"type":"message","role":"user","content":"one"},
				{"type":"message","role":"user","content":"two"}
			]}`,
			wantOn: []Turn{
				{Role: "user", Text: "one\nrule\n\ntwo"},
			},
		},
		// Channel 5: the Responses top-level instructions field. Like the
		// Anthropic system field, nothing read it before.
		{
			name:     "responses_instructions_with_items",
			protocol: ProtocolOpenAIResponses,
			request: `{"instructions":"be terse","input":[
				{"type":"message","role":"user","content":"hi"},
				{"type":"message","role":"assistant","content":"yo"}
			]}`,
			stripped: `{"input":[
				{"type":"message","role":"user","content":"hi"},
				{"type":"message","role":"assistant","content":"yo"}
			]}`,
			wantOn: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
			},
		},
		{
			name:     "responses_instructions_with_string_input",
			protocol: ProtocolOpenAIResponses,
			request:  `{"instructions":"be terse","input":"hi"}`,
			stripped: `{"input":"hi"}`,
			wantOn: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
			},
		},
	}
}

// TestSystemTextOffMatchesRequestWithoutIt is the regression pin. The default
// has to reproduce what the parser did before the option existed.
func TestSystemTextOffMatchesRequestWithoutIt(t *testing.T) {
	for _, test := range systemTextCases() {
		t.Run(test.name, func(t *testing.T) {
			want, err := NormalizeTurns(
				test.protocol,
				[]byte(test.stripped),
				TurnOptions{},
			)
			if err != nil {
				t.Fatalf("stripped request failed to normalize: %v", err)
			}
			got, err := NormalizeTurns(
				test.protocol,
				[]byte(test.request),
				TurnOptions{},
			)
			if err != nil {
				t.Fatalf("request failed to normalize: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf(
					"system text leaked with the option off: got %#v, want %#v",
					got,
					want,
				)
			}
		})
	}
}

// TestSystemTextOnFoldsIntoTheTurnItGoverns is the behaviour the option buys.
func TestSystemTextOnFoldsIntoTheTurnItGoverns(t *testing.T) {
	for _, test := range systemTextCases() {
		t.Run(test.name, func(t *testing.T) {
			got, err := NormalizeTurns(
				test.protocol,
				[]byte(test.request),
				TurnOptions{IncludeSystemText: true},
			)
			if err != nil {
				t.Fatalf("request failed to normalize: %v", err)
			}
			if !reflect.DeepEqual(got, test.wantOn) {
				t.Fatalf("turns = %#v, want %#v", got, test.wantOn)
			}
		})
	}
}

// TestSystemTextOnKeepsTheTurnShape guards the constraint that made folding the
// text into an existing turn the only option available. The encoder wire schema
// pins ArcTurn.role to user or assistant, and the serializer writes both the
// role string and the turn number into the token stream, so a request must
// produce the same turn count and the same roles either way. Only the text may
// differ.
func TestSystemTextOnKeepsTheTurnShape(t *testing.T) {
	for _, test := range systemTextCases() {
		t.Run(test.name, func(t *testing.T) {
			off, err := NormalizeTurns(
				test.protocol,
				[]byte(test.request),
				TurnOptions{},
			)
			if err != nil {
				t.Fatalf("option off failed to normalize: %v", err)
			}
			on, err := NormalizeTurns(
				test.protocol,
				[]byte(test.request),
				TurnOptions{IncludeSystemText: true},
			)
			if err != nil {
				t.Fatalf("option on failed to normalize: %v", err)
			}
			if len(on) != len(off) {
				t.Fatalf(
					"turn count changed: on %d, off %d",
					len(on),
					len(off),
				)
			}
			for index := range on {
				if on[index].Role != off[index].Role {
					t.Fatalf(
						"turn %d role changed: on %q, off %q",
						index,
						on[index].Role,
						off[index].Role,
					)
				}
			}
		})
	}
}

// TestSystemTextOffDoesNotReadSystemContent pins the deliberate ordering in the
// system branches: the content is parsed only when it is wanted. A malformed
// system message must not start failing requests that succeed today, because
// that would be a behaviour change the option is supposed to gate.
func TestSystemTextOffDoesNotReadSystemContent(t *testing.T) {
	malformed := []struct {
		name     string
		protocol InputProtocol
		request  string
	}{
		{
			"anthropic_system_field",
			ProtocolAnthropicMessages,
			`{"system":42,"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			"chat_system_content",
			ProtocolOpenAIChat,
			`{"messages":[
				{"role":"system","content":42},
				{"role":"user","content":"hi"}
			]}`,
		},
		{
			"responses_system_content",
			ProtocolOpenAIResponses,
			`{"input":[
				{"type":"message","role":"system","content":42},
				{"type":"message","role":"user","content":"hi"}
			]}`,
		},
		{
			"responses_instructions",
			ProtocolOpenAIResponses,
			`{"instructions":42,"input":"hi"}`,
		},
	}
	want := []Turn{{Role: "user", Text: "hi"}}
	for _, test := range malformed {
		t.Run(test.name, func(t *testing.T) {
			got, err := NormalizeTurns(
				test.protocol,
				[]byte(test.request),
				TurnOptions{},
			)
			if err != nil {
				t.Fatalf("option off must ignore malformed system: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("turns = %#v, want %#v", got, want)
			}
			// With the option on the same request is read, so it must fail
			// rather than route a conversation the parser could not read.
			_, err = NormalizeTurns(
				test.protocol,
				[]byte(test.request),
				TurnOptions{IncludeSystemText: true},
			)
			if code := TurnNormalizationErrorCode(err); code != "invalid_field" {
				t.Fatalf(
					"error code = %q (%v), want invalid_field",
					code,
					err,
				)
			}
		})
	}
}

// TestMidConvSystemStaysUnknownSafe holds the fail-closed default in place. The
// option widens what one known block type contributes; it must not turn an
// unreadable block into a silently shortened conversation.
func TestMidConvSystemStaysUnknownSafe(t *testing.T) {
	request := `{"messages":[{"role":"user","content":[
		{"type":"mid_conv_system","content":[{"type":"image"}]},
		{"type":"text","text":"hi"}
	]}]}`
	_, err := NormalizeTurns(
		ProtocolAnthropicMessages,
		[]byte(request),
		TurnOptions{IncludeSystemText: true},
	)
	if code := TurnNormalizationErrorCode(err); code != "unknown_item" {
		t.Fatalf("error code = %q (%v), want unknown_item", code, err)
	}
}
