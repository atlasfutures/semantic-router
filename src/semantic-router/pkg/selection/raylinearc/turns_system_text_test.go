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

// These tests pin the two system-text scopes the normalizer separates.
//
// The conversation-opening prompt stays behind TurnOptions.IncludeSystemText.
// Mid-conversation system text is folded in by default, with
// TurnOptions.DropMidConversationSystemText as the kill switch. That is four
// option states, and every case is parsed through all four.
//
// The state with both scopes off is compared against the same request with the
// system text deleted, not against a literal. That keeps the claim "this is
// what the parser did before the scopes existed" independent of the rendering
// format. The turn count and the roles must match that request in every state,
// because the encoder wire contract admits only user and assistant and the
// serializer writes the role string into the token stream.

// systemTextCase is one request in five forms: as sent, with every piece of
// system text deleted, and the turns each of the three folding states produce.
type systemTextCase struct {
	name     string
	protocol InputProtocol
	request  string
	// stripped is the request with all system text deleted. It doubles as the
	// expectation for the legacy state and as the turn-shape reference.
	stripped string
	// wantDefault is both fields at their zero value: the opening prompt
	// dropped, mid-conversation text folded in.
	wantDefault []Turn
	// wantOpeningOn adds IncludeSystemText, so both scopes are folded in.
	wantOpeningOn []Turn
	// wantOpeningOnly sets both fields: the opening prompt folded in,
	// mid-conversation text dropped.
	wantOpeningOnly []Turn
}

// systemTextStates is the option matrix. legacy is the behaviour that shipped
// before the scopes were separated: no system text reaches the selector.
var systemTextStates = []struct {
	name    string
	options TurnOptions
}{
	{"default", TurnOptions{}},
	{"opening_on", TurnOptions{IncludeSystemText: true}},
	{"legacy", TurnOptions{DropMidConversationSystemText: true}},
	{
		"opening_only",
		TurnOptions{
			IncludeSystemText:             true,
			DropMidConversationSystemText: true,
		},
	},
}

func (test systemTextCase) want(state string) []Turn {
	switch state {
	case "default":
		return test.wantDefault
	case "opening_on":
		return test.wantOpeningOn
	default:
		return test.wantOpeningOnly
	}
}

func systemTextCases() []systemTextCase {
	return append(
		anthropicSystemTextCases(),
		append(chatSystemTextCases(), responsesSystemTextCases()...)...,
	)
}

func anthropicSystemTextCases() []systemTextCase {
	return []systemTextCase{
		// Channel 1: the Anthropic top-level system field. It is the opening
		// prompt by construction -- the API has no system role for input
		// messages, so this is the only place a standing brief can arrive.
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
			wantDefault: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "again"},
			},
			// The standing prompt lands on the first user turn only. That turn
			// is what the serializer renders as the episode's [Task] block.
			wantOpeningOn: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "again"},
			},
			wantOpeningOnly: []Turn{
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
			stripped:    `{"messages":[{"role":"user","content":"hi"}]}`,
			wantDefault: []Turn{{Role: "user", Text: "hi"}},
			wantOpeningOn: []Turn{
				{Role: "user", Text: "first\nsecond\n\nhi"},
			},
			wantOpeningOnly: []Turn{
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
			wantDefault:     []Turn{{Role: "user", Text: "hi"}},
			wantOpeningOn:   []Turn{{Role: "user", Text: "hi"}},
			wantOpeningOnly: []Turn{{Role: "user", Text: "hi"}},
		},
		// Channel 2: mid_conv_system, a system instruction relocated into the
		// message list. It renders in place, because it already sits at the
		// position it governs, and it is mid-conversation by construction.
		{
			// Both scopes in one request, which is the only place their
			// ordering is observable: the opening prompt goes in front of the
			// whole turn, the relocated instruction stays where it was written.
			name:     "anthropic_system_and_mid_conv_system",
			protocol: ProtocolAnthropicMessages,
			request: `{"system":"opening","messages":[{"role":"user","content":[
				{"type":"mid_conv_system","text":"rule"},
				{"type":"text","text":"hi"}
			]}]}`,
			stripped: `{"messages":[{"role":"user","content":[
				{"type":"text","text":"hi"}
			]}]}`,
			wantDefault:     []Turn{{Role: "user", Text: "rule\nhi"}},
			wantOpeningOn:   []Turn{{Role: "user", Text: "opening\n\nrule\nhi"}},
			wantOpeningOnly: []Turn{{Role: "user", Text: "opening\n\nhi"}},
		},
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
			wantDefault:     []Turn{{Role: "user", Text: "before\nrule\nafter"}},
			wantOpeningOn:   []Turn{{Role: "user", Text: "before\nrule\nafter"}},
			wantOpeningOnly: []Turn{{Role: "user", Text: "before\nafter"}},
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
			wantDefault:     []Turn{{Role: "user", Text: "rule\nafter"}},
			wantOpeningOn:   []Turn{{Role: "user", Text: "rule\nafter"}},
			wantOpeningOnly: []Turn{{Role: "user", Text: "after"}},
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
			wantDefault: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "rule\nyo"},
			},
			wantOpeningOn: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "rule\nyo"},
			},
			wantOpeningOnly: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "yo"},
			},
		},
	}
}

func chatSystemTextCases() []systemTextCase {
	return []systemTextCase{
		// Channel 3: the OpenAI Chat system and developer roles. Position
		// decides the scope: the leading run is the opening prompt, anything
		// after the first turn arrived mid-conversation.
		{
			name:     "chat_leading_system_role",
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
			wantDefault: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "again"},
			},
			wantOpeningOn: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "again"},
			},
			wantOpeningOnly: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "again"},
			},
		},
		{
			name:     "chat_leading_developer_role",
			protocol: ProtocolOpenAIChat,
			request: `{"messages":[
				{"role":"developer","content":[
					{"type":"text","text":"be terse"}
				]},
				{"role":"user","content":"hi"}
			]}`,
			stripped:        `{"messages":[{"role":"user","content":"hi"}]}`,
			wantDefault:     []Turn{{Role: "user", Text: "hi"}},
			wantOpeningOn:   []Turn{{Role: "user", Text: "be terse\n\nhi"}},
			wantOpeningOnly: []Turn{{Role: "user", Text: "be terse\n\nhi"}},
		},
		{
			// The boundary case. Two system messages open the request, so both
			// belong to the opening prompt however many of them there are. The
			// third one follows a turn, so it belongs to the other scope, and
			// the two scopes have to be separable in the same request.
			name:     "chat_leading_run_boundary",
			protocol: ProtocolOpenAIChat,
			request: `{"messages":[
				{"role":"system","content":"opening"},
				{"role":"developer","content":"also opening"},
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"yo"},
				{"role":"system","content":"later"},
				{"role":"user","content":"again"}
			]}`,
			stripped: `{"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"yo"},
				{"role":"user","content":"again"}
			]}`,
			wantDefault: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "later\n\nagain"},
			},
			wantOpeningOn: []Turn{
				{Role: "user", Text: "opening\n\nalso opening\n\nhi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "later\n\nagain"},
			},
			wantOpeningOnly: []Turn{
				{Role: "user", Text: "opening\n\nalso opening\n\nhi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "again"},
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
			wantDefault: []Turn{
				{Role: "user", Text: "one"},
				{Role: "assistant", Text: "ack"},
				{Role: "user", Text: "second\n\ntwo"},
			},
			wantOpeningOn: []Turn{
				{Role: "user", Text: "first\n\none"},
				{Role: "assistant", Text: "ack"},
				{Role: "user", Text: "second\n\ntwo"},
			},
			wantOpeningOnly: []Turn{
				{Role: "user", Text: "first\n\none"},
				{Role: "assistant", Text: "ack"},
				{Role: "user", Text: "two"},
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
			wantDefault: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "late\n\nyo"},
			},
			wantOpeningOn: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "late\n\nyo"},
			},
			wantOpeningOnly: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "yo"},
			},
		},
	}
}

func responsesSystemTextCases() []systemTextCase {
	return []systemTextCase{
		// Channel 4: the OpenAI Responses system and developer message roles,
		// scoped by position exactly as the Chat roles are.
		{
			name:     "responses_leading_system_message",
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
			wantDefault: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "yo"},
			},
			wantOpeningOn: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
			},
			wantOpeningOnly: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
			},
		},
		{
			// The same leading-run boundary as the Chat case, on the item
			// array. Both parsers have to draw the line in the same place.
			name:     "responses_leading_run_boundary",
			protocol: ProtocolOpenAIResponses,
			request: `{"input":[
				{"type":"message","role":"system","content":"opening"},
				{"type":"message","role":"developer","content":"also opening"},
				{"type":"message","role":"user","content":"hi"},
				{"type":"message","role":"assistant","content":"yo"},
				{"type":"message","role":"system","content":"later"},
				{"type":"message","role":"user","content":"again"}
			]}`,
			stripped: `{"input":[
				{"type":"message","role":"user","content":"hi"},
				{"type":"message","role":"assistant","content":"yo"},
				{"type":"message","role":"user","content":"again"}
			]}`,
			wantDefault: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "later\n\nagain"},
			},
			wantOpeningOn: []Turn{
				{Role: "user", Text: "opening\n\nalso opening\n\nhi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "later\n\nagain"},
			},
			wantOpeningOnly: []Turn{
				{Role: "user", Text: "opening\n\nalso opening\n\nhi"},
				{Role: "assistant", Text: "yo"},
				{Role: "user", Text: "again"},
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
			wantDefault:     []Turn{{Role: "user", Text: "one\nrule\n\ntwo"}},
			wantOpeningOn:   []Turn{{Role: "user", Text: "one\nrule\n\ntwo"}},
			wantOpeningOnly: []Turn{{Role: "user", Text: "one\ntwo"}},
		},
		// Channel 5: the Responses top-level instructions field. Like the
		// Anthropic system field, it is the opening prompt by construction and
		// nothing read it before.
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
			wantDefault: []Turn{
				{Role: "user", Text: "hi"},
				{Role: "assistant", Text: "yo"},
			},
			wantOpeningOn: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
			},
			wantOpeningOnly: []Turn{
				{Role: "user", Text: "be terse\n\nhi"},
				{Role: "assistant", Text: "yo"},
			},
		},
		{
			name:            "responses_instructions_with_string_input",
			protocol:        ProtocolOpenAIResponses,
			request:         `{"instructions":"be terse","input":"hi"}`,
			stripped:        `{"input":"hi"}`,
			wantDefault:     []Turn{{Role: "user", Text: "hi"}},
			wantOpeningOn:   []Turn{{Role: "user", Text: "be terse\n\nhi"}},
			wantOpeningOnly: []Turn{{Role: "user", Text: "be terse\n\nhi"}},
		},
	}
}

// TestSystemTextLegacyStateMatchesRequestWithoutIt is the regression pin. With
// the opening prompt off and mid-conversation text dropped, the parser has to
// reproduce what it did before either scope existed.
func TestSystemTextLegacyStateMatchesRequestWithoutIt(t *testing.T) {
	options := TurnOptions{DropMidConversationSystemText: true}
	for _, test := range systemTextCases() {
		t.Run(test.name, func(t *testing.T) {
			want, err := NormalizeTurns(
				test.protocol,
				[]byte(test.stripped),
				options,
			)
			if err != nil {
				t.Fatalf("stripped request failed to normalize: %v", err)
			}
			got, err := NormalizeTurns(
				test.protocol,
				[]byte(test.request),
				options,
			)
			if err != nil {
				t.Fatalf("request failed to normalize: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf(
					"system text leaked with both scopes off: got %#v, want %#v",
					got,
					want,
				)
			}
		})
	}
}

// TestSystemTextFoldsPerScope walks the whole option matrix. The default state
// is the one that changed: mid-conversation text now reaches the selector while
// the opening prompt still does not.
func TestSystemTextFoldsPerScope(t *testing.T) {
	for _, state := range systemTextStates {
		if state.name == "legacy" {
			// Pinned against the stripped request instead of a literal.
			continue
		}
		for _, test := range systemTextCases() {
			t.Run(state.name+"/"+test.name, func(t *testing.T) {
				got, err := NormalizeTurns(
					test.protocol,
					[]byte(test.request),
					state.options,
				)
				if err != nil {
					t.Fatalf("request failed to normalize: %v", err)
				}
				want := test.want(state.name)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("turns = %#v, want %#v", got, want)
				}
			})
		}
	}
}

// TestSystemTextKeepsTheTurnShapeInEveryState guards the constraint that made
// folding the text into an existing turn the only option available. The encoder
// wire schema pins ArcTurn.role to user or assistant, and the serializer writes
// both the role string and the turn number into the token stream, so no option
// state may change the turn count or the roles. Only the text may differ, and
// the request with its system text deleted is the reference for all four.
func TestSystemTextKeepsTheTurnShapeInEveryState(t *testing.T) {
	for _, test := range systemTextCases() {
		t.Run(test.name, func(t *testing.T) {
			reference, err := NormalizeTurns(
				test.protocol,
				[]byte(test.stripped),
				TurnOptions{},
			)
			if err != nil {
				t.Fatalf("stripped request failed to normalize: %v", err)
			}
			for _, state := range systemTextStates {
				got, stateErr := NormalizeTurns(
					test.protocol,
					[]byte(test.request),
					state.options,
				)
				if stateErr != nil {
					t.Fatalf("%s failed to normalize: %v", state.name, stateErr)
				}
				if len(got) != len(reference) {
					t.Fatalf(
						"%s turn count changed: got %d, want %d",
						state.name,
						len(got),
						len(reference),
					)
				}
				for index := range got {
					if got[index].Role != reference[index].Role {
						t.Fatalf(
							"%s turn %d role changed: got %q, want %q",
							state.name,
							index,
							got[index].Role,
							reference[index].Role,
						)
					}
				}
			}
		})
	}
}

// malformedSystemCase is a request whose system content cannot be read.
type malformedSystemCase struct {
	name     string
	protocol InputProtocol
	request  string
}

// TestMalformedOpeningSystemTextOnlyFailsWhenAskedFor pins the deliberate
// ordering in the opening-prompt branches: the content is parsed only when the
// operator has asked for it. Off, a malformed system message must not start
// failing requests that succeed today. On, the operator asked for that text, so
// a request that cannot supply it fails rather than routing a conversation the
// parser could not read.
func TestMalformedOpeningSystemTextOnlyFailsWhenAskedFor(t *testing.T) {
	malformed := []malformedSystemCase{
		{
			"anthropic_system_field",
			ProtocolAnthropicMessages,
			`{"system":42,"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			"chat_leading_system_content",
			ProtocolOpenAIChat,
			`{"messages":[
				{"role":"system","content":42},
				{"role":"user","content":"hi"}
			]}`,
		},
		{
			"responses_leading_system_content",
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
			for _, state := range systemTextStates {
				got, err := NormalizeTurns(
					test.protocol,
					[]byte(test.request),
					state.options,
				)
				if !state.options.IncludeSystemText {
					if err != nil {
						t.Fatalf(
							"%s must ignore malformed system text: %v",
							state.name,
							err,
						)
					}
					if !reflect.DeepEqual(got, want) {
						t.Fatalf(
							"%s turns = %#v, want %#v",
							state.name,
							got,
							want,
						)
					}
					continue
				}
				code := TurnNormalizationErrorCode(err)
				if code != "invalid_field" {
					t.Fatalf(
						"%s error code = %q (%v), want invalid_field",
						state.name,
						code,
						err,
					)
				}
			}
		})
	}
}

// TestMalformedMidConversationSystemTextNeverFails is the robustness pin for
// the scope that is now on by default. Mid-conversation system content is
// parsed on every request, so it must not be able to fail one that routes
// today: unreadable content folds nothing and the episode continues. That is
// exactly what happened before the scope was included, when the same text was
// discarded without being looked at.
func TestMalformedMidConversationSystemTextNeverFails(t *testing.T) {
	malformed := []malformedSystemCase{
		{
			"chat_system_content_not_readable",
			ProtocolOpenAIChat,
			`{"messages":[
				{"role":"user","content":"hi"},
				{"role":"system","content":42}
			]}`,
		},
		{
			"chat_developer_unknown_content_block",
			ProtocolOpenAIChat,
			`{"messages":[
				{"role":"user","content":"hi"},
				{"role":"developer","content":[{"type":"not_a_real_type"}]}
			]}`,
		},
		{
			"responses_system_content_not_readable",
			ProtocolOpenAIResponses,
			`{"input":[
				{"type":"message","role":"user","content":"hi"},
				{"type":"message","role":"system","content":42}
			]}`,
		},
		{
			"anthropic_mid_conv_system_content_not_readable",
			ProtocolAnthropicMessages,
			`{"messages":[{"role":"user","content":[
				{"type":"mid_conv_system","content":42},
				{"type":"text","text":"hi"}
			]}]}`,
		},
		{
			"anthropic_mid_conv_system_unknown_block",
			ProtocolAnthropicMessages,
			`{"messages":[{"role":"user","content":[
				{"type":"mid_conv_system","content":[{"type":"image"}]},
				{"type":"text","text":"hi"}
			]}]}`,
		},
	}
	want := []Turn{{Role: "user", Text: "hi"}}
	for _, test := range malformed {
		t.Run(test.name, func(t *testing.T) {
			for _, state := range systemTextStates {
				got, err := NormalizeTurns(
					test.protocol,
					[]byte(test.request),
					state.options,
				)
				if err != nil {
					t.Fatalf(
						"%s must not fail on malformed mid-conversation "+
							"system text: %v",
						state.name,
						err,
					)
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("%s turns = %#v, want %#v", state.name, got, want)
				}
			}
		})
	}
}

// TestMidConvSystemStaysUnknownSafe holds the fail-closed default in place. The
// mid-conversation scope forgives content it cannot read inside a system block,
// because that text was discarded outright until now. It must not forgive an
// unrecognised block sitting next to one: that is ordinary conversation the
// parser cannot render, and showing a trained selector a silently shortened
// conversation is still worse than refusing to route.
func TestMidConvSystemStaysUnknownSafe(t *testing.T) {
	request := `{"messages":[{"role":"user","content":[
		{"type":"mid_conv_system","text":"rule"},
		{"type":"not_a_real_block"}
	]}]}`
	for _, state := range systemTextStates {
		_, err := NormalizeTurns(
			ProtocolAnthropicMessages,
			[]byte(request),
			state.options,
		)
		if code := TurnNormalizationErrorCode(err); code != "unknown_item" {
			t.Fatalf(
				"%s error code = %q (%v), want unknown_item",
				state.name,
				code,
				err,
			)
		}
	}
}
