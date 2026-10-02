package raylinearc

import (
	"encoding/json"
	"testing"
)

func digestOf(t *testing.T, messages ...string) string {
	t.Helper()
	raw := make([]json.RawMessage, len(messages))
	for index, message := range messages {
		raw[index] = json.RawMessage(message)
	}
	return MessagesDigest(raw, len(raw))
}

// The prefix rule shared with the training side's extends (pathfinder#3214):
// equal conversations compare equal however the client re-encodes them, and
// any real change still breaks the prefix.
func TestMessagesDigestComparesTheCanonicalConversation(t *testing.T) {
	const reminder = "<system-reminder>files changed</system-reminder>"
	cases := []struct {
		name      string
		before    string
		after     string
		wantEqual bool
	}{
		{
			"Claude Code demotes a system message from one block to a string",
			`{"role":"system","content":[{"type":"text","text":"` + reminder + `","cache_control":{"type":"ephemeral"}}]}`,
			`{"role":"system","content":"` + reminder + `"}`,
			true,
		},
		{
			"a cache breakpoint moves off a message",
			`{"role":"user","content":[{"type":"text","text":"go on","cache_control":{"type":"ephemeral"}}]}`,
			`{"role":"user","content":[{"type":"text","text":"go on"}]}`,
			true,
		},
		{
			"key order and whitespace",
			`{"role":"user","content":[{"type":"text","text":"a < b"}]}`,
			`{ "content" : [ { "text" : "a < b" , "type" : "text" } ] , "role" : "user" }`,
			true,
		},
		{
			"a system message whose text changed",
			`{"role":"system","content":[{"type":"text","text":"` + reminder + `"}]}`,
			`{"role":"system","content":"<system-reminder>other files</system-reminder>"}`,
			false,
		},
		{
			"trailing whitespace is text, not formatting",
			`{"role":"system","content":[{"type":"text","text":"x"}]}`,
			`{"role":"system","content":"x "}`,
			false,
		},
		{
			"a block with another member stays a list",
			`{"role":"system","content":[{"type":"text","text":"x","citations":[]}]}`,
			`{"role":"system","content":"x"}`,
			false,
		},
		{
			"two blocks never equal their joined string",
			`{"role":"system","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`,
			`{"role":"system","content":"ab"}`,
			false,
		},
		{
			"only system messages are demoted",
			`{"role":"user","content":[{"type":"text","text":"x"}]}`,
			`{"role":"user","content":"x"}`,
			false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if equal := digestOf(t, tc.before) == digestOf(t, tc.after); equal != tc.wantEqual {
				t.Fatalf("equal = %v, want %v", equal, tc.wantEqual)
			}
		})
	}
}

// End to end at the episode level: a turn after Claude Code demoted a system
// message keeps its epoch and attribution instead of starting a new context.
func TestPolicyTurnExtendsThroughSystemMessageDemotion(t *testing.T) {
	asLast := `{"role":"system","content":[{"type":"text","text":"<system-reminder>r</system-reminder>","cache_control":{"type":"ephemeral"}}]}`
	asString := `{"role":"system","content":"<system-reminder>r</system-reminder>"}`
	first, roles := rawMessages(t, user1, asLast)
	turn, _, _ := PolicyTurn(nil, first, roles, 0, PolicyTurnSignals{})
	committed := turn.Next(first, "action-a", "arm-a")

	second, roles := rawMessages(t, user1, asString, asst1, user2)
	turn, attribution, transition := PolicyTurn(committed, second, roles, 1, PolicyTurnSignals{})
	if transition != PolicyTransitionExtends || turn.Epoch != 0 || len(attribution) != 1 || attribution[0].Message != 2 {
		t.Fatalf("after demotion: transition %s epoch %d attribution %+v, want an extension attributing message 2",
			transition, turn.Epoch, attribution)
	}
}
