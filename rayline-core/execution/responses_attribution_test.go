package execution

import (
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

func TestResponsesAttributesFirstOutputItemAtExactInputIndex(t *testing.T) {
	roles, err := wireMessageRoles(llmprotocol.OpenAIResponsesV1, []byte(`{"input":[{"role":"developer","content":"rules"},{"role":"user","content":"read"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	state := &branchState{}
	state.recordAttribution(llmprotocol.OpenAIResponsesV1, roles, Decision{ActionID: "action", Worker: "local", RouteID: "route", Result: "top"})
	for _, output := range []string{
		`{"type":"message","role":"assistant","content":[]}`,
		`{"type":"function_call","name":"read","call_id":"c","arguments":"{}"}`,
		`{"type":"reasoning","summary":[]}`,
	} {
		body := []byte(`{"input":[{"role":"developer","content":"rules"},{"role":"user","content":"read"},` + output + `,{"role":"user","content":"next"}]}`)
		grown, err := wireMessageRoles(llmprotocol.OpenAIResponsesV1, body)
		if err != nil {
			t.Fatal(err)
		}
		got := state.evidenceForRequest("instance", llmprotocol.OpenAIResponsesV1, grown).Attribution
		if len(got) != 1 || got[0].Message != 2 {
			t.Fatalf("output was not attributed at exact input index: %+v", got)
		}
	}
	roles, err = wireMessageRoles(llmprotocol.OpenAIResponsesV1, []byte(`{"input":[{"role":"developer","content":"rules"},{"role":"user","content":"read"},{"type":"function_call_output","call_id":"c","output":"result"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(state.evidenceForRequest("instance", llmprotocol.OpenAIResponsesV1, roles).Attribution) != 0 {
		t.Fatal("tool output was attributed as assistant output")
	}
}

func TestResponsesStringInputHasExplicitlyUnknownAttribution(t *testing.T) {
	roles, err := wireMessageRoles(llmprotocol.OpenAIResponsesV1, []byte(`{"input":"hello"}`))
	if err != nil {
		t.Fatal(err)
	}
	state := &branchState{}
	state.recordAttribution(llmprotocol.OpenAIResponsesV1, roles, Decision{ActionID: "action"})
	if len(state.ledger) != 0 {
		t.Fatal("array position was invented for a string input")
	}
}
