package protocolcodec

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A refused request must say which member caused the refusal. Claude Code
// grows its request shape with every beta it negotiates, so "request JSON
// contains a non-canonical field" on its own leaves nobody able to act: the
// body is the user's conversation and cannot be read back out of a log.
//
// A member the wire contract does not name is now carried rather than refused,
// so the naming rule is pinned on the refusals that remain. A spelling that
// differs from a declared member only in case is one: Go's decoder matches it
// case-insensitively, so accepting it would write a value into a member under
// a spelling no contract states. A key repeated inside one object is the
// other.
func TestNonCanonicalFieldRejectionNamesTheMember(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name   string
		body   string
		member string
	}{
		{
			name: "case-folded member inside a named object",
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}],` +
				`"metadata":{"User_id":"u"}}`,
			member: "User_id",
		},
		{
			name: "case-folded member inside a tool definition",
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}],` +
				`"tools":[{"name":"t","Input_schema":{"type":"object"}}]}`,
			member: "Input_schema",
		},
		{
			name: "case-folded member inside a content block",
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user",` +
				`"content":[{"type":"text","TEXT":"hi"}]}]}`,
			member: "TEXT",
		},
		{
			name: "duplicate key inside a named object",
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}],` +
				`"metadata":{"user_id":"u","user_id":"v"}}`,
			member: "user_id",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(test.body))
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) {
				t.Fatalf("returned %T %v, want a protocol error", err, err)
			}
			if protocolError.Cause == nil {
				t.Fatalf("%s left no cause, so nothing can name the member", protocolError.Code)
			}
			if !strings.Contains(protocolError.Cause.Error(), test.member) {
				t.Fatalf("cause %q does not name %q", protocolError.Cause, test.member)
			}
		})
	}
}

// The cause carries member names only. The values beside them are the user's
// conversation and must never reach a log line.
func TestNonCanonicalFieldCauseCarriesNoRequestValues(t *testing.T) {
	engine := NewBuiltinEngine()
	secret := "a-user-utterance-that-must-not-be-logged"
	body := `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"` + secret + `"}],` +
		`"metadata":{"User_id":"` + secret + `"}}`
	_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Cause == nil {
		t.Fatalf("returned %T %v, want a protocol error with a cause", err, err)
	}
	if strings.Contains(protocolError.Cause.Error(), secret) {
		t.Fatalf("the rejection cause repeated a request value: %s", protocolError.Cause)
	}
}

// A member that belongs to another variant of a discriminated union is still
// refused. The discriminator names the variant, and a member of a different
// one contradicts it rather than extending it, so the request has no single
// reading the Router could route.
func TestUnionVariantMemberIsStillRefused(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name   string
		choice string
	}{
		{name: "auto with a tool name", choice: `{"type":"auto","name":"lookup"}`},
		{name: "none with a parallel control", choice: `{"type":"none","disable_parallel_tool_use":true}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],` +
				`"tool_choice":` + test.choice + `}`
			_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != "invalid_tool_choice_variant" {
				t.Fatalf("returned %T %v, want invalid_tool_choice_variant", err, err)
			}
		})
	}
}
