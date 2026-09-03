package protocolcodec

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A request refused for a member the wire contract does not name must say
// which member. Claude Code grows its request shape with every beta it
// negotiates, so "request JSON contains a non-canonical field" on its own
// leaves nobody able to act: the body is the user's conversation and cannot be
// read back out of a log.
func TestNonCanonicalFieldRejectionNamesTheMember(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name   string
		body   string
		member string
	}{
		{
			name: "inside a named object",
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}],` +
				`"metadata":{"user_id":"u","previous_message_id":"m_1"}}`,
			member: "previous_message_id",
		},
		{
			name: "inside a tool definition",
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}],` +
				`"tools":[{"name":"t","input_schema":{"type":"object"},"eager_output_streaming":true}]}`,
			member: "eager_output_streaming",
		},
		{
			name: "inside a content block",
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user",` +
				`"content":[{"type":"text","text":"hi","provenance":"agent"}]}]}`,
			member: "provenance",
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
		`"metadata":{"user_id":"` + secret + `","previous_message_id":"` + secret + `"}}`
	_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Cause == nil {
		t.Fatalf("returned %T %v, want a protocol error with a cause", err, err)
	}
	if strings.Contains(protocolError.Cause.Error(), secret) {
		t.Fatalf("the rejection cause repeated a request value: %s", protocolError.Cause)
	}
}
