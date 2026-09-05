package protocolcodec

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Rule 6: every refusal accept-by-default leaves standing names its block, its
// index and its field path, in the log line and in the client error body.
//
// Both, because they are read by different parties for different reasons. The
// cause reaches ingress_request_refused, and an operator needs it because the
// refused body is the user's conversation and is never stored. The message
// reaches the client, and a client needs it because that is all a client
// error body carries: the 2.1.260 Claude Code retries a request without a beta
// field only when the 400 body names that field. A refusal that names the
// field only in the log makes the client's own repair impossible.

func encodedRefusalBody(t *testing.T, format llmprotocol.WireFormat, err error) string {
	t.Helper()
	protocolError, ok := err.(*llmprotocol.ProtocolError)
	if !ok {
		t.Fatalf("error = %v, want a protocol error", err)
	}
	body, encodeErr := NewBuiltinEngine().EncodeError(format, protocolError)
	if encodeErr != nil {
		t.Fatalf("EncodeError() error = %v", encodeErr)
	}
	// The assertions below read the message the way a person does. A field
	// path is quoted inside a JSON string, so its quotes arrive escaped.
	return strings.ReplaceAll(string(body), `\"`, `"`)
}

func TestRefusalBodiesNameTheFieldOnEveryClientFormat(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, test := range []struct {
		name  string
		body  string
		want  []string
		cause []string
	}{
		{
			name: "content union variant",
			body: `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[` +
				`{"type":"text","text":"look"},` +
				`{"type":"text","text":"hi","source":{"type":"base64","media_type":"image/png","data":"aW1n"}}]}]}`,
			want:  []string{"content block 1", `type "text"`, `"content.source"`},
			cause: []string{"content block 1", `"content.source"`},
		},
		{
			name: "content missing a required member",
			body: `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[` +
				`{"type":"text"}]}]}`,
			want:  []string{"content block 0", `type "text"`, `"content.text"`},
			cause: []string{"content block 0", `"content.text"`},
		},
		{
			name: "top-level member the contract does not support",
			body: `{"model":"m","max_tokens":16,"service_tier":"priority",` +
				`"messages":[{"role":"user","content":"hi"}]}`,
			want:  []string{`"service_tier"`},
			cause: []string{`"service_tier"`},
		},
		{
			name: "tool description over the limit",
			body: `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],` +
				`"tools":[{"name":"lookup","input_schema":{"type":"object"},"description":"` +
				strings.Repeat("x", 1<<21) + `"}]}`,
			want:  []string{`tool 0 named "lookup"`, `"tools.description"`, "limit"},
			cause: []string{`tool 0 named "lookup"`, `"tools.description"`},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequestForMutation(
				llmprotocol.AnthropicMessagesV1, []byte(test.body),
			)
			if err == nil {
				t.Fatal("the request was accepted")
			}
			protocolError, ok := err.(*llmprotocol.ProtocolError)
			if !ok {
				t.Fatalf("error = %v, want a protocol error", err)
			}
			if protocolError.Cause == nil {
				t.Fatal("refusal carries no cause, so the log line names nothing")
			}
			for _, want := range test.cause {
				if !strings.Contains(protocolError.Cause.Error(), want) {
					t.Fatalf("cause %q does not carry %s", protocolError.Cause, want)
				}
			}
			for _, format := range []llmprotocol.WireFormat{
				llmprotocol.AnthropicMessagesV1,
				llmprotocol.OpenAIChatV1,
				llmprotocol.OpenAIResponsesV1,
			} {
				encoded := encodedRefusalBody(t, format, err)
				for _, want := range test.want {
					if !strings.Contains(encoded, want) {
						t.Fatalf("%s error body %s does not carry %s", format, encoded, want)
					}
				}
			}
		})
	}
}

// A refusal body may name the contract and may not quote the conversation. The
// field path and the block index are contract text; the text a user wrote is
// not, and it must not leave the cell in an error.
func TestRefusalBodiesCarryNoRequestValues(t *testing.T) {
	engine := NewBuiltinEngine()
	secret := "the-user-wrote-this"
	body := `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[` +
		`{"type":"text","text":"` + secret + `","source":{"type":"base64","media_type":"image/png","data":"aW1n"}}]}]}`
	_, _, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err == nil {
		t.Fatal("the request was accepted")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("refusal %q carries a request value", err)
	}
	for _, format := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
	} {
		if encoded := encodedRefusalBody(t, format, err); strings.Contains(encoded, secret) {
			t.Fatalf("%s error body carries a request value: %s", format, encoded)
		}
	}
}

// The OpenAI wire formats have a param member for exactly this. A client that
// parses the body rather than the prose gets the field path from there.
func TestRefusalsSetTheFieldAsTheErrorParameter(t *testing.T) {
	engine := NewBuiltinEngine()
	body := `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text"}]}]}`
	_, _, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body))
	protocolError, ok := err.(*llmprotocol.ProtocolError)
	if !ok {
		t.Fatalf("error = %v, want a protocol error", err)
	}
	if protocolError.Parameter != "content.text" {
		t.Fatalf("parameter = %q, want content.text", protocolError.Parameter)
	}
	for _, format := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
	} {
		if encoded := encodedRefusalBody(t, format, err); !strings.Contains(encoded, `"param"`) {
			t.Fatalf("%s error body states no param: %s", format, encoded)
		}
	}
}
