package protocolcodec

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// eager_input_streaming rides on 11 of the 12 tools Claude Code forwards in
// proxy mode. It asks the provider to stream a tool's input as it is generated
// instead of buffering and validating it first; the tool chosen, the input it
// finally holds and the effect of the call are the same either way. Refusing a
// turn over it loses the conversation to buy nothing, so it is dropped and
// counted instead.
func TestEagerToolInputStreamingIsDroppedNotRefused(t *testing.T) {
	body := `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hello"}],` +
		`"tools":[{"name":"lookup","input_schema":{"type":"object"},"eager_input_streaming":true}]}`
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(body), nil,
	)
	if err != nil {
		t.Fatalf("a tool asking for eager input streaming was refused: %v", err)
	}
	assertDiagnosticField(t, result.Diagnostics, "tools.eager_input_streaming")
}

// The members beside it take the same route. defer_loading decides whether a
// tool enters the context window, input_examples is few-shot guidance, and
// allowed_callers says who may call the tool. Each one changes what the model
// does, so the Router must not silently discard it -- but refusing the turn
// discards the whole conversation, which is worse. The tool wire struct names
// none of them, so all three ride the carrier: a Messages destination gets
// them back unchanged, and a destination that cannot express them counts each
// one by name.
func TestSemanticToolMembersAreCarriedNotRefused(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, member := range []string{"defer_loading", "input_examples", "allowed_callers"} {
		t.Run(member, func(t *testing.T) {
			body := []byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hello"}],` +
				`"tools":[{"name":"lookup","input_schema":{"type":"object"},"` + member + `":true}]}`)
			request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
			if err != nil {
				t.Fatalf("a tool declaring %q was refused: %v", member, err)
			}
			// A fresh envelope forces a real encode rather than a replay of the
			// source body, which would pass without carrying anything.
			encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, llmprotocol.Envelope{})
			if err != nil {
				t.Fatalf("the Messages target refused the carried tool: %v", err)
			}
			if !strings.Contains(string(encoded.Body), `"`+member+`":true`) {
				t.Fatalf("the Messages target dropped %q: %s", member, encoded.Body)
			}
			result, err := engine.TranslateRequest(
				llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil,
			)
			if err != nil {
				t.Fatalf("the Chat target refused a tool declaring %q: %v", member, err)
			}
			if strings.Contains(string(result.Body), member) {
				t.Fatalf("%q reached a wire format that cannot name it: %s", member, result.Body)
			}
			assertDroppedDiagnosticField(t, result.Diagnostics, "tools."+member)
		})
	}
}
