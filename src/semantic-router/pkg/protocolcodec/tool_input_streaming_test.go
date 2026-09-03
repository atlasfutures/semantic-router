package protocolcodec

import (
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

// The members beside it stay refused. defer_loading decides whether a tool
// enters the context window at all, and input_examples is few-shot guidance
// that changes the call the model writes. Dropping either quietly changes what
// the model does, which is not a loss the Router may take on a caller's behalf.
func TestSemanticToolMembersAreStillRefused(t *testing.T) {
	engine := NewBuiltinEngine()
	for member, code := range map[string]string{
		"defer_loading":   "unsupported_tools_defer_loading",
		"input_examples":  "unsupported_tools_input_examples",
		"allowed_callers": "unsupported_tools_allowed_callers",
	} {
		t.Run(member, func(t *testing.T) {
			body := `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hello"}],` +
				`"tools":[{"name":"lookup","input_schema":{"type":"object"},"` + member + `":true}]}`
			_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, code)
		})
	}
}
