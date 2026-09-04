package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// max_tokens does not bound a reasoning model. The encoder writes it as
// max_completion_tokens, and the models behind the thinking-on arms do not
// count reasoning tokens against that. Measured on the dev cell 2026-09-03:
// max_tokens 512 produced 37,213 completion tokens, and max_tokens 1 produced
// 21,299. The same requests stop at exactly 512 with stop_reason max_tokens on
// the thinking-off arms. Unbounded, a thinking turn then runs to the 630 s
// platform deadline, which is where item 5's truncation comes from.
//
// OpenRouter bounds reasoning with its own control, which the encoder never
// sent: reasoning.max_tokens, "Specific token limit (Anthropic-style)".
// Read 2026-09-03: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
func TestChatBoundsReasoningTokens(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		maxTokens int64
		effort    string
	}{
		{
			// A stated budget is the client's own number. Send it verbatim.
			name: "thinking budget is sent verbatim",
			body: `{"model":"m","max_tokens":4096,"messages":[{"role":"user","content":"hi"}],` +
				`"thinking":{"type":"enabled","budget_tokens":2048}}`,
			maxTokens: 2048,
		},
		{
			// Adaptive states no budget, so reasoning is bounded by what the
			// client allowed for output. The effort level does not travel
			// beside a bound the Router derived: OpenRouter takes one control
			// or the other, and an effort level there makes the bound inert.
			name: "adaptive is bounded by the output allowance",
			body: `{"model":"m","max_tokens":4096,"messages":[{"role":"user","content":"hi"}],` +
				`"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`,
			maxTokens: 4096,
		},
		{
			// A small output allowance still gets a usable reasoning budget:
			// below 1024 no documented reasoning API accepts one at all.
			name: "a small allowance is floored",
			body: `{"model":"m","max_tokens":512,"messages":[{"role":"user","content":"hi"}],` +
				`"thinking":{"type":"adaptive"}}`,
			maxTokens: 1024,
		},
		{
			// An effort level on its own is still a request to reason, and the
			// bound derived from it replaces it.
			name: "an effort level alone is bounded",
			body: `{"model":"m","max_tokens":8192,"messages":[{"role":"user","content":"hi"}],` +
				`"output_config":{"effort":"medium"}}`,
			maxTokens: 8192,
		},
	}
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			wire := translateToChatWire(t, engine, test.body)
			if wire.Reasoning == nil || wire.Reasoning.MaxTokens == nil {
				t.Fatalf("reasoning was left unbounded: %+v", wire.Reasoning)
			}
			if *wire.Reasoning.MaxTokens != test.maxTokens {
				t.Fatalf("reasoning.max_tokens = %d, want %d", *wire.Reasoning.MaxTokens, test.maxTokens)
			}
			if wire.ReasoningEffort != test.effort {
				t.Fatalf("reasoning_effort = %q, want %q", wire.ReasoningEffort, test.effort)
			}
		})
	}
}

// A turn that asks for no reasoning gets no reasoning control. Sending a
// budget to a thinking-off arm would ask it to start reasoning.
func TestChatSendsNoReasoningBoundWhenNoneIsAsked(t *testing.T) {
	engine := NewBuiltinEngine()
	for name, body := range map[string]string{
		"no thinking at all": `{"model":"m","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`,
		"thinking disabled": `{"model":"m","max_tokens":4096,"messages":[{"role":"user","content":"hi"}],` +
			`"thinking":{"type":"disabled"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if wire := translateToChatWire(t, engine, body); wire.Reasoning != nil {
				t.Fatalf("an unasked-for reasoning bound was sent: %+v", wire.Reasoning)
			}
		})
	}
}

// Without an output allowance there is nothing to derive a bound from, and
// inventing one would cap a client that asked for no cap.
func TestChatDerivesNoBoundWithoutAnOutputAllowance(t *testing.T) {
	engine := NewBuiltinEngine()
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`
	result, err := engine.TranslateRequest(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(body),
		func(request *llmprotocol.Request) error { request.Model = "routed"; return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	var wire chatRequestWire
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Reasoning != nil {
		t.Fatalf("a bound was invented with no output allowance: %+v", wire.Reasoning)
	}
}

func translateToChatWire(t *testing.T, engine *Engine, body string) chatRequestWire {
	t.Helper()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(body), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	var wire chatRequestWire
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	return wire
}
