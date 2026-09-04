package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func translateChatToChatWire(t *testing.T, engine *Engine, body string) chatRequestWire {
	t.Helper()
	// The mutator makes this a routed turn, so the encoder runs rather than
	// the original body passing through untouched.
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
	return wire
}

// OpenRouter's reasoning control accepts an effort level or a token bound, and
// says which of the two a request may carry:
//
//	"// One of the following (not both):"
//	"\"effort\": \"high\", // Can be \"max\", \"xhigh\", \"high\", \"medium\", \"low\",
//	 \"minimal\" or \"none\""
//	"\"max_tokens\": 2000, // Specific token limit (Anthropic-style)"
//
// Read 2026-09-04:
// https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
//
// "Not both" is enforced. A top-level reasoning_effort is folded into the same
// object, so a body carrying the bound and an effort is refused with
//
//	Only one of "reasoning.effort" and "reasoning.max_tokens" can be specified
//
// Measured on the dev cell 2026-09-04: that 400 answered every thinking-arm
// turn on build 17, which sent the pair. CP9l read an earlier 200 as the pair
// being accepted; the earlier build's bound happened to be the whole allowance,
// which is a different shape from the one that fails.
//
// So the bound travels alone. It is the number that stops the turn, and an
// effort level the client stated cannot be stated beside it.
func TestChatSendsTheBoundWithoutAnEffortBesideIt(t *testing.T) {
	engine := NewBuiltinEngine()
	body := `{"model":"m","max_tokens":512,"messages":[{"role":"user","content":"hi"}],` +
		`"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`
	wire := translateToChatWire(t, engine, body)
	if wire.Reasoning == nil || wire.Reasoning.MaxTokens == nil {
		t.Fatalf("reasoning was left unbounded: %+v", wire.Reasoning)
	}
	if wire.ReasoningEffort != "" {
		t.Fatalf("reasoning_effort = %q travelled beside the bound of %d, which OpenRouter refuses",
			wire.ReasoningEffort, *wire.Reasoning.MaxTokens)
	}
}

// The effort dial is the only control a turn with no output allowance can
// carry, because there is nothing to derive a bound from. Dropping it there
// would leave the request saying nothing about reasoning at all.
func TestChatKeepsTheEffortDialWhenNoBoundApplies(t *testing.T) {
	// Chat ingress, because an Anthropic request without max_tokens is refused
	// before it reaches the encoder.
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`
	wire := translateChatToChatWire(t, NewBuiltinEngine(), body)
	if wire.Reasoning != nil {
		t.Fatalf("a bound was invented with no output allowance: %+v", wire.Reasoning)
	}
	if wire.ReasoningEffort != "high" {
		t.Fatalf("reasoning_effort = %q, want the effort the client asked for", wire.ReasoningEffort)
	}
}

// A client that sends both controls still gets one on the wire. Relaying the
// pair verbatim would be relaying a request OpenRouter refuses, so the bound
// travels and the effort does not: the bound is the client's own number too,
// and it is the one that stops the turn.
func TestChatSendsOnlyTheBoundWhenTheClientSentBoth(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],` +
		`"reasoning_effort":"high","reasoning_budget_tokens":512}`
	wire := translateChatToChatWire(t, NewBuiltinEngine(), body)
	if wire.Reasoning == nil || wire.Reasoning.MaxTokens == nil || *wire.Reasoning.MaxTokens != 512 {
		t.Fatalf("the client's own budget did not travel: %+v", wire.Reasoning)
	}
	if wire.ReasoningEffort != "" {
		t.Fatalf("reasoning_effort = %q travelled beside the client's own budget", wire.ReasoningEffort)
	}
}

// Thinking-off is not an effort dial that a bound replaces. It is the explicit
// off-signal, it comes with no bound, and it has to survive.
func TestChatKeepsTheOffSignal(t *testing.T) {
	engine := NewBuiltinEngine()
	body := `{"model":"m","max_tokens":4096,"messages":[{"role":"user","content":"hi"}],` +
		`"thinking":{"type":"disabled"}}`
	wire := translateToChatWire(t, engine, body)
	if wire.Reasoning != nil {
		t.Fatalf("a bound was sent to a turn that asked for no reasoning: %+v", wire.Reasoning)
	}
	if wire.ReasoningEffort != "none" {
		t.Fatalf("reasoning_effort = %q, want the off-signal", wire.ReasoningEffort)
	}
}
