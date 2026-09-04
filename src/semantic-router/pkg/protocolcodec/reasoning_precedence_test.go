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
// The encoder sends both, and both are measured to be needed.
//
// On the dev cell 2026-09-03 an adaptive turn with effort high and max_tokens
// 512 was encoded with reasoning.max_tokens 1024 beside reasoning_effort high,
// and xiaomi/mimo-v2.5-pro@thinking-on spent 16,030 reasoning tokens on it.
// CP9k read that as the effort making the bound inert and sent the bound
// alone. Build 16 measured the result on 2026-09-04: the same arm spent 21,674
// reasoning tokens against a bound of 1024, and
// deepseek/deepseek-v4-flash@thinking-on spent 20,974. Neither arm obeys the
// bound at all, and the effort dial is the only control they act on, so
// removing it cost about 30 percent.
//
// The pair is also accepted. The build that sent both was answered 200, so
// "not both" describes the API rather than constraining it.
//
// So the bound stays as the cap for providers that honour it, and the effort
// stays as the brake for the ones that do not. What changed is which effort:
// a level the client stated is carried, and otherwise the level is derived
// from the bound by OpenRouter's own documented conversion, so the two
// controls cannot disagree.
func TestChatSendsTheBoundWithTheEffortBesideIt(t *testing.T) {
	engine := NewBuiltinEngine()
	body := `{"model":"m","max_tokens":512,"messages":[{"role":"user","content":"hi"}],` +
		`"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`
	wire := translateToChatWire(t, engine, body)
	if wire.Reasoning == nil || wire.Reasoning.MaxTokens == nil {
		t.Fatalf("reasoning was left unbounded: %+v", wire.Reasoning)
	}
	if wire.ReasoningEffort != "high" {
		t.Fatalf("reasoning_effort = %q, want the client's own high beside the bound of %d",
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

// A budget the client stated is a different case from one the Router derived.
// Both fields are then the client's own, the Router is only carrying them, and
// dropping one would be the Router editing a request it was asked to relay.
func TestChatCarriesBothControlsWhenTheClientSentBoth(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],` +
		`"reasoning_effort":"high","reasoning_budget_tokens":512}`
	wire := translateChatToChatWire(t, NewBuiltinEngine(), body)
	if wire.Reasoning == nil || wire.Reasoning.MaxTokens == nil || *wire.Reasoning.MaxTokens != 512 {
		t.Fatalf("the client's own budget did not travel: %+v", wire.Reasoning)
	}
	if wire.ReasoningEffort != "high" {
		t.Fatalf("reasoning_effort = %q, want the effort the client sent", wire.ReasoningEffort)
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
