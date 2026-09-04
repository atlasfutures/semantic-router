package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code sends adaptive thinking together with an effort level on every
// turn. Anthropic documents that pair as the way to control thinking depth on
// models that have dropped budget_tokens, so the Router must route it rather
// than refuse it. The body below is the shape captured from claude-cli 2.1.259
// against the dev cell on 2026-09-03.
const claudeCodeAdaptiveEffortBody = `{` +
	`"model":"rayline-router",` +
	`"messages":[{"role":"user","content":"hello"}],` +
	`"max_tokens":32000,` +
	`"thinking":{"type":"adaptive","display":"omitted"},` +
	`"output_config":{"effort":"high"},` +
	`"stream":true}`

func TestClaudeCodeAdaptiveThinkingWithEffortReachesTheChatLeg(t *testing.T) {
	engine := NewBuiltinEngine()
	request, _, _, err := engine.DecodeRequest(
		llmprotocol.AnthropicMessagesV1, []byte(claudeCodeAdaptiveEffortBody),
	)
	if err != nil {
		t.Fatalf("Claude Code's default body was refused at decode: %v", err)
	}
	if request.ReasoningMode != llmprotocol.ReasoningModeAdaptive ||
		request.ReasoningEffort != "high" || request.ReasoningDisplay != "omitted" {
		t.Fatalf(
			"decoded reasoning = mode %q effort %q display %q",
			request.ReasoningMode, request.ReasoningEffort, request.ReasoningDisplay,
		)
	}

	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		[]byte(claudeCodeAdaptiveEffortBody), nil,
	)
	if err != nil {
		t.Fatalf("Claude Code's default body was refused on the Chat leg: %v", err)
	}
	var wire chatRequestWire
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	// Adaptive is the Chat default -- the model decides -- and it states no
	// budget, so the Router derives one from the output allowance and that
	// bound is the single reasoning control the request carries. OpenRouter
	// takes an effort level or a bound, "One of the following (not both)", and
	// the effort level beside a derived bound is what made the bound inert on
	// the dev cell.
	if wire.Reasoning == nil || wire.Reasoning.MaxTokens == nil {
		t.Fatalf("Claude Code's default body reached the Chat leg unbounded: %+v", wire.Reasoning)
	}
	if wire.ReasoningEffort != "" {
		t.Fatalf("Chat reasoning_effort = %q, want the derived bound to travel alone", wire.ReasoningEffort)
	}
	// The display preference has no Chat control. The turn still routes and
	// the drop is recorded rather than refused.
	assertDiagnosticField(t, result.Diagnostics, "reasoning_display")
}

// Both display values are dropped and counted, never refused.
func TestChatRecordsTheReasoningDisplayItCannotCarry(t *testing.T) {
	body := `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hello"}],` +
		`"thinking":{"type":"adaptive","display":"summarized"}}`
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(body), nil,
	)
	if err != nil {
		t.Fatalf("summarized reasoning display was refused on the Chat leg: %v", err)
	}
	assertDiagnosticField(t, result.Diagnostics, "reasoning_display")
}

// thinking-off must reach the provider as an explicit off-signal, not as
// silence: a provider that reasons by default would otherwise still reason.
func TestChatCarriesDisabledThinkingAsAnExplicitOffSignal(t *testing.T) {
	body := `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hello"}],` +
		`"thinking":{"type":"disabled"},"output_config":{"effort":"high"}}`
	engine := NewBuiltinEngine()
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
	if wire.ReasoningEffort != "none" {
		t.Fatalf("Chat reasoning_effort = %q, want none", wire.ReasoningEffort)
	}
}

// thinking.type disabled with an effort level is the other shape Claude Code
// sends. Anthropic answered it 200 on 2026-09-03 for claude-sonnet-5, so the
// neutral contract must not treat effort as a contradiction there either.
func TestAnthropicDisabledThinkingWithEffortIsRoutable(t *testing.T) {
	body := `{"model":"m","max_tokens":64000,"messages":[{"role":"user","content":"hello"}],` +
		`"thinking":{"type":"disabled"},"output_config":{"effort":"high"}}`
	engine := NewBuiltinEngine()
	request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err != nil {
		t.Fatalf("disabled thinking with an effort level was refused: %v", err)
	}
	if request.ReasoningMode != llmprotocol.ReasoningModeDisabled || request.ReasoningEffort != "high" {
		t.Fatalf("decoded reasoning = mode %q effort %q", request.ReasoningMode, request.ReasoningEffort)
	}
	if _, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(body), nil,
	); err != nil {
		t.Fatalf("disabled thinking with an effort level was refused on the Chat leg: %v", err)
	}
}

// A token budget remains a contradiction for both modes: neither variant of
// Anthropic's thinking object carries budget_tokens.
func TestReasoningBudgetStillContradictsAdaptiveAndDisabledModes(t *testing.T) {
	limits := llmprotocol.DefaultPolicy().Limits
	for _, mode := range []llmprotocol.ReasoningMode{
		llmprotocol.ReasoningModeAdaptive, llmprotocol.ReasoningModeDisabled,
	} {
		request := minimalNeutralRequest()
		request.ReasoningMode = mode
		request.ReasoningBudgetTokens = llmprotocol.Int64(512)
		if err := llmprotocol.ValidateRequest(request, limits); err == nil {
			t.Fatalf("%s reasoning with a token budget was accepted", mode)
		}
	}
}

// An adaptive request that goes back out as Anthropic keeps both controls.
func TestAdaptiveThinkingWithEffortRoundTripsToAnthropic(t *testing.T) {
	engine := NewBuiltinEngine()
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		[]byte(claudeCodeAdaptiveEffortBody), func(request *llmprotocol.Request) error {
			request.Model = "routed-model"
			request.Generation++
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	var wire anthropicRequestWire
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Thinking == nil || wire.Thinking.Type != "adaptive" || wire.Thinking.Display != "omitted" {
		t.Fatalf("re-encoded thinking = %+v", wire.Thinking)
	}
	if wire.OutputConfig == nil || wire.OutputConfig.Effort != "high" {
		t.Fatalf("re-encoded output_config = %+v", wire.OutputConfig)
	}
}

func minimalNeutralRequest() llmprotocol.Request {
	return llmprotocol.Request{
		Model: "m",
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}},
		}},
	}
}

func assertDiagnosticField(t *testing.T, diagnostics llmprotocol.Diagnostics, field string) {
	t.Helper()
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == field {
			return
		}
	}
	t.Fatalf("diagnostics %+v do not record %q", diagnostics, field)
}
