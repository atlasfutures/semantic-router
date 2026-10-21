package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A thinking-off arm must not forward Claude Code's clear_thinking edit:
// Anthropic refuses it without thinking (prod 2026-10-21, claude-sonnet-5.5).
func TestAnthropicThinkingOffDropsClearThinkingEdits(t *testing.T) {
	const prefix = `{"model":"source-model","max_tokens":2048,"messages":[{"role":"user","content":"hello"}],` +
		`"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},`
	tests := []struct {
		name       string
		management string
		mode       llmprotocol.ReasoningMode
		want       string // context_management as sent; "" means absent
	}{
		{
			name:       "only edit is clear_thinking",
			management: `{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}`,
			mode:       llmprotocol.ReasoningModeDisabled,
			want:       "",
		},
		{
			name:       "other edits stay",
			management: `{"edits":[{"type":"clear_thinking_20251015","keep":"all"},{"type":"clear_tool_uses_20250919"}]}`,
			mode:       llmprotocol.ReasoningModeDisabled,
			want:       `{"edits":[{"type":"clear_tool_uses_20250919"}]}`,
		},
		{
			name:       "adaptive keeps the edit",
			management: `{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}`,
			mode:       llmprotocol.ReasoningModeAdaptive,
			want:       `{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}`,
		},
	}
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1,
				[]byte(prefix+`"context_management":`+test.management+`}`))
			if err != nil {
				t.Fatal(err)
			}
			request.Model = "routed-model"
			request.Generation++
			if test.mode == llmprotocol.ReasoningModeDisabled {
				request.ReasoningMode, request.ReasoningEffort = llmprotocol.ReasoningModeDisabled, ""
			}
			encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			var sent map[string]json.RawMessage
			if err := json.Unmarshal(encoded.Body, &sent); err != nil {
				t.Fatal(err)
			}
			got, present := sent["context_management"]
			if test.want == "" {
				if present {
					t.Fatalf("context_management = %s, want absent", got)
				}
				return
			}
			if !present || !jsonEqual(t, got, json.RawMessage(test.want)) {
				t.Fatalf("context_management = %s, want %s", got, test.want)
			}
		})
	}
}
