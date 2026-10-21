package protocolcodec

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// withoutThinkingEditsWhenThinkingOff drops the client's clear_thinking_*
// context-management edits from a Messages request the Router sends with
// thinking disabled. Claude Code sends clear_thinking_20251015 beside adaptive
// thinking; a thinking-off arm turns thinking off, and Anthropic refuses that
// edit without thinking, which OpenRouter relays as 400 "Invalid Anthropic
// Messages API request" (prod 2026-10-21, every claude-sonnet-5.5 thinking-off
// turn). The edit only clears thinking blocks, so a turn without thinking loses
// nothing by its absence. The other edits stay; context_management goes when
// none remain. The carrier is copied, never changed in place.
func withoutThinkingEditsWhenThinkingOff(
	request llmprotocol.Request,
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
) llmprotocol.Request {
	carrier := request.Unmodeled
	if request.ReasoningMode != llmprotocol.ReasoningModeDisabled || carrier.Empty() ||
		carrier.Format != llmprotocol.AnthropicMessagesV1 {
		return request
	}
	raw, ok := carrier.Fields["context_management"]
	if !ok {
		return request
	}
	var management map[string]json.RawMessage
	if json.Unmarshal(raw, &management) != nil {
		return request
	}
	var edits []json.RawMessage
	if json.Unmarshal(management["edits"], &edits) != nil {
		return request
	}
	kept := make([]json.RawMessage, 0, len(edits))
	for _, edit := range edits {
		var head struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(edit, &head) == nil && strings.HasPrefix(head.Type, "clear_thinking_") {
			continue
		}
		kept = append(kept, edit)
	}
	if len(kept) == len(edits) {
		return request
	}
	copied := *carrier
	copied.Fields = make(map[string]json.RawMessage, len(carrier.Fields))
	for name, value := range carrier.Fields {
		copied.Fields[name] = value
	}
	if len(kept) == 0 && len(management) == 1 {
		delete(copied.Fields, "context_management")
	} else {
		encodedEdits, err := json.Marshal(kept)
		if err != nil {
			return request
		}
		management["edits"] = encodedEdits
		encoded, err := json.Marshal(management)
		if err != nil {
			return request
		}
		copied.Fields["context_management"] = encoded
	}
	request.Unmodeled = &copied
	appendProviderFieldOmission(diagnostics, policy, request.Trusted.SourceFormat,
		"context_management.edits[clear_thinking]", "Messages refuses a clear_thinking edit when thinking is disabled")
	return request
}
