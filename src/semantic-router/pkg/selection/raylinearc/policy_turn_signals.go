/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"unicode"
)

// The detectors below read what a harness itself writes on the wire. Each is
// a positive match on a literal the harness emits, ported from pathfinder's
// collection-side rules (sandbox/wire_compaction.py and
// sandbox/request_checkpoint_gate.py), which were captured from real traffic.
// None of them infers anything from a transcript change, and a harness that
// stops writing a literal shows up as the signal no longer firing, never as a
// request misread as something else.

// ClaudeCodeContinuationMarker is Claude Code's preamble on the first message
// of the first request after it compacts.
const ClaudeCodeContinuationMarker = "This session is being continued from a previous conversation that ran out of context."

const (
	systemReminderOpen    = "<system-reminder>"
	billingHeaderPrefix   = "x-anthropic-billing-header:"
	billingSubagentKey    = "cc_is_subagent"
	codexCompactionPrompt = "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary " +
		"for another LLM that will resume the task.\n\nInclude:\n- Current progress and " +
		"key decisions made\n- Important context, constraints, or user preferences\n- " +
		"What remains to be done (clear next steps)\n- Any critical data, examples, or " +
		"references needed to continue\n\nBe concise, structured, and focused on helping " +
		"the next LLM seamlessly continue the work.\n"
)

// ClaudeCodeCompactionSummaryDigest returns the sha256 of the continuation
// summary when this Anthropic Messages transcript is a compacted context, and
// "" otherwise.
//
// The marker must open the first real text block of the first message, which
// must be a user message; injected <system-reminder> blocks ahead of it are
// skipped. A later message, or a block that merely quotes the marker, is
// transcript content.
func ClaudeCodeCompactionSummaryDigest(messages []json.RawMessage) string {
	if len(messages) == 0 {
		return ""
	}
	var first struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(messages[0], &first) != nil {
		return ""
	}
	if first.Role != "" && first.Role != "user" {
		return ""
	}
	for _, text := range textBlocks(first.Content) {
		trimmed := strings.TrimLeftFunc(text, unicode.IsSpace)
		if strings.HasPrefix(trimmed, systemReminderOpen) {
			continue
		}
		if !strings.HasPrefix(trimmed, ClaudeCodeContinuationMarker) {
			return ""
		}
		digest := sha256.Sum256([]byte(text))
		return hex.EncodeToString(digest[:])
	}
	return ""
}

// ClaudeCodeSubagentClaim reports whether Claude Code declared this request a
// subagent's, in the cc_is_subagent field of its billing system block. Only
// an explicit "true" is a claim; Claude Code omits the key on main-stream
// turns, and an absent or unrecognised value is not a subagent.
func ClaudeCodeSubagentClaim(system json.RawMessage) bool {
	text := billingHeaderText(system)
	if text == "" {
		return false
	}
	for _, field := range strings.Split(text[len(billingHeaderPrefix):], ";") {
		key, value, _ := strings.Cut(field, "=")
		if strings.TrimSpace(key) == billingSubagentKey {
			return strings.EqualFold(strings.TrimSpace(value), "true")
		}
	}
	return false
}

// IsCodexCompactionRequest reports whether this Responses input is codex's
// summarization call: its last item is a user message whose whole text is
// codex's compaction directive. A prefix match or a directive that is not
// last is not.
func IsCodexCompactionRequest(input []json.RawMessage) bool {
	if len(input) == 0 {
		return false
	}
	var item struct {
		Type    string          `json:"type"`
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(input[len(input)-1], &item) != nil ||
		item.Type != "message" || item.Role != "user" {
		return false
	}
	var whole string
	if json.Unmarshal(item.Content, &whole) == nil {
		return whole == codexCompactionPrompt
	}
	var parts []struct {
		Type string  `json:"type"`
		Text *string `json:"text"`
	}
	if json.Unmarshal(item.Content, &parts) != nil || len(parts) != 1 ||
		parts[0].Type != "input_text" || parts[0].Text == nil {
		return false
	}
	return *parts[0].Text == codexCompactionPrompt
}

// textBlocks returns the text of a message content: the string itself, or
// every text block whose text is a string.
func textBlocks(content json.RawMessage) []string {
	var whole string
	if json.Unmarshal(content, &whole) == nil {
		return []string{whole}
	}
	var blocks []struct {
		Type string  `json:"type"`
		Text *string `json:"text"`
	}
	if json.Unmarshal(content, &blocks) != nil {
		return nil
	}
	texts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if block.Type == "text" && block.Text != nil {
			texts = append(texts, *block.Text)
		}
	}
	return texts
}

func billingHeaderText(system json.RawMessage) string {
	var whole string
	if json.Unmarshal(system, &whole) == nil {
		if strings.HasPrefix(whole, billingHeaderPrefix) {
			return whole
		}
		return ""
	}
	var blocks []json.RawMessage
	if json.Unmarshal(system, &blocks) != nil {
		return ""
	}
	for _, raw := range blocks {
		var text string
		if json.Unmarshal(raw, &text) != nil {
			var block struct {
				Text *string `json:"text"`
			}
			if json.Unmarshal(raw, &block) != nil || block.Text == nil {
				continue
			}
			text = *block.Text
		}
		if strings.HasPrefix(text, billingHeaderPrefix) {
			return text
		}
	}
	return ""
}
