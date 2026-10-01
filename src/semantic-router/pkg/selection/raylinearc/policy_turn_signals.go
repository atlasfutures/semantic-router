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

// ClaudeCodeSummarizeDirective opens the text block Claude Code appends to the
// last user message of its compaction (summarization) call. The whole block is
// several kilobytes of instructions; its opening line is the stable literal.
// Captured from Claude Code 2.1.238 and 2.1.280, byte-identical in both.
const ClaudeCodeSummarizeDirective = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools."

// ClaudeCodeTitlePrompt opens the system block of Claude Code's
// session-title call, as captured from Claude Code 2.1.238 and 2.1.280.
const ClaudeCodeTitlePrompt = "You are naming a coding session so the user can pick it out of a long list of sessions."

// CodexSummaryPrefix opens the summary message codex puts at the end of the
// replacement history after it compacts (codex 0.154.0); the summary follows.
const CodexSummaryPrefix = "Another language model started to solve this problem and produced a summary " +
	"of its thinking process. You also have access to the state of the tools that " +
	"were used by that language model. Use this to build on the work that has " +
	"already been done and avoid duplicating work. Here is the summary produced by " +
	"the other language model, use the information in this summary to assist with " +
	"your own analysis:\n"

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
	if json.Unmarshal(input[len(input)-1], &item) != nil {
		return false
	}
	text, ok := responsesUserText(item.Type, item.Role, item.Content)
	return ok && text == codexCompactionPrompt
}

// IsClaudeCodeCompactionRequest reports whether this Anthropic Messages
// transcript is Claude Code's summarization call: the last message is a user
// message whose last block is a text block opening with the summarize
// directive. The directive anywhere else is transcript content.
func IsClaudeCodeCompactionRequest(messages []json.RawMessage) bool {
	if len(messages) == 0 {
		return false
	}
	var last struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(messages[len(messages)-1], &last) != nil || last.Role != "user" {
		return false
	}
	var whole string
	if json.Unmarshal(last.Content, &whole) == nil {
		return strings.HasPrefix(whole, ClaudeCodeSummarizeDirective)
	}
	var blocks []struct {
		Type string  `json:"type"`
		Text *string `json:"text"`
	}
	if json.Unmarshal(last.Content, &blocks) != nil || len(blocks) == 0 {
		return false
	}
	final := blocks[len(blocks)-1]
	return final.Type == "text" && final.Text != nil && strings.HasPrefix(*final.Text, ClaudeCodeSummarizeDirective)
}

// IsClaudeCodeTitleRequest reports whether a system block of this Anthropic
// Messages request opens with Claude Code's session-title prompt.
func IsClaudeCodeTitleRequest(system json.RawMessage) bool {
	for _, text := range systemTexts(system) {
		if strings.HasPrefix(text, ClaudeCodeTitlePrompt) {
			return true
		}
	}
	return false
}

// responsesModelItemTypes are the Responses input items that are model
// output. A codex summary message is the compacted base only while none
// precedes it.
var responsesModelItemTypes = map[string]bool{
	"reasoning": true, "function_call": true, "custom_tool_call": true,
	"function_call_output": true, "custom_tool_call_output": true,
}

// CodexCompactionSummaryDigest returns the sha256 of codex's current
// compaction summary when this Responses input is a compacted context, and ""
// otherwise. Ported from pathfinder's wire rule: the compacted base is the
// initial context and user messages, then one user message opening with
// CodexSummaryPrefix. A summary counts only while no model output precedes
// it, and the last one in that leading run is the current one (a later
// compaction keeps the earlier summary among the user messages it retains).
func CodexCompactionSummaryDigest(input []json.RawMessage) string {
	summary, found := "", false
	for _, raw := range input {
		var item struct {
			Type    string          `json:"type"`
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(raw, &item) != nil {
			return ""
		}
		if responsesModelItemTypes[item.Type] || item.Type == "message" && item.Role == "assistant" {
			break
		}
		if text, ok := responsesUserText(item.Type, item.Role, item.Content); ok &&
			strings.HasPrefix(text, CodexSummaryPrefix) {
			summary, found = text, true
		}
	}
	if !found {
		return ""
	}
	digest := sha256.Sum256([]byte(summary))
	return hex.EncodeToString(digest[:])
}

// responsesUserText is the text of a Responses user message whose content is
// a string or exactly one input_text part.
func responsesUserText(itemType string, role string, content json.RawMessage) (string, bool) {
	if itemType != "message" || role != "user" {
		return "", false
	}
	var whole string
	if json.Unmarshal(content, &whole) == nil {
		return whole, true
	}
	var parts []struct {
		Type string  `json:"type"`
		Text *string `json:"text"`
	}
	if json.Unmarshal(content, &parts) != nil || len(parts) != 1 ||
		parts[0].Type != "input_text" || parts[0].Text == nil {
		return "", false
	}
	return *parts[0].Text, true
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
	for _, text := range systemTexts(system) {
		if strings.HasPrefix(text, billingHeaderPrefix) {
			return text
		}
	}
	return ""
}

// systemTexts is the text of an Anthropic system field: the string itself, or
// each block that is a string or carries a string text.
func systemTexts(system json.RawMessage) []string {
	var whole string
	if json.Unmarshal(system, &whole) == nil {
		return []string{whole}
	}
	var blocks []json.RawMessage
	if json.Unmarshal(system, &blocks) != nil {
		return nil
	}
	texts := make([]string, 0, len(blocks))
	for _, raw := range blocks {
		var text string
		if json.Unmarshal(raw, &text) == nil {
			texts = append(texts, text)
			continue
		}
		var block struct {
			Text *string `json:"text"`
		}
		if json.Unmarshal(raw, &block) == nil && block.Text != nil {
			texts = append(texts, *block.Text)
		}
	}
	return texts
}
