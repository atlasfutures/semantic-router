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
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"
)

// PolicyPrefixShape is a diagnostic summary of one request's prefix: the
// digest PolicyTurn compares, a short digest of each message, and short
// digests of its system prompt and tool definitions. It explains a
// prefix_break (where the transcript stopped extending the recorded prefix)
// and an encoder rebuild under an extending transcript (the system or the
// tools changed, which the message digest does not cover). It never decides
// anything and is never persisted.
type PolicyPrefixShape struct {
	Len      int
	Digest   string
	Messages []uint64
	System   uint64
	Tools    uint64
	// Tail is the block layout of the last PolicyPrefixShapeTail messages
	// (PolicyMessageLayout), so a divergence near the end of the transcript,
	// where a client rewrites what it sent, can be described on both sides.
	Tail []PolicyMessageLayout
}

// PolicyPrefixShapeTail is how many trailing messages a shape keeps the
// layout of.
const PolicyPrefixShapeTail = 8

// MaxPolicyPrefixShapeMessages bounds the per-message digests a shape keeps;
// a longer transcript keeps its first ones, and a divergence past them is
// reported unknown.
const MaxPolicyPrefixShapeMessages = 1024

// NewPolicyPrefixShape summarizes messages (each in the canonical form
// PolicyTurn compares) with the request's system prompt and tools.
func NewPolicyPrefixShape(messages []json.RawMessage, system, tools json.RawMessage) *PolicyPrefixShape {
	return NewPolicyPrefixShapeFrom(NewPrefixDigests(messages), system, tools)
}

// NewPolicyPrefixShapeFrom is NewPolicyPrefixShape over a request's memoized
// digests.
func NewPolicyPrefixShapeFrom(digests *PrefixDigests, system, tools json.RawMessage) *PolicyPrefixShape {
	count := len(digests.Messages())
	kept := min(count, MaxPolicyPrefixShapeMessages)
	shape := &PolicyPrefixShape{
		Len:      count,
		Digest:   digests.Digest(count),
		Messages: make([]uint64, kept),
		System:   prefixShapeDigest(canonicalSystem(system)),
		Tools:    prefixShapeDigest(canonicalTools(tools)),
	}
	for index := range kept {
		shape.Messages[index] = prefixShapeDigest(digests.Canonical(index))
	}
	for index := max(0, count-PolicyPrefixShapeTail); index < count; index++ {
		shape.Tail = append(shape.Tail, NewPolicyMessageLayout(digests.Messages()[index]))
	}
	return shape
}

// Layout is the layout of message index, when the shape kept it.
func (shape *PolicyPrefixShape) Layout(index int) (PolicyMessageLayout, bool) {
	start := shape.Len - len(shape.Tail)
	if index < start || index >= shape.Len {
		return PolicyMessageLayout{}, false
	}
	return shape.Tail[index-start], true
}

// Records reports whether this shape is of the prefix the episode recorded.
func (shape *PolicyPrefixShape) Records(state *PolicyEpisodeState) bool {
	return shape != nil && state != nil && shape.Len == state.PrefixLen && shape.Digest == state.PrefixDigest
}

// FirstDivergence is the index of the first message where current differs
// from this shape's transcript: a changed message, or the end of the shorter
// of the two. It is shape.Len when current extends the shape. ok is false
// when the divergence lies past the digests the shape kept.
func (shape *PolicyPrefixShape) FirstDivergence(current *PolicyPrefixShape) (int, bool) {
	common := min(shape.Len, current.Len)
	for index := range common {
		if index >= len(shape.Messages) || index >= len(current.Messages) {
			return 0, false
		}
		if shape.Messages[index] != current.Messages[index] {
			return index, true
		}
	}
	return common, true
}

// policyMessageRoles and policyBlockTypes are the protocol values a
// prefix_break log reports as they are; any other value the client sent is
// reported as "other", so a log never carries a client-chosen string.
var (
	policyMessageRoles = map[string]bool{
		"user": true, "assistant": true, "system": true, "developer": true, "tool": true,
		"message": true, "function_call": true, "function_call_output": true, "reasoning": true,
		"custom_tool_call": true, "custom_tool_call_output": true, "local_shell_call": true,
		"local_shell_call_output": true, "web_search_call": true, "compaction": true,
	}
	policyBlockTypes = map[string]bool{
		"text": true, "image": true, "document": true, "tool_use": true, "tool_result": true,
		"thinking": true, "redacted_thinking": true, "server_tool_use": true, "web_search_tool_result": true,
		"search_result": true, "container_upload": true, "mcp_tool_use": true, "mcp_tool_result": true,
		"input_text": true, "input_image": true, "input_file": true, "output_text": true, "refusal": true,
	}
)

func policyKindValue(value string, known map[string]bool) string {
	if value == "" || known[value] {
		return value
	}
	return "other"
}

// PolicyMessageKind is a message's role (or a Responses item's type, when it
// has no role) and the type of its first content block ("text" for string
// content), as a prefix_break log reports them. Neither carries content: a
// value outside the protocol's is "other".
func PolicyMessageKind(message json.RawMessage) (string, string) {
	role, block := policyMessageKind(message)
	return policyKindValue(role, policyMessageRoles), policyKindValue(block, policyBlockTypes)
}

func policyMessageKind(message json.RawMessage) (string, string) {
	var value struct {
		Role    string          `json:"role"`
		Type    string          `json:"type"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(message, &value) != nil {
		return "", ""
	}
	role := value.Role
	if role == "" {
		role = value.Type
	}
	content := bytes.TrimSpace(value.Content)
	switch {
	case len(content) == 0:
		return role, ""
	case content[0] == '"':
		return role, "text"
	}
	var blocks []struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(content, &blocks) != nil || len(blocks) == 0 {
		return role, ""
	}
	return role, blocks[0].Type
}

// HasClaudeCodeBillingHeader reports whether an Anthropic system prompt
// carries Claude Code's billing header line.
func HasClaudeCodeBillingHeader(system json.RawMessage) bool {
	for _, text := range systemTexts(system) {
		if strings.HasPrefix(strings.TrimSpace(text), billingHeaderPrefix) {
			return true
		}
	}
	return false
}

// canonicalSystem is the system prompt without Claude Code's billing header
// line, whose build suffix changes between requests of one conversation (a
// block left empty without it is dropped),
// and without cache_control (the service compares it the same way).
func canonicalSystem(system json.RawMessage) []byte {
	var text string
	if json.Unmarshal(system, &text) == nil {
		stripped, _ := withoutBillingHeaderLines(text)
		raw, err := json.Marshal(stripped)
		if err != nil {
			return system
		}
		return raw
	}
	var blocks []map[string]any
	if json.Unmarshal(system, &blocks) != nil {
		return canonicalJSON(system)
	}
	kept := make([]any, 0, len(blocks))
	for _, block := range blocks {
		text, isText := block["text"].(string)
		if !isText {
			kept = append(kept, block)
			continue
		}
		// Only the header line goes; text after it in the same block stays.
		stripped, changed := withoutBillingHeaderLines(text)
		if !changed {
			kept = append(kept, block)
			continue
		}
		if strings.TrimSpace(stripped) != "" {
			copied := make(map[string]any, len(block))
			for key, value := range block {
				copied[key] = value
			}
			copied["text"] = stripped
			kept = append(kept, copied)
		}
	}
	raw, err := json.Marshal(kept)
	if err != nil {
		return system
	}
	return canonicalJSON(raw)
}

// withoutBillingHeaderLines is text without Claude Code's billing header
// lines, and whether it had any.
func withoutBillingHeaderLines(text string) (string, bool) {
	lines := strings.Split(text, "\n")
	rest := make([]string, 0, len(lines))
	for _, line := range lines {
		if !strings.HasPrefix(strings.TrimSpace(line), billingHeaderPrefix) {
			rest = append(rest, line)
		}
	}
	return strings.Join(rest, "\n"), len(rest) != len(lines)
}

// canonicalTools is the tool definitions without each tool's own
// cache_control; a schema's contents, a property named cache_control
// included, are compared as sent.
func canonicalTools(tools json.RawMessage) []byte {
	var list []map[string]json.RawMessage
	if json.Unmarshal(tools, &list) != nil {
		return canonicalJSON(tools)
	}
	for _, tool := range list {
		delete(tool, "cache_control")
	}
	raw, err := json.Marshal(list)
	if err != nil {
		return tools
	}
	return canonicalJSONKeepingCacheControl(raw)
}

// canonicalJSON is raw without cache_control, with insignificant key order
// and whitespace removed; raw that is not JSON is returned as is.
func canonicalJSON(raw json.RawMessage) []byte {
	return canonicalJSONValue(raw, true)
}

// canonicalJSONKeepingCacheControl is canonicalJSON with every member kept.
func canonicalJSONKeepingCacheControl(raw json.RawMessage) []byte {
	return canonicalJSONValue(raw, false)
}

func canonicalJSONValue(raw json.RawMessage, dropCacheControl bool) []byte {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	decoder.UseNumber()
	var value any
	if decoder.Decode(&value) != nil {
		return trimmed
	}
	if dropCacheControl {
		withoutCacheControl(value)
	}
	var canonical bytes.Buffer
	encoder := json.NewEncoder(&canonical)
	encoder.SetEscapeHTML(false)
	if encoder.Encode(value) != nil {
		return trimmed
	}
	return bytes.TrimSuffix(canonical.Bytes(), []byte("\n"))
}

func prefixShapeDigest(data []byte) uint64 {
	sum := sha256.Sum256(data)
	return binary.BigEndian.Uint64(sum[:8])
}

// PolicyMessageLayout is how one message is laid out, as a prefix_break log
// reports it: its member names, whether its content is a string or a list of
// blocks, and each block's type, member names, text length, cache_control,
// a short digest of its text and whether that text opens with a
// <system-reminder>. It carries no content: a name outside the protocol's is
// "other", and text appears only as a length and a 4-byte sha256 prefix.
type PolicyMessageLayout struct {
	Role        string
	Keys        []string
	ContentForm string
	BlockCount  int
	Blocks      []PolicyBlockLayout
}

// PolicyBlockLayout is one content block of a PolicyMessageLayout. TextLen
// is -1 for a block without a text member.
type PolicyBlockLayout struct {
	Type           string
	Keys           []string
	TextLen        int
	TextSHA256     string
	CacheControl   bool
	SystemReminder bool
}

// MaxPolicyLayoutBlocks bounds the blocks a layout describes; BlockCount is
// the message's full count.
const MaxPolicyLayoutBlocks = 16

// policyLayoutKeys are the member names a layout reports as they are.
var policyLayoutKeys = map[string]bool{
	"role": true, "content": true, "type": true, "text": true, "cache_control": true, "citations": true,
	"source": true, "id": true, "name": true, "input": true, "tool_use_id": true, "is_error": true,
	"thinking": true, "signature": true, "data": true, "title": true, "context": true, "status": true,
	"call_id": true, "arguments": true, "output": true, "summary": true, "tool_call_id": true, "tool_calls": true,
}

const systemReminderOpening = "<system-reminder>"

// NewPolicyMessageLayout describes message; one that is not a JSON object
// has form "other".
func NewPolicyMessageLayout(message json.RawMessage) PolicyMessageLayout {
	var members map[string]json.RawMessage
	if json.Unmarshal(message, &members) != nil || members == nil {
		return PolicyMessageLayout{ContentForm: "other"}
	}
	role, _ := PolicyMessageKind(message)
	layout := PolicyMessageLayout{Role: role, Keys: policyLayoutKeyNames(members)}
	content := bytes.TrimSpace(members["content"])
	var text string
	var blocks []json.RawMessage
	switch {
	case len(content) == 0:
		layout.ContentForm = "absent"
	case bytes.Equal(content, []byte("null")):
		layout.ContentForm = "null"
	case content[0] == '"' && json.Unmarshal(content, &text) == nil:
		layout.ContentForm = "string"
		layout.BlockCount = 1
		layout.Blocks = []PolicyBlockLayout{textLayout("text", nil, text, true, false)}
	case json.Unmarshal(content, &blocks) == nil:
		layout.ContentForm = "list"
		layout.BlockCount = len(blocks)
		for _, block := range blocks[:min(len(blocks), MaxPolicyLayoutBlocks)] {
			layout.Blocks = append(layout.Blocks, blockLayout(block))
		}
	default:
		layout.ContentForm = "other"
	}
	return layout
}

func blockLayout(block json.RawMessage) PolicyBlockLayout {
	var members map[string]json.RawMessage
	if json.Unmarshal(block, &members) != nil || members == nil {
		return PolicyBlockLayout{Type: "other", TextLen: -1}
	}
	var kind string
	_ = json.Unmarshal(members["type"], &kind)
	var text string
	raw := bytes.TrimSpace(members["text"])
	hasText := len(raw) > 0 && raw[0] == '"' && json.Unmarshal(raw, &text) == nil
	_, cached := members["cache_control"]
	return textLayout(policyKindValue(kind, policyBlockTypes), policyLayoutKeyNames(members), text, hasText, cached)
}

func textLayout(kind string, keys []string, text string, hasText, cached bool) PolicyBlockLayout {
	layout := PolicyBlockLayout{Type: kind, Keys: keys, TextLen: -1, CacheControl: cached}
	if hasText {
		sum := sha256.Sum256([]byte(text))
		layout.TextLen = len(text)
		layout.TextSHA256 = hex.EncodeToString(sum[:4])
		layout.SystemReminder = strings.HasPrefix(text, systemReminderOpening)
	}
	return layout
}

// policyLayoutKeyNames is members' names, each once: every name outside
// policyLayoutKeys is one "other", so the list is bounded by that set.
func policyLayoutKeyNames(members map[string]json.RawMessage) []string {
	seen := make(map[string]bool, len(policyLayoutKeys)+1)
	keys := make([]string, 0, min(len(members), len(policyLayoutKeys)+1))
	for key := range members {
		name := policyKindValue(key, policyLayoutKeys)
		if !seen[name] {
			seen[name] = true
			keys = append(keys, name)
		}
	}
	sort.Strings(keys)
	return keys
}

// Fields is the layout as log fields.
func (layout PolicyMessageLayout) Fields() map[string]interface{} {
	blocks := make([]interface{}, 0, len(layout.Blocks))
	for _, block := range layout.Blocks {
		fields := map[string]interface{}{
			"type": block.Type, "keys": block.Keys, "text_len": block.TextLen,
			"cache_control": block.CacheControl, "system_reminder": block.SystemReminder,
		}
		if block.TextSHA256 != "" {
			fields["text_sha256_8"] = block.TextSHA256
		}
		blocks = append(blocks, fields)
	}
	return map[string]interface{}{
		"role": layout.Role, "keys": layout.Keys, "content_form": layout.ContentForm,
		"block_count": layout.BlockCount, "blocks": blocks,
	}
}
