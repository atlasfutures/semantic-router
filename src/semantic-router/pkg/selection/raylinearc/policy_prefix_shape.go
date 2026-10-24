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
	"encoding/json"
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
}

// MaxPolicyPrefixShapeMessages bounds the per-message digests a shape keeps;
// a longer transcript keeps its first ones, and a divergence past them is
// reported unknown.
const MaxPolicyPrefixShapeMessages = 1024

// NewPolicyPrefixShape summarizes messages (each in the canonical form
// PolicyTurn compares) with the request's system prompt and tools.
func NewPolicyPrefixShape(messages []json.RawMessage, system, tools json.RawMessage) *PolicyPrefixShape {
	kept := min(len(messages), MaxPolicyPrefixShapeMessages)
	shape := &PolicyPrefixShape{
		Len:      len(messages),
		Digest:   MessagesDigest(messages, len(messages)),
		Messages: make([]uint64, kept),
		System:   shortDigest(canonicalSystem(system)),
		Tools:    shortDigest(canonicalJSON(tools)),
	}
	for index := range kept {
		shape.Messages[index] = shortDigest(canonicalPrefixMessage(messages[index]))
	}
	return shape
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
		lines := strings.Split(text, "\n")
		rest := lines[:0]
		for _, line := range lines {
			if !strings.HasPrefix(strings.TrimSpace(line), billingHeaderPrefix) {
				rest = append(rest, line)
			}
		}
		if len(rest) == len(lines) {
			kept = append(kept, block)
			continue
		}
		if stripped := strings.Join(rest, "\n"); strings.TrimSpace(stripped) != "" {
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

// canonicalJSON is raw without cache_control, with insignificant key order
// and whitespace removed; raw that is not JSON is returned as is.
func canonicalJSON(raw json.RawMessage) []byte {
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
	withoutCacheControl(value)
	var canonical bytes.Buffer
	encoder := json.NewEncoder(&canonical)
	encoder.SetEscapeHTML(false)
	if encoder.Encode(value) != nil {
		return trimmed
	}
	return bytes.TrimSuffix(canonical.Bytes(), []byte("\n"))
}

func shortDigest(data []byte) uint64 {
	sum := sha256.Sum256(data)
	return binary.BigEndian.Uint64(sum[:8])
}
