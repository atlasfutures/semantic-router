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

// PolicyMessageKind is a message's role (or a Responses item's type, when it
// has no role) and the type of its first content block ("text" for string
// content), as a prefix_break log reports them. Neither carries content.
func PolicyMessageKind(message json.RawMessage) (string, string) {
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
// block, whose build suffix changes between requests of one conversation,
// and without cache_control (the service compares it the same way).
func canonicalSystem(system json.RawMessage) []byte {
	var blocks []map[string]any
	if json.Unmarshal(system, &blocks) != nil {
		return canonicalJSON(system)
	}
	kept := make([]any, 0, len(blocks))
	for _, block := range blocks {
		if text, _ := block["text"].(string); strings.HasPrefix(strings.TrimSpace(text), billingHeaderPrefix) {
			continue
		}
		kept = append(kept, block)
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
