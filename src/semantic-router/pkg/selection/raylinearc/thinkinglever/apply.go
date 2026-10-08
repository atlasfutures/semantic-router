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

package thinkinglever

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Messages projects the client transcript for the planner.
func Messages(messages []llmprotocol.Message) []Message {
	projected := make([]Message, len(messages))
	for index, message := range messages {
		projected[index] = Message{
			Role:   string(message.Role),
			Digest: messageDigest(message),
		}
	}
	return projected
}

// ImageToolTail reports whether a turn's tool run returned an image on a
// worker wire that ADR 0129 governs, as pathfinder's placer reads the
// encoded body. An image in the trailing user messages counts as well as
// one in a tool output: Chat's images travel in user messages after the run,
// and a Responses codec may hoist one into a user item there. Messages keeps
// an image inside its tool_result and is not governed. A message the target
// encoder omits whole, of any role, is read past and counts for nothing.
func ImageToolTail(request llmprotocol.Request, wire llmprotocol.WireFormat) bool {
	if wire != llmprotocol.OpenAIChatV1 && wire != llmprotocol.OpenAIResponsesV1 {
		return false
	}
	messages := request.Messages
	position := len(messages) - 1
	images := false
	for ; position >= 0; position-- {
		message := messages[position]
		if protocolcodec.MessageEncodesToNothing(request, message, wire) || unspoken(message) {
			continue
		}
		if message.Role != llmprotocol.RoleUser {
			break
		}
		images = images || carriesImage(message.Content)
	}
	run := false
	for ; position >= 0; position-- {
		message := messages[position]
		if protocolcodec.MessageEncodesToNothing(request, message, wire) || unspoken(message) {
			continue
		}
		if message.Role != llmprotocol.RoleTool {
			break
		}
		run = true
		for _, content := range message.Content {
			if content.ToolResult != nil {
				images = images || carriesImage(content.ToolResult.Content)
			}
		}
	}
	return images && run
}

// unspoken is an assistant message holding reasoning and nothing else,
// which the provider adapter may drop on the way to the worker. The detector
// reads past it, so the turn is judged as the worker may see it; where the
// adapter keeps it, this refuses more, never less. Anything the worker reads
// as the assistant's answer (text, a refusal, a tool call) ends the run.
func unspoken(message llmprotocol.Message) bool {
	if message.Role != llmprotocol.RoleAssistant || len(message.Content) == 0 {
		return false
	}
	for _, content := range message.Content {
		if content.Kind != llmprotocol.ContentReasoning && !opaqueReasoning(content) {
			return false
		}
	}
	return true
}

// opaqueReasoning is an Anthropic thinking block the contract carries whole.
func opaqueReasoning(content llmprotocol.Content) bool {
	block := content.Unmodeled
	return content.Kind == llmprotocol.ContentUnmodeled && block != nil &&
		block.Format == llmprotocol.AnthropicMessagesV1 &&
		(block.Type == "redacted_thinking" || block.Type == "thinking")
}

func carriesImage(content []llmprotocol.Content) bool {
	for _, part := range content {
		if part.Kind == llmprotocol.ContentImage {
			return true
		}
	}
	return false
}

// messageDigest identifies a client message across turns. It ignores
// what agent clients rewrite in history without changing what the model was
// asked: moving cache breakpoints, re-sent <system-reminder> blocks, caller
// annotations and members no contract names. Anything else that changes is a
// rewrite, and the ledger starts a new epoch.
func messageDigest(message llmprotocol.Message) string {
	projection := struct {
		Role    llmprotocol.Role
		Content []llmprotocol.Content
	}{Role: message.Role, Content: stableContent(message.Content)}
	payload, _ := json.Marshal(projection)
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:DigestBytes])
}

func stableContent(blocks []llmprotocol.Content) []llmprotocol.Content {
	stable := make([]llmprotocol.Content, 0, len(blocks))
	for _, block := range blocks {
		if block.Kind == llmprotocol.ContentText &&
			strings.HasPrefix(strings.TrimSpace(block.Text), systemReminderPrefix) {
			continue
		}
		block.Cache = nil
		block.Extensions = nil
		if block.ToolCall != nil {
			call := *block.ToolCall
			call.Caller = nil
			block.ToolCall = &call
		}
		if block.ToolResult != nil {
			result := *block.ToolResult
			result.Content = stableContent(result.Content)
			block.ToolResult = &result
		}
		stable = append(stable, block)
	}
	return stable
}

// ApplyLedger returns the provider-bound transcript for a worker whose
// binding uses lever: the client's messages with every item of that lever
// put back where it was first written. Items of the other lever are left
// out, so a worker never receives an item its binding was not admitted for
// and each worker's transcript stays what it saw before. The input slice and
// its messages are not modified. Items are applied from the highest anchor
// down so an insertion never shifts an anchor still to come.
func ApplyLedger(
	messages []llmprotocol.Message,
	lever Lever,
	ledger Ledger,
) ([]llmprotocol.Message, error) {
	result := append([]llmprotocol.Message(nil), messages...)
	for position := len(ledger.Entries) - 1; position >= 0; position-- {
		entry := ledger.Entries[position]
		index := int(entry.Index)
		if index < 0 || index >= len(result) {
			return nil, fmt.Errorf("thinking ledger anchor %d is outside the transcript", index)
		}
		if entry.Payload < 0 || entry.Payload >= len(ledger.Payloads) {
			return nil, fmt.Errorf("thinking ledger payload %d is unknown", entry.Payload)
		}
		payload := ledger.Payloads[entry.Payload]
		if payload.Lever != lever {
			continue
		}
		var err error
		result, err = applyEntry(result, index, entry.Placement, payload)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

func applyEntry(
	messages []llmprotocol.Message,
	index int,
	placement Placement,
	payload Payload,
) ([]llmprotocol.Message, error) {
	if !placementFitsLever(payload.Lever, placement) {
		return nil, fmt.Errorf("placement %q does not fit lever %q", placement, payload.Lever)
	}
	switch placement {
	case PlaceAppendTailUserText:
		anchored := messages[index]
		anchored.Content = append(
			append([]llmprotocol.Content(nil), anchored.Content...),
			llmprotocol.Content{Kind: llmprotocol.ContentText, Text: payload.Suffix},
		)
		messages[index] = anchored
		return messages, nil
	case PlaceUserAfterToolRun:
		// The steer joins the wire group of the tool results it follows, so
		// on Messages it travels as a trailing text block of the user message
		// that carries them rather than as a message of its own.
		// On Chat, where the tool results' images move into user messages
		// after the run, it joins the last of those (ADR 0129).
		return insertMessage(messages, index+1, llmprotocol.Message{
			Role:           llmprotocol.RoleUser,
			Content:        []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: payload.Suffix}},
			WireGroup:      messages[index].WireGroup,
			JoinsToolMedia: true,
		}), nil
	case PlaceSystemBeforeTurn:
		return insertMessage(messages, index, effortMessage(payload)), nil
	case PlaceSystemAfterToolRun:
		return insertMessage(messages, index+1, effortMessage(payload)), nil
	default:
		return nil, fmt.Errorf("unknown thinking placement %q", placement)
	}
}

func effortMessage(payload Payload) llmprotocol.Message {
	return llmprotocol.Message{
		Role:          llmprotocol.RoleSystem,
		Configuration: &llmprotocol.ConfigurationUpdate{ReasoningEffort: payload.Effort},
	}
}

func insertMessage(
	messages []llmprotocol.Message,
	at int,
	message llmprotocol.Message,
) []llmprotocol.Message {
	result := make([]llmprotocol.Message, 0, len(messages)+1)
	result = append(result, messages[:at]...)
	result = append(result, message)
	return append(result, messages[at:]...)
}
