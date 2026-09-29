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
		return insertMessage(messages, index+1, llmprotocol.Message{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: payload.Suffix}},
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
