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
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The encoder wire schema pins a turn role to user or assistant, and the
// serializer writes the role string into the token stream. A third role would
// emit tokens the trained checkpoint has never been shown.
const (
	turnRoleUser      = "user"
	turnRoleAssistant = "assistant"
)

// droppedContentKinds lists the neutral content kinds that contribute no turn
// text. Binary and rich payloads are dropped because Rayline drops them.
// Reasoning and refusals are dropped because the encoder was never shown them.
//
// This table plus the rendered kinds must cover the whole llmprotocol content
// union. A kind in neither fails the episode closed, which is deliberate: the
// selector is a trained artifact, so showing it a silently truncated
// conversation is worse than refusing to route. Widen this table from the
// union in pkg/llmprotocol, never from a failure report.
var droppedContentKinds = map[llmprotocol.ContentKind]bool{
	llmprotocol.ContentAudio:          true,
	llmprotocol.ContentFile:           true,
	llmprotocol.ContentGeneratedImage: true,
	llmprotocol.ContentImage:          true,
	llmprotocol.ContentReasoning:      true,
	llmprotocol.ContentRefusal:        true,
	llmprotocol.ContentVideo:          true,
}

// ProjectTurns renders the neutral request into the turn list the encoder was
// trained on.
//
// The router decodes every public wire format exactly once, so this is a
// projection and not a parser: the wire body is already gone by the time the
// selector runs. The projection is deliberately lossy. It keeps role-tagged
// text, tool calls and tool results, and drops everything else.
func ProjectTurns(
	request *llmprotocol.Request,
	options TurnOptions,
) ([]Turn, error) {
	if request == nil {
		return nil, turnError(
			"missing_request",
			"",
			"neutral request is unavailable",
		)
	}
	systemText := newSystemTextBuffer(options)
	if err := collectInstructionText(
		&systemText,
		request.Instructions,
	); err != nil {
		return nil, err
	}
	turns := make([]Turn, 0, len(request.Messages))
	toolNames := make(map[string]string)
	for index, message := range request.Messages {
		path := fmt.Sprintf("messages[%d]", index)
		projected, err := appendProjectedMessage(
			turns,
			message,
			path,
			toolNames,
			&systemText,
		)
		if err != nil {
			return nil, err
		}
		turns = projected
	}
	return systemText.flushTrailing(turns), nil
}

// collectInstructionText reads the conversation-opening prompt.
//
// The codec hoists system text out of the message sequence and into
// Instructions: the Anthropic top-level system field, the Responses
// instructions field, and every system or developer message in a Chat
// Completions array all land here, with no record of where they sat. So the
// neutral request cannot distinguish an opening brief from a correction
// inserted mid-conversation, and everything in Instructions is treated as the
// opening scope.
func collectInstructionText(
	systemText *systemTextBuffer,
	instructions []llmprotocol.InstructionBlock,
) error {
	for index, block := range instructions {
		path := fmt.Sprintf("instructions[%d].content", index)
		if err := systemText.collect(
			systemTextOriginal,
			func() (string, error) {
				return contentText(block.Content, path)
			},
		); err != nil {
			return err
		}
	}
	return nil
}

func appendProjectedMessage(
	turns []Turn,
	message llmprotocol.Message,
	path string,
	toolNames map[string]string,
	systemText *systemTextBuffer,
) ([]Turn, error) {
	switch message.Role {
	case llmprotocol.RoleSystem, llmprotocol.RoleDeveloper:
		// The codec hoists system text into Instructions, so this arm is
		// unreachable from a decoded request today. It stays because the
		// position is known when it does run, which is the only place the
		// mid-conversation scope can still be honoured.
		scope := systemTextScopeAt(len(turns) > 0)
		err := systemText.collect(scope, func() (string, error) {
			return contentText(message.Content, path+".content")
		})
		if err != nil {
			return nil, err
		}
		return turns, nil
	case llmprotocol.RoleUser:
		text, err := projectConversationContent(
			message.Content,
			toolNames,
			path+".content",
		)
		if err != nil {
			return nil, err
		}
		// take() empties the buffer, so each stretch of system text lands on
		// the first user turn that follows it -- the turn it governs. That
		// turn is what the serializer renders as the episode's [Task] block.
		return append(turns, Turn{
			Role: turnRoleUser,
			Text: joinSystemText(systemText.take(), text),
		}), nil
	case llmprotocol.RoleAssistant:
		text, err := projectAssistantContent(
			message.Content,
			toolNames,
			path+".content",
		)
		if err != nil {
			return nil, err
		}
		return append(turns, Turn{Role: turnRoleAssistant, Text: text}), nil
	case llmprotocol.RoleTool:
		// A tool result is not a turn of its own. It answers the user turn it
		// follows, so it is folded into that turn rather than given a role the
		// encoder has never been shown.
		text, err := projectConversationContent(
			message.Content,
			toolNames,
			path+".content",
		)
		if err != nil {
			return nil, err
		}
		return appendToolResultText(turns, text), nil
	default:
		return nil, turnErrorWithDetail(
			"unknown_item",
			path+".role",
			string(message.Role),
			"unsupported message role %q",
			message.Role,
		)
	}
}

func appendToolResultText(turns []Turn, text string) []Turn {
	if len(turns) == 0 || turns[len(turns)-1].Role != turnRoleUser {
		return append(turns, Turn{Role: turnRoleUser, Text: text})
	}
	last := &turns[len(turns)-1]
	if last.Text == "" {
		last.Text = text
	} else {
		last.Text += "\n\n" + text
	}
	return turns
}

// projectConversationContent renders the blocks a user or tool message may
// carry. Text and tool results are kept apart because they join differently:
// text runs together, while each result is its own block.
func projectConversationContent(
	blocks []llmprotocol.Content,
	toolNames map[string]string,
	path string,
) (string, error) {
	texts := make([]string, 0, len(blocks))
	results := make([]string, 0, len(blocks))
	for index, block := range blocks {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		if block.Kind == llmprotocol.ContentToolResult {
			result, err := renderToolResultBlock(block, toolNames, blockPath)
			if err != nil {
				return "", err
			}
			results = append(results, result)
			continue
		}
		text, err := plainContentText(block, blockPath)
		if err != nil {
			return "", err
		}
		if text != "" {
			texts = append(texts, text)
		}
	}
	return joinUserTextAndResults(texts, results), nil
}

func projectAssistantContent(
	blocks []llmprotocol.Content,
	toolNames map[string]string,
	path string,
) (string, error) {
	parts := make([]string, 0, len(blocks))
	for index, block := range blocks {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		if block.Kind == llmprotocol.ContentToolCall {
			rendered, err := renderToolCallBlock(block, toolNames, blockPath)
			if err != nil {
				return "", err
			}
			parts = append(parts, rendered)
			continue
		}
		text, err := plainContentText(block, blockPath)
		if err != nil {
			return "", err
		}
		if text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, "\n"), nil
}

// contentText renders a block run that may only carry text, which is what a
// system prompt and a tool result body are.
func contentText(blocks []llmprotocol.Content, path string) (string, error) {
	texts := make([]string, 0, len(blocks))
	for index, block := range blocks {
		text, err := plainContentText(
			block,
			fmt.Sprintf("%s[%d]", path, index),
		)
		if err != nil {
			return "", err
		}
		if text != "" {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n"), nil
}

// plainContentText decides the fate of every block the callers do not claim.
// Keeping that decision in one place is what makes the set of accepted content
// kinds auditable.
func plainContentText(
	block llmprotocol.Content,
	path string,
) (string, error) {
	if block.Kind == llmprotocol.ContentText {
		return block.Text, nil
	}
	if droppedContentKinds[block.Kind] {
		return "", nil
	}
	return "", turnErrorWithDetail(
		"unknown_item",
		path+".kind",
		string(block.Kind),
		"unsupported content kind %q",
		block.Kind,
	)
}

func renderToolCallBlock(
	block llmprotocol.Content,
	toolNames map[string]string,
	path string,
) (string, error) {
	call := block.ToolCall
	if call == nil {
		return "", turnError(
			"invalid_field",
			path,
			"tool call block carries no call",
		)
	}
	if err := recordToolName(
		toolNames,
		call.ID,
		call.Name,
		path,
	); err != nil {
		return "", err
	}
	arguments, err := decodeArgumentJSON(call.Arguments, path+".arguments")
	if err != nil {
		return "", err
	}
	return renderToolCall(call.Name, arguments), nil
}

func renderToolResultBlock(
	block llmprotocol.Content,
	toolNames map[string]string,
	path string,
) (string, error) {
	result := block.ToolResult
	if result == nil {
		return "", turnError(
			"invalid_field",
			path,
			"tool result block carries no result",
		)
	}
	name, err := resolveToolName(toolNames, result.CallID, path+".call_id")
	if err != nil {
		return "", err
	}
	text, err := contentText(result.Content, path+".content")
	if err != nil {
		return "", err
	}
	isError := result.IsError != nil && *result.IsError
	return renderToolResult(name, text, isError), nil
}

func joinUserTextAndResults(texts []string, results []string) string {
	switch {
	case len(results) == 0:
		return strings.Join(texts, "\n")
	case len(texts) == 0:
		return strings.Join(results, "\n\n")
	default:
		return strings.Join(texts, "\n") + "\n\n" +
			strings.Join(results, "\n\n")
	}
}
