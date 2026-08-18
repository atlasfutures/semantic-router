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
)

// anthropicDroppedBlockTypes lists the blocks that contribute no turn text.
// Server-executed tool calls and their results are dropped as a pair: the call
// never records a tool name, so its result never has to resolve one. Rich and
// binary payloads are dropped because Rayline drops them. fallback drops text
// it does carry, because it is documented as never rendered into the prompt.
//
// mid_conv_system is listed here but is not unconditional. It is the system
// prompt relocated into the message list, so TurnOptions.IncludeSystemText
// decides its fate: off, it drops with the rest of the system text; on, it
// renders in place, which is where it already sits.
//
// This table plus the rendered cases must cover the whole Anthropic request
// block union. A type in neither fails the episode closed, which is
// deliberate: the selector is a trained artifact, so showing it a silently
// truncated conversation is worse than refusing to route.
//
// Verified complete against anthropic-sdk-python 0.109.1: the GA and beta
// request unions, plus the response-side unions those accept back for
// round-tripping. Widen it from the union, not from a failure report.
var anthropicDroppedBlockTypes = map[string]bool{
	"advisor_tool_result":                    true,
	"bash_code_execution_tool_result":        true,
	"code_execution_tool_result":             true,
	"container_upload":                       true,
	"document":                               true,
	"fallback":                               true,
	"image":                                  true,
	"mcp_tool_result":                        true,
	"mcp_tool_use":                           true,
	"mid_conv_system":                        true,
	"redacted_thinking":                      true,
	"search_result":                          true,
	"server_tool_use":                        true,
	"text_editor_code_execution_tool_result": true,
	"thinking":                               true,
	"tool_search_tool_result":                true,
	"web_fetch_tool_result":                  true,
	"web_search_tool_result":                 true,
}

func normalizeAnthropicTurns(
	request map[string]any,
	options TurnOptions,
) ([]Turn, error) {
	messages, err := requiredArray(request, "messages", "request")
	if err != nil {
		return nil, err
	}
	// Anthropic has no system role for input messages, so the standing system
	// prompt arrives only in this top-level field. Nothing read it before, so
	// it was not merely dropped at the parser: it was never looked at.
	systemText := newSystemTextBuffer(options)
	if options.IncludeSystemText {
		text, systemErr := anthropicSystemText(
			request["system"],
			"request.system",
		)
		if systemErr != nil {
			return nil, systemErr
		}
		systemText.add(text)
	}
	turns := make([]Turn, 0, len(messages))
	toolNames := make(map[string]string)
	for index, value := range messages {
		path := fmt.Sprintf("messages[%d]", index)
		message, objectErr := requiredObject(value, path)
		if objectErr != nil {
			return nil, objectErr
		}
		role, _ := message["role"].(string)
		if role != "assistant" && role != "user" {
			continue
		}
		blocks, blockErr := anthropicContentBlocks(message["content"], path+".content")
		if blockErr != nil {
			return nil, blockErr
		}
		if role == "assistant" {
			text, renderErr := renderAnthropicAssistant(
				blocks,
				toolNames,
				path+".content",
				options.IncludeSystemText,
			)
			if renderErr != nil {
				return nil, renderErr
			}
			turns = append(turns, Turn{Role: role, Text: text})
			continue
		}
		text, renderErr := renderAnthropicUser(
			blocks,
			toolNames,
			path+".content",
			options.IncludeSystemText,
		)
		if renderErr != nil {
			return nil, renderErr
		}
		// take() empties the buffer, so the standing system prompt lands on
		// the first user turn only. That turn is what the serializer renders
		// as the episode's [Task] block.
		turns = append(
			turns,
			Turn{Role: role, Text: joinSystemText(systemText.take(), text)},
		)
	}
	return systemText.flushTrailing(turns), nil
}

// anthropicSystemText renders the top-level system field, which the API
// accepts either as a bare string or as an array of text blocks.
func anthropicSystemText(value any, path string) (string, error) {
	if value == nil {
		return "", nil
	}
	if text, ok := value.(string); ok {
		return text, nil
	}
	blocks, ok := value.([]any)
	if !ok {
		return "", turnError(
			"invalid_field",
			path,
			"system must be a string or an array of text blocks",
		)
	}
	return anthropicTextBlocksText(blocks, path)
}

// anthropicTextBlocksText joins an array that the API declares as text blocks
// and nothing else. Both the top-level system field and a mid_conv_system
// block carry exactly this shape.
func anthropicTextBlocksText(blocks []any, path string) (string, error) {
	texts := make([]string, 0, len(blocks))
	for index, value := range blocks {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		block, blockType, err := anthropicBlockType(value, blockPath)
		if err != nil {
			return "", err
		}
		if blockType != "text" {
			return "", turnErrorWithDetail(
				"unknown_item",
				blockPath+".type",
				blockType,
				"system content accepts only text blocks, got %q",
				blockType,
			)
		}
		text, textErr := requiredString(block, "text", blockPath)
		if textErr != nil {
			return "", textErr
		}
		if text != "" {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n"), nil
}

// midConvSystemText renders a system instruction relocated into the message
// list. The API declares content as an array of text blocks. A bare string is
// accepted too, because callers write that shape and it costs nothing to read.
func midConvSystemText(block map[string]any, path string) (string, error) {
	if text, ok := block["text"].(string); ok {
		return text, nil
	}
	value, ok := block["content"]
	if !ok || value == nil {
		return "", nil
	}
	if text, ok := value.(string); ok {
		return text, nil
	}
	blocks, ok := value.([]any)
	if !ok {
		return "", turnError(
			"invalid_field",
			path+".content",
			"mid_conv_system content must be a string or an array",
		)
	}
	return anthropicTextBlocksText(blocks, path+".content")
}

func anthropicContentBlocks(value any, path string) ([]any, error) {
	if value == nil {
		return nil, nil
	}
	if text, ok := value.(string); ok {
		return []any{map[string]any{"type": "text", "text": text}}, nil
	}
	blocks, ok := value.([]any)
	if !ok {
		return nil, turnError(
			"invalid_field",
			path,
			"content must be a string or array",
		)
	}
	return blocks, nil
}

// anthropicBlockType reads the discriminator every content block must carry.
func anthropicBlockType(value any, path string) (map[string]any, string, error) {
	block, err := requiredObject(value, path)
	if err != nil {
		return nil, "", err
	}
	blockType, err := requiredString(block, "type", path)
	if err != nil {
		return nil, "", err
	}
	return block, blockType, nil
}

// anthropicSharedBlockText renders the blocks that mean the same thing in
// either role, and decides the fate of every block neither renderer claims.
// Blocks with a Rayline-defined drop behavior contribute no text; anything
// else fails the episode closed. Keeping that decision in one place is what
// makes the set of accepted block types auditable.
func anthropicSharedBlockText(
	block map[string]any,
	blockType string,
	blockPath string,
	includeSystem bool,
) (string, error) {
	if blockType == "text" {
		return requiredString(block, "text", blockPath)
	}
	if blockType == "compaction" {
		return compactionText(block, blockPath)
	}
	// Rendered in place rather than buffered: the block already sits at the
	// position it governs, inside a turn that will carry it.
	if blockType == "mid_conv_system" && includeSystem {
		return midConvSystemText(block, blockPath)
	}
	if anthropicDroppedBlockTypes[blockType] {
		return "", nil
	}
	return "", turnErrorWithDetail(
		"unknown_item",
		blockPath+".type",
		blockType,
		"unsupported Anthropic block type %q",
		blockType,
	)
}

// compactionText reads the summary a compaction block carries in place of the
// turns it replaced. The summary renders as ordinary text because it is the
// only surviving record of that stretch of conversation: dropping it would
// show the selector a long session as a short one, and route it accordingly.
// The API sets content to null when compaction failed, which contributes no
// text rather than failing the episode.
func compactionText(block map[string]any, path string) (string, error) {
	value, ok := block["content"]
	if !ok || value == nil {
		return "", nil
	}
	text, ok := value.(string)
	if !ok {
		return "", turnError(
			"invalid_field",
			path+".content",
			"compaction content must be a string",
		)
	}
	return text, nil
}

func renderAnthropicAssistant(
	blocks []any,
	toolNames map[string]string,
	path string,
	includeSystem bool,
) (string, error) {
	parts := make([]string, 0, len(blocks))
	for index, value := range blocks {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		block, blockType, err := anthropicBlockType(value, blockPath)
		if err != nil {
			return "", err
		}
		if blockType == "tool_use" {
			rendered, renderErr := renderAnthropicToolUse(
				block,
				toolNames,
				blockPath,
			)
			if renderErr != nil {
				return "", renderErr
			}
			parts = append(parts, rendered)
			continue
		}
		text, textErr := anthropicSharedBlockText(
			block,
			blockType,
			blockPath,
			includeSystem,
		)
		if textErr != nil {
			return "", textErr
		}
		if text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, "\n"), nil
}

func renderAnthropicToolUse(
	block map[string]any,
	toolNames map[string]string,
	path string,
) (string, error) {
	callID, err := requiredString(block, "id", path)
	if err != nil {
		return "", err
	}
	name, err := requiredString(block, "name", path)
	if err != nil {
		return "", err
	}
	if err := recordToolName(toolNames, callID, name, path); err != nil {
		return "", err
	}
	arguments := block["input"]
	if _, object := arguments.(map[string]any); object {
		return renderToolCall(name, arguments), nil
	}
	return fmt.Sprintf("[tool_call %s] %s", name, stringCoerce(arguments)), nil
}

func renderAnthropicUser(
	blocks []any,
	toolNames map[string]string,
	path string,
	includeSystem bool,
) (string, error) {
	texts := make([]string, 0, len(blocks))
	results := make([]string, 0, len(blocks))
	for index, value := range blocks {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		block, blockType, err := anthropicBlockType(value, blockPath)
		if err != nil {
			return "", err
		}
		if blockType == "tool_result" {
			result, resultErr := renderAnthropicToolResult(
				block,
				toolNames,
				blockPath,
			)
			if resultErr != nil {
				return "", resultErr
			}
			results = append(results, result)
			continue
		}
		text, textErr := anthropicSharedBlockText(
			block,
			blockType,
			blockPath,
			includeSystem,
		)
		if textErr != nil {
			return "", textErr
		}
		if text != "" {
			texts = append(texts, text)
		}
	}
	return joinUserTextAndResults(texts, results), nil
}

func renderAnthropicToolResult(
	block map[string]any,
	toolNames map[string]string,
	path string,
) (string, error) {
	callID, err := requiredString(block, "tool_use_id", path)
	if err != nil {
		return "", err
	}
	name, err := resolveToolName(toolNames, callID, path+".tool_use_id")
	if err != nil {
		return "", err
	}
	text, err := toolResultText(block["content"], path+".content")
	if err != nil {
		return "", err
	}
	isError, err := optionalBool(block, "is_error", path)
	if err != nil {
		return "", err
	}
	return renderToolResult(name, text, isError), nil
}

func toolResultText(value any, path string) (string, error) {
	if value == nil {
		return "", nil
	}
	if text, ok := value.(string); ok {
		return text, nil
	}
	blocks, ok := value.([]any)
	if !ok {
		return "", turnError(
			"invalid_field",
			path,
			"tool result must be a string or content array",
		)
	}
	texts := make([]string, 0, len(blocks))
	for index, value := range blocks {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		text, include, err := toolResultBlockText(value, blockPath)
		if err != nil {
			return "", err
		}
		if include {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n"), nil
}

func toolResultBlockText(value any, path string) (string, bool, error) {
	block, blockType, err := anthropicBlockType(value, path)
	if err != nil {
		return "", false, err
	}
	if blockType == "text" || blockType == "input_text" ||
		blockType == "output_text" {
		text, textErr := requiredString(block, "text", path)
		return text, true, textErr
	}
	// tool_reference names a tool instead of carrying text. It is legal only
	// inside tool_result.content, so it is dropped here rather than in the
	// top-level table, which stays strict about what may appear as a block.
	if blockType == "tool_reference" {
		return "", false, nil
	}
	if anthropicDroppedBlockTypes[blockType] ||
		openAIDroppedContentTypes[blockType] {
		return "", false, nil
	}
	return "", false, turnErrorWithDetail(
		"unknown_item",
		path+".type",
		blockType,
		"unsupported result content type %q",
		blockType,
	)
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
