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

var anthropicDroppedBlockTypes = map[string]bool{
	"bash_code_execution_tool_result":        true,
	"code_execution_tool_result":             true,
	"container_upload":                       true,
	"document":                               true,
	"image":                                  true,
	"redacted_thinking":                      true,
	"search_result":                          true,
	"server_tool_use":                        true,
	"text_editor_code_execution_tool_result": true,
	"thinking":                               true,
	"web_fetch_tool_result":                  true,
	"web_search_tool_result":                 true,
}

func normalizeAnthropicTurns(request map[string]any) ([]Turn, error) {
	messages, err := requiredArray(request, "messages", "request")
	if err != nil {
		return nil, err
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
		)
		if renderErr != nil {
			return nil, renderErr
		}
		turns = append(turns, Turn{Role: role, Text: text})
	}
	return turns, nil
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

func renderAnthropicAssistant(
	blocks []any,
	toolNames map[string]string,
	path string,
) (string, error) {
	parts := make([]string, 0, len(blocks))
	for index, value := range blocks {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		block, err := requiredObject(value, blockPath)
		if err != nil {
			return "", err
		}
		blockType, err := requiredString(block, "type", blockPath)
		if err != nil {
			return "", err
		}
		switch blockType {
		case "text":
			text, textErr := requiredString(block, "text", blockPath)
			if textErr != nil {
				return "", textErr
			}
			if text != "" {
				parts = append(parts, text)
			}
		case "tool_use":
			rendered, renderErr := renderAnthropicToolUse(
				block,
				toolNames,
				blockPath,
			)
			if renderErr != nil {
				return "", renderErr
			}
			parts = append(parts, rendered)
		default:
			if !anthropicDroppedBlockTypes[blockType] {
				return "", turnErrorWithDetail(
					"unknown_item",
					blockPath+".type",
					blockType,
					"unsupported Anthropic block type %q",
					blockType,
				)
			}
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
) (string, error) {
	texts := make([]string, 0, len(blocks))
	results := make([]string, 0, len(blocks))
	for index, value := range blocks {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		block, err := requiredObject(value, blockPath)
		if err != nil {
			return "", err
		}
		blockType, err := requiredString(block, "type", blockPath)
		if err != nil {
			return "", err
		}
		switch blockType {
		case "text":
			text, textErr := requiredString(block, "text", blockPath)
			if textErr != nil {
				return "", textErr
			}
			if text != "" {
				texts = append(texts, text)
			}
		case "tool_result":
			result, resultErr := renderAnthropicToolResult(
				block,
				toolNames,
				blockPath,
			)
			if resultErr != nil {
				return "", resultErr
			}
			results = append(results, result)
		default:
			if !anthropicDroppedBlockTypes[blockType] {
				return "", turnErrorWithDetail(
					"unknown_item",
					blockPath+".type",
					blockType,
					"unsupported Anthropic block type %q",
					blockType,
				)
			}
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
	block, err := requiredObject(value, path)
	if err != nil {
		return "", false, err
	}
	blockType, err := requiredString(block, "type", path)
	if err != nil {
		return "", false, err
	}
	if blockType == "text" || blockType == "input_text" ||
		blockType == "output_text" {
		text, textErr := requiredString(block, "text", path)
		return text, true, textErr
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
