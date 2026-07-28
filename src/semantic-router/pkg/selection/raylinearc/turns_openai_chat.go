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

var openAIDroppedContentTypes = map[string]bool{
	"file":         true,
	"image_url":    true,
	"input_audio":  true,
	"input_file":   true,
	"input_image":  true,
	"output_image": true,
	"refusal":      true,
}

func normalizeOpenAIChatTurns(request map[string]any) ([]Turn, error) {
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
		role, roleErr := requiredString(message, "role", path)
		if roleErr != nil {
			return nil, roleErr
		}
		turns, err = appendOpenAIChatMessage(
			turns,
			message,
			role,
			toolNames,
			path,
		)
		if err != nil {
			return nil, err
		}
	}
	return turns, nil
}

func appendOpenAIChatMessage(
	turns []Turn,
	message map[string]any,
	role string,
	toolNames map[string]string,
	path string,
) ([]Turn, error) {
	switch role {
	case "system", "developer":
		return turns, nil
	case "user":
		text, err := openAIContentText(message["content"], path+".content")
		if err != nil {
			return nil, err
		}
		return append(turns, Turn{Role: role, Text: text}), nil
	case "assistant":
		text, err := renderOpenAIChatAssistant(message, toolNames, path)
		if err != nil {
			return nil, err
		}
		return append(turns, Turn{Role: role, Text: text}), nil
	case "tool":
		text, err := renderOpenAIChatToolResult(message, toolNames, path)
		if err != nil {
			return nil, err
		}
		return appendOpenAIChatToolResult(turns, text), nil
	default:
		return nil, turnError(
			"unknown_item",
			path+".role",
			"unsupported Chat Completions role %q",
			role,
		)
	}
}

func appendOpenAIChatToolResult(turns []Turn, text string) []Turn {
	if len(turns) == 0 || turns[len(turns)-1].Role != "user" {
		return append(turns, Turn{Role: "user", Text: text})
	}
	last := &turns[len(turns)-1]
	if last.Text == "" {
		last.Text = text
	} else {
		last.Text += "\n\n" + text
	}
	return turns
}

func renderOpenAIChatAssistant(
	message map[string]any,
	toolNames map[string]string,
	path string,
) (string, error) {
	text, err := openAIContentText(message["content"], path+".content")
	if err != nil {
		return "", err
	}
	parts := make([]string, 0)
	if text != "" {
		parts = append(parts, text)
	}
	value, exists := message["tool_calls"]
	if !exists || value == nil {
		return strings.Join(parts, "\n"), nil
	}
	calls, ok := value.([]any)
	if !ok {
		return "", turnError(
			"invalid_field",
			path+".tool_calls",
			"tool_calls must be an array",
		)
	}
	for index, value := range calls {
		callPath := fmt.Sprintf("%s.tool_calls[%d]", path, index)
		call, objectErr := requiredObject(value, callPath)
		if objectErr != nil {
			return "", objectErr
		}
		rendered, renderErr := renderOpenAIChatToolCall(
			call,
			toolNames,
			callPath,
		)
		if renderErr != nil {
			return "", renderErr
		}
		parts = append(parts, rendered)
	}
	return strings.Join(parts, "\n"), nil
}

func renderOpenAIChatToolCall(
	call map[string]any,
	toolNames map[string]string,
	path string,
) (string, error) {
	callType, err := requiredString(call, "type", path)
	if err != nil {
		return "", err
	}
	if callType != "function" {
		return "", turnError(
			"unknown_item",
			path+".type",
			"unsupported Chat Completions tool type %q",
			callType,
		)
	}
	callID, err := requiredString(call, "id", path)
	if err != nil {
		return "", err
	}
	function, err := requiredObject(call["function"], path+".function")
	if err != nil {
		return "", err
	}
	name, err := requiredString(function, "name", path+".function")
	if err != nil {
		return "", err
	}
	arguments, err := requiredString(function, "arguments", path+".function")
	if err != nil {
		return "", err
	}
	if recordErr := recordToolName(
		toolNames,
		callID,
		name,
		path+".id",
	); recordErr != nil {
		return "", recordErr
	}
	parsed, err := decodeArgumentJSON(arguments, path+".function.arguments")
	if err != nil {
		return "", err
	}
	return renderToolCall(name, parsed), nil
}

func renderOpenAIChatToolResult(
	message map[string]any,
	toolNames map[string]string,
	path string,
) (string, error) {
	callID, err := requiredString(message, "tool_call_id", path)
	if err != nil {
		return "", err
	}
	name, err := resolveToolName(toolNames, callID, path+".tool_call_id")
	if err != nil {
		return "", err
	}
	text, err := toolResultText(message["content"], path+".content")
	if err != nil {
		return "", err
	}
	return renderToolResult(name, text, false), nil
}

func openAIContentText(value any, path string) (string, error) {
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
			"content must be a string, array, or null",
		)
	}
	texts := make([]string, 0, len(blocks))
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
		case "text", "input_text", "output_text":
			text, textErr := requiredString(block, "text", blockPath)
			if textErr != nil {
				return "", textErr
			}
			if text != "" {
				texts = append(texts, text)
			}
		default:
			if !openAIDroppedContentTypes[blockType] {
				return "", turnError(
					"unknown_item",
					blockPath+".type",
					"unsupported OpenAI content type %q",
					blockType,
				)
			}
		}
	}
	return strings.Join(texts, "\n"), nil
}
