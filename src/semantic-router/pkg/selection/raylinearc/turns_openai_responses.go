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
)

// droppedResponsesItemTypes lists the input items that contribute no turn text.
// Only plain messages and function calls reach the selector: every other tool
// family is a built-in whose call and output are dropped as a pair, so no
// output ever has to resolve a tool name its call never recorded.
//
// This table plus the rendered cases must cover the whole Responses input item
// union. An item type in neither fails the episode closed, on the same
// reasoning as the Anthropic table: a silently shortened conversation routes
// worse than a refused one.
//
// Verified complete against openai-python 2.24.0.
var droppedResponsesItemTypes = map[string]bool{
	"apply_patch_call":        true,
	"apply_patch_call_output": true,
	"code_interpreter_call":   true,
	"compaction":              true,
	"computer_call":           true,
	"computer_call_output":    true,
	"custom_tool_call":        true,
	"custom_tool_call_output": true,
	"file_search_call":        true,
	"image_generation_call":   true,
	"item_reference":          true,
	"local_shell_call":        true,
	"local_shell_call_output": true,
	"mcp_approval_request":    true,
	"mcp_approval_response":   true,
	"mcp_call":                true,
	"mcp_list_tools":          true,
	"reasoning":               true,
	"shell_call":              true,
	"shell_call_output":       true,
	"web_search_call":         true,
}

type responseTurnPart struct {
	role   string
	text   string
	result bool
}

func normalizeOpenAIResponsesTurns(
	request map[string]any,
	options TurnOptions,
) ([]Turn, error) {
	input, ok := request["input"]
	if !ok {
		return nil, turnError(
			"missing_field",
			"request.input",
			"field is required",
		)
	}
	// The Responses API carries its standing system prompt in this top-level
	// field. Nothing read it before, so like the Anthropic system field it was
	// never dropped at the parser -- it was never looked at.
	systemText := newSystemTextBuffer(options)
	if options.IncludeSystemText {
		instructions, instructionsErr := responsesInstructionsText(
			request["instructions"],
			"request.instructions",
		)
		if instructionsErr != nil {
			return nil, instructionsErr
		}
		systemText.add(instructions)
	}
	if text, isText := input.(string); isText {
		return []Turn{{
			Role: "user",
			Text: joinSystemText(systemText.take(), text),
		}}, nil
	}
	items, ok := input.([]any)
	if !ok {
		return nil, turnError(
			"invalid_field",
			"request.input",
			"input must be a string or array",
		)
	}
	parts := make([]responseTurnPart, 0, len(items))
	toolNames := make(map[string]string)
	for index, value := range items {
		path := fmt.Sprintf("input[%d]", index)
		item, err := requiredObject(value, path)
		if err != nil {
			return nil, err
		}
		itemParts, err := normalizeResponseItem(
			item,
			toolNames,
			path,
			&systemText,
		)
		if err != nil {
			return nil, err
		}
		parts = append(parts, itemParts...)
	}
	turns := coalesceResponseTurnParts(parts)
	return systemText.flushTrailing(turns), nil
}

// responsesInstructionsText reads the top-level instructions field, which the
// API declares as a plain string.
func responsesInstructionsText(value any, path string) (string, error) {
	if value == nil {
		return "", nil
	}
	text, ok := value.(string)
	if !ok {
		return "", turnError(
			"invalid_field",
			path,
			"instructions must be a string",
		)
	}
	return text, nil
}

func normalizeResponseItem(
	item map[string]any,
	toolNames map[string]string,
	path string,
	systemText *systemTextBuffer,
) ([]responseTurnPart, error) {
	itemType, err := requiredString(item, "type", path)
	if err != nil {
		return nil, err
	}
	switch itemType {
	case "message":
		return normalizeResponseMessage(item, path, systemText)
	case "function_call":
		part, callErr := normalizeResponseFunctionCall(item, toolNames, path)
		return singletonResponsePart(part, callErr)
	case "function_call_output":
		part, outputErr := normalizeResponseFunctionOutput(
			item,
			toolNames,
			path,
		)
		return singletonResponsePart(part, outputErr)
	default:
		if droppedResponsesItemTypes[itemType] {
			return nil, nil
		}
		return nil, turnErrorWithDetail(
			"unknown_item",
			path+".type",
			itemType,
			"unsupported Responses item type %q",
			itemType,
		)
	}
}

func singletonResponsePart(
	part responseTurnPart,
	err error,
) ([]responseTurnPart, error) {
	if err != nil {
		return nil, err
	}
	return []responseTurnPart{part}, nil
}

func normalizeResponseMessage(
	item map[string]any,
	path string,
	systemText *systemTextBuffer,
) ([]responseTurnPart, error) {
	role, err := requiredString(item, "role", path)
	if err != nil {
		return nil, err
	}
	if role == "system" || role == "developer" {
		// Read the content only when it is wanted, so that a malformed system
		// message cannot start failing requests that succeed today.
		if !systemText.enabled {
			return nil, nil
		}
		text, systemErr := openAIContentText(
			item["content"],
			path+".content",
		)
		if systemErr != nil {
			return nil, systemErr
		}
		systemText.add(text)
		return nil, nil
	}
	if role != "user" && role != "assistant" {
		return nil, turnErrorWithDetail(
			"unknown_item",
			path+".role",
			role,
			"unsupported Responses message role %q",
			role,
		)
	}
	text, err := openAIContentText(item["content"], path+".content")
	if err != nil {
		return nil, err
	}
	if role == "user" {
		// take() empties the buffer, so each stretch of system text lands on
		// the first user turn that follows it.
		text = joinSystemText(systemText.take(), text)
	}
	return []responseTurnPart{{role: role, text: text}}, nil
}

func normalizeResponseFunctionCall(
	item map[string]any,
	toolNames map[string]string,
	path string,
) (responseTurnPart, error) {
	callID, err := requiredString(item, "call_id", path)
	if err != nil {
		return responseTurnPart{}, err
	}
	name, err := requiredString(item, "name", path)
	if err != nil {
		return responseTurnPart{}, err
	}
	arguments, err := requiredString(item, "arguments", path)
	if err != nil {
		return responseTurnPart{}, err
	}
	if recordErr := recordToolName(
		toolNames,
		callID,
		name,
		path+".call_id",
	); recordErr != nil {
		return responseTurnPart{}, recordErr
	}
	parsed, err := decodeArgumentJSON(arguments, path+".arguments")
	if err != nil {
		return responseTurnPart{}, err
	}
	return responseTurnPart{
		role: "assistant",
		text: renderToolCall(name, parsed),
	}, nil
}

func normalizeResponseFunctionOutput(
	item map[string]any,
	toolNames map[string]string,
	path string,
) (responseTurnPart, error) {
	callID, err := requiredString(item, "call_id", path)
	if err != nil {
		return responseTurnPart{}, err
	}
	name, err := resolveToolName(toolNames, callID, path+".call_id")
	if err != nil {
		return responseTurnPart{}, err
	}
	text, err := toolResultText(item["output"], path+".output")
	if err != nil {
		return responseTurnPart{}, err
	}
	isError, err := optionalBool(item, "is_error", path)
	if err != nil {
		return responseTurnPart{}, err
	}
	return responseTurnPart{
		role:   "user",
		text:   renderToolResult(name, text, isError),
		result: true,
	}, nil
}

func coalesceResponseTurnParts(parts []responseTurnPart) []Turn {
	turns := make([]Turn, 0, len(parts))
	previousWasResult := false
	for _, part := range parts {
		if len(turns) == 0 || turns[len(turns)-1].Role != part.role {
			turns = append(turns, Turn{Role: part.role, Text: part.text})
			previousWasResult = part.result
			continue
		}
		separator := "\n"
		if part.result || previousWasResult {
			separator = "\n\n"
		}
		if turns[len(turns)-1].Text == "" {
			separator = ""
		}
		turns[len(turns)-1].Text += separator + part.text
		previousWasResult = part.result
	}
	return turns
}
