package llmprotocol

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Tool declarations are the part of a request that grew fastest: an ordinary
// Claude Code turn declares thirty-two of them, and a beta adds a new member
// or a new type every few weeks. This file holds what a declaration must
// state and which tool a refusal came from, apart from the rest of the
// request contract.

func validateRequestTools(tools []Tool, limits Limits) (map[string]struct{}, int, error) {
	namedTools := make(map[string]struct{}, len(tools))
	schemaBytes := 0
	for index, tool := range tools {
		location := toolLocation(index, tool)
		if err := validateRequestTool(tool, limits, location); err != nil {
			return nil, 0, err
		}
		if _, duplicate := namedTools[tool.Identity()]; duplicate {
			return nil, 0, NewFieldError(ErrorInvalidRequest, "duplicate_tool",
				"tool names must be unique", location, "tools.name")
		}
		schemaBytes += len(tool.InputSchema)
		if limits.SchemaBytes > 0 && schemaBytes > limits.SchemaBytes {
			return nil, 0, NewFieldError(ErrorInvalidRequest, "schema_limit",
				"total schema limit exceeded", location, "tools.input_schema")
		}
		namedTools[tool.Identity()] = struct{}{}
	}
	return namedTools, schemaBytes, nil
}

// toolLocation names which declared tool a refusal came from. A refused body
// is never stored, and a request declares thirty-two tools on an ordinary
// Claude Code turn, so "some tool was too large" is not something an operator
// can act on. On 2026-09-03 locating one such refusal took a log join.
func toolLocation(index int, tool Tool) string {
	if identity := tool.Identity(); identity != "" {
		return fmt.Sprintf("tool %d named %q", index, identity)
	}
	return fmt.Sprintf("tool %d", index)
}

// validateRequestTool checks what a declared tool must state. A callable tool
// has to name the function the model calls and describe its arguments. A
// server tool states neither: the source API runs it, and its type is the
// whole declaration -- {"type":"web_search_20250305"} is the shape Claude Code
// sends. Refusing that shape refused the turn around it.
func validateRequestTool(tool Tool, limits Limits, location string) error {
	if err := validateToolDeclaration(tool, location); err != nil {
		return err
	}
	if err := validateToolTextLimits(tool, limits, location); err != nil {
		return err
	}
	if err := validateCacheDirective(tool.Cache, location, "tools.cache_control"); err != nil {
		return err
	}
	if len(tool.InputSchema) == 0 {
		return nil
	}
	return validateSchemaObject(tool.InputSchema, "tool schema", location, "tools.input_schema", limits)
}

// validateToolDeclaration checks what a tool must state to be usable at all.
func validateToolDeclaration(tool Tool, location string) error {
	if len(tool.InputSchema) > 0 && !json.Valid(tool.InputSchema) {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool",
			"tool JSON Schema is not valid JSON", location, "tools.input_schema")
	}
	if tool.ServerTool() {
		return nil
	}
	if strings.TrimSpace(tool.Name) == "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool",
			"a callable tool must name the function the model calls", location, "tools.name")
	}
	if len(tool.InputSchema) == 0 {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool",
			"a callable tool must describe its arguments", location, "tools.input_schema")
	}
	return nil
}

func validateToolTextLimits(tool Tool, limits Limits, location string) error {
	if exceeds(tool.Name, limits.ToolNameBytes) {
		return toolTextLimit(location, "tools.name", len(tool.Name), limits.ToolNameBytes)
	}
	if exceeds(tool.Description, limits.ToolDescriptionBytes) {
		return toolTextLimit(location, "tools.description", len(tool.Description), limits.ToolDescriptionBytes)
	}
	if limits.SchemaBytes > 0 && len(tool.InputSchema) > limits.SchemaBytes {
		return NewFieldError(ErrorInvalidRequest, "schema_limit",
			"tool schema limit exceeded", location, "tools.input_schema")
	}
	return nil
}

// toolTextLimit names the tool, the member and the two byte counts, and
// nothing the client wrote. The counts are what tell an operator whether the
// limit is wrong or the request is.
func toolTextLimit(location, field string, observed, limit int) error {
	return NewFieldError(
		ErrorInvalidRequest, "tool_text_limit",
		"tool name or description exceeds the configured limit", location, field,
	).WithCount("bytes", observed, limit)
}

func validateToolChoice(choice ToolChoice, namedTools map[string]struct{}, toolCount int, hasImageGeneration bool) error {
	if !validToolChoiceMode(choice.Mode) {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool_choice",
			"tool choice is invalid", "", "tool_choice")
	}
	if choice.Mode == ToolChoiceNamed {
		return validateNamedToolChoice(choice.Name, namedTools)
	}
	if choice.Name != "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool_choice",
			"only named tool choice may contain a name", "", "tool_choice.name")
	}
	if choice.Mode == ToolChoiceImageGeneration && !hasImageGeneration {
		return NewFieldError(ErrorInvalidRequest, "image_generation_tool_required",
			"image-generation tool choice requires a declared image-generation tool", "", "tool_choice")
	}
	if choice.Mode == ToolChoiceRequired && toolCount == 0 && !hasImageGeneration {
		return NewFieldError(ErrorInvalidRequest, "tools_required",
			"tool choice requires at least one declared tool", "", "tool_choice")
	}
	return nil
}

func validToolChoiceMode(mode ToolChoiceMode) bool {
	switch mode {
	case "", ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired, ToolChoiceNamed, ToolChoiceImageGeneration:
		return true
	default:
		return false
	}
}

func validateNamedToolChoice(name string, namedTools map[string]struct{}) error {
	if strings.TrimSpace(name) == "" {
		return NewFieldError(ErrorInvalidRequest, "tool_choice_name_required",
			"named tool choice requires a name", "", "tool_choice.name")
	}
	if _, found := namedTools[name]; !found {
		return NewFieldError(ErrorInvalidRequest, "unknown_tool_choice",
			"named tool choice does not reference a declared tool", "", "tool_choice.name")
	}
	return nil
}
