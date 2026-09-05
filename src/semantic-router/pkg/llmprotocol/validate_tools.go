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
	if len(tool.InputSchema) > 0 && !json.Valid(tool.InputSchema) {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool",
			"tool JSON Schema is not valid JSON", location, "tools.input_schema")
	}
	if !tool.ServerTool() && strings.TrimSpace(tool.Name) == "" {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool",
			"a callable tool must name the function the model calls", location, "tools.name")
	}
	if !tool.ServerTool() && len(tool.InputSchema) == 0 {
		return NewFieldError(ErrorInvalidRequest, "invalid_tool",
			"a callable tool must describe its arguments", location, "tools.input_schema")
	}
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
	if err := validateCacheDirective(tool.Cache); err != nil {
		return err
	}
	if len(tool.InputSchema) == 0 {
		return nil
	}
	return validateSchemaObject(tool.InputSchema, "tool schema", limits)
}

// toolTextOverflow names the field and the two byte counts, and nothing the
// client wrote. Without it the refusal log records only that some tool was too
// long, which is what made the 2026-09-04 dev-cell refusals take a code read
// and a client capture to explain.
// toolTextLimit names the tool, the member and the two byte counts, and
// nothing the client wrote. The counts are what tell an operator whether the
// limit is wrong or the request is.
func toolTextLimit(location, field string, observed, limit int) error {
	refusal := NewFieldError(
		ErrorInvalidRequest, "tool_text_limit",
		"tool name or description exceeds the configured limit", location, field,
	)
	overflow := fmt.Sprintf("%d bytes, limit %d", observed, limit)
	refusal.Message += " (" + overflow + ")"
	refusal.Cause = fmt.Errorf("%w: %s", refusal.Cause, overflow)
	return refusal
}

func validateToolChoice(choice ToolChoice, namedTools map[string]struct{}, toolCount int, hasImageGeneration bool) error {
	if !validToolChoiceMode(choice.Mode) {
		return NewError(ErrorInvalidRequest, "invalid_tool_choice", "tool choice is invalid", nil)
	}
	if choice.Mode == ToolChoiceNamed {
		return validateNamedToolChoice(choice.Name, namedTools)
	}
	if choice.Name != "" {
		return NewError(ErrorInvalidRequest, "invalid_tool_choice", "only named tool choice may contain a name", nil)
	}
	if choice.Mode == ToolChoiceImageGeneration && !hasImageGeneration {
		return NewError(ErrorInvalidRequest, "image_generation_tool_required", "image-generation tool choice requires a declared image-generation tool", nil)
	}
	if choice.Mode == ToolChoiceRequired && toolCount == 0 && !hasImageGeneration {
		return NewError(ErrorInvalidRequest, "tools_required", "tool choice requires at least one declared tool", nil)
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
		return NewError(ErrorInvalidRequest, "tool_choice_name_required", "named tool choice requires a name", nil)
	}
	if _, found := namedTools[name]; !found {
		return NewError(ErrorInvalidRequest, "unknown_tool_choice", "named tool choice does not reference a declared tool", nil)
	}
	return nil
}
