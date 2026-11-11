package protocolcodec

import (
	"encoding/json"
	"slices"
)

// anthropicSchemaCombinators are the members Messages refuses at the top
// level of a tool's input_schema ("input_schema does not support oneOf,
// allOf, or anyOf at the top level"). Chat and Responses accept them, and a
// Codex MCP tool declares one (Computer Use's cua_repl js tool: a oneOf of
// a code call and a completion_summary call), so a turn from those formats
// that an ARC cell gives to a Claude arm would otherwise be a 400.
var anthropicSchemaCombinators = []string{"oneOf", "anyOf", "allOf"}

// anthropicInputSchema returns schema in the form Messages accepts, and
// whether it changed it. A schema with no top-level combinator is returned as
// it came, and so is one whose members are not the shape JSON Schema gives
// them: Messages refuses that schema either way, and says why.
//
// Each combinator is removed and its object branches folded into the object
// itself: a property any branch declares joins the top level (a property the
// top level declares keeps its own schema), and every name an allOf branch
// requires stays required. What is lost is which oneOf/anyOf branch applies,
// so their required names are not carried: requiring them all would refuse
// every valid call. The tool's own server still validates the arguments.
func anthropicInputSchema(schema json.RawMessage) (json.RawMessage, bool) {
	var top map[string]json.RawMessage
	if json.Unmarshal(schema, &top) != nil || !hasAnthropicSchemaCombinator(top) {
		return schema, false
	}
	properties := map[string]json.RawMessage{}
	var required []string
	if !decodeSchemaMember(top, "properties", &properties) || !decodeSchemaMember(top, "required", &required) {
		return schema, false
	}
	if properties == nil { // "properties": null
		properties = map[string]json.RawMessage{}
	}
	for _, combinator := range anthropicSchemaCombinators {
		var branches []map[string]json.RawMessage
		if !decodeSchemaMember(top, combinator, &branches) {
			return schema, false
		}
		delete(top, combinator)
		for _, branch := range branches {
			var branchProperties map[string]json.RawMessage
			var branchRequired []string
			if !decodeSchemaMember(branch, "properties", &branchProperties) ||
				!decodeSchemaMember(branch, "required", &branchRequired) {
				return schema, false
			}
			for name, property := range branchProperties {
				if _, declared := properties[name]; !declared {
					properties[name] = property
				}
			}
			for _, name := range branchRequired {
				if combinator == "allOf" && !slices.Contains(required, name) {
					required = append(required, name)
				}
			}
		}
	}
	if _, typed := top["type"]; !typed {
		top["type"] = json.RawMessage(`"object"`)
	}
	if len(properties) > 0 {
		top["properties"], _ = json.Marshal(properties)
	}
	if len(required) > 0 {
		top["required"], _ = json.Marshal(required)
	}
	encoded, err := json.Marshal(top)
	if err != nil {
		return schema, false
	}
	return encoded, true
}

func hasAnthropicSchemaCombinator(schema map[string]json.RawMessage) bool {
	for _, combinator := range anthropicSchemaCombinators {
		if _, present := schema[combinator]; present {
			return true
		}
	}
	return false
}

// decodeSchemaMember decodes member into target when it is present, and
// reports false only when it is present and not that shape.
func decodeSchemaMember(schema map[string]json.RawMessage, member string, target any) bool {
	raw, present := schema[member]
	return !present || json.Unmarshal(raw, target) == nil
}
