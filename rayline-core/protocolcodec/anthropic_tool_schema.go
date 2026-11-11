package protocolcodec

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
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
// it came, and so is one it cannot fold without losing a branch's properties
// (members not the shape JSON Schema gives them, a branch that is itself a
// union or intersection, or a $ref outside the schema): Messages refuses
// that schema either way, and says why.
//
// Each combinator is removed and its object branches folded into the object
// itself. A property the top level declares keeps its own schema; one only
// branches declare joins the top level, as the branches' schemas combined by
// the same combinator nested on the property (anyOf for oneOf and anyOf, so
// every alternative stays a valid value). Every name an allOf branch requires
// stays required. What is lost is which oneOf/anyOf branch applies, so their
// required names are not carried: requiring them all would refuse every valid
// call. The tool's own server still validates the arguments.
func anthropicInputSchema(schema json.RawMessage) (json.RawMessage, bool) {
	var top map[string]json.RawMessage
	if json.Unmarshal(schema, &top) != nil || !hasAnthropicSchemaCombinator(top) {
		return schema, false
	}
	var properties map[string]json.RawMessage
	var required []string
	if !decodeSchemaMember(top, "properties", &properties) || !decodeSchemaMember(top, "required", &required) {
		return schema, false
	}
	if properties == nil {
		properties = map[string]json.RawMessage{}
	}
	for _, combinator := range anthropicSchemaCombinators {
		var branches []map[string]json.RawMessage
		if !decodeSchemaMember(top, combinator, &branches) {
			return schema, false
		}
		delete(top, combinator)
		alternatives := map[string][]json.RawMessage{}
		seen := map[string]bool{}
		for _, branch := range branches {
			branch, resolved := resolveLocalSchemaRef(top, branch)
			if !resolved || hasAnthropicSchemaCombinator(branch) {
				return schema, false
			}
			var branchProperties map[string]json.RawMessage
			var branchRequired []string
			if !decodeSchemaMember(branch, "properties", &branchProperties) ||
				!decodeSchemaMember(branch, "required", &branchRequired) {
				return schema, false
			}
			for name, property := range branchProperties {
				if _, declared := properties[name]; declared {
					continue
				}
				if key := name + "\x00" + string(compactJSON(property)); !seen[key] {
					seen[key] = true
					alternatives[name] = append(alternatives[name], property)
				}
			}
			for _, name := range branchRequired {
				if combinator == "allOf" && !slices.Contains(required, name) {
					required = append(required, name)
				}
			}
		}
		for name, schemas := range alternatives {
			properties[name] = combinedPropertySchema(combinator, schemas)
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

// combinedPropertySchema is one property's schema from the distinct schemas
// the branches of combinator gave it.
func combinedPropertySchema(combinator string, schemas []json.RawMessage) json.RawMessage {
	if len(schemas) == 1 {
		return schemas[0]
	}
	if combinator != "allOf" {
		combinator = "anyOf"
	}
	combined, _ := json.Marshal(map[string][]json.RawMessage{combinator: schemas})
	return combined
}

func compactJSON(value json.RawMessage) []byte {
	var compacted bytes.Buffer
	if json.Compact(&compacted, value) != nil {
		return value
	}
	return compacted.Bytes()
}

// resolveLocalSchemaRef returns the definition a branch names with a "$ref"
// to the schema's own $defs or definitions (how generated union schemas spell
// their branches), or the branch itself when it has no $ref. It reports false
// for a $ref it cannot resolve, or one with sibling members, so the schema is
// sent unchanged rather than folded into an object that lost the branch's
// properties.
func resolveLocalSchemaRef(top, branch map[string]json.RawMessage) (map[string]json.RawMessage, bool) {
	raw, present := branch["$ref"]
	if !present {
		return branch, true
	}
	var ref string
	if len(branch) > 1 || json.Unmarshal(raw, &ref) != nil {
		return nil, false
	}
	for _, member := range []string{"$defs", "definitions"} {
		name, local := strings.CutPrefix(ref, "#/"+member+"/")
		if !local {
			continue
		}
		var definitions map[string]map[string]json.RawMessage
		if !decodeSchemaMember(top, member, &definitions) {
			return nil, false
		}
		definition, defined := definitions[name]
		if _, nested := definition["$ref"]; !defined || nested {
			return nil, false
		}
		return definition, true
	}
	return nil, false
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
