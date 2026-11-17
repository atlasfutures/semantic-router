package protocolcodec

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
)

// anthropicSchemaCombinators are the members Messages refuses at the top
// level of a tool's input_schema ("input_schema does not support oneOf,
// allOf, or anyOf at the top level"). Chat and Responses accept them, and
// client tools declare them: Codex's Computer Use js tool is a oneOf of a
// code call and a completion_summary call. A turn from those formats that is
// routed to a Claude arm would otherwise fail as a whole with a 400.
var anthropicSchemaCombinators = []string{"oneOf", "anyOf", "allOf"}

// foldableBranchMembers are what a combinator branch may say for the fold to
// keep its meaning: an object's properties, which names it requires, what it
// allows beyond them, and its annotations.
var foldableBranchMembers = []string{"type", "properties", "required", "additionalProperties", "title", "description"}

// localDefinitionMembers hold the definitions a branch's "$ref" can name.
var localDefinitionMembers = []string{"$defs", "definitions"}

// anthropicInputSchema returns schema in the form Messages accepts, and
// whether it changed it. A schema with no top-level combinator is returned as
// it came, and so is one it cannot fold without changing what it means beyond
// dropping the choice of branch: a member that is not the shape JSON Schema
// gives it, a branch that says more than foldableBranchMembers (so a
// non-object branch, a nested combinator, patternProperties, its own $id or
// $defs), a $ref branch that cannot be resolved here, a top-level
// unevaluatedProperties (which depends on the branches), or a reference
// anywhere whose target the fold could remove (hasReferenceTheFoldCouldBreak).
// Messages refuses that schema either way, and says why. A folded schema has
// no top-level combinator, so folding it again changes nothing.
//
// A property the top level declares keeps its own schema. A property only
// branches declare joins the top level with the schema each combinator's
// branches give it (combinatorPropertySchema); one that several combinators
// constrain gets all their schemas under allOf, since every top-level
// combinator applies. Every name an allOf branch requires stays required, and
// so does a name every oneOf/anyOf branch requires. What is lost is which
// oneOf/anyOf branch applies, so names only some of them require are not
// carried: requiring them all would refuse every valid call. The tool's own
// server still validates the arguments.
func anthropicInputSchema(schema json.RawMessage) (json.RawMessage, bool) {
	var top map[string]json.RawMessage
	if json.Unmarshal(schema, &top) != nil || !hasAnthropicSchemaCombinator(top) || hasReferenceTheFoldCouldBreak(schema) {
		return schema, false
	}
	if _, unevaluated := top["unevaluatedProperties"]; unevaluated {
		return schema, false
	}
	properties, required, ok := schemaObjectMembers(top)
	if !ok {
		return schema, false
	}
	isRequired := map[string]bool{}
	for _, name := range required {
		isRequired[name] = true
	}
	definitions := map[string]map[string]json.RawMessage{}
	for _, member := range localDefinitionMembers {
		var table map[string]json.RawMessage
		if !decodeSchemaMember(top, member, &table) {
			return schema, false
		}
		definitions[member] = table
	}
	constraints := map[string][]json.RawMessage{}
	// The fold's work and output are charged to a budget of four times the
	// client's schema: each branch's bytes (a $ref branch costs its
	// definition, however often it is named), each alternative a property
	// gets from a branch, and each folded property. Past it the schema is
	// sent unchanged, so a small schema cannot make the encoder do or send
	// many times more than the client did.
	budget := 4 * len(schema)
	for _, combinator := range anthropicSchemaCombinators {
		raw, present := top[combinator]
		if !present {
			continue
		}
		var members []json.RawMessage
		if json.Unmarshal(raw, &members) != nil || len(members) == 0 {
			return schema, false
		}
		branches := make([]schemaBranch, 0, len(members))
		for _, member := range members {
			branch, source, resolved := resolveSchemaBranch(definitions, member)
			if budget -= len(source); !resolved || budget < 0 || !isFoldableBranch(branch) {
				return schema, false
			}
			branchProperties, branchRequired, ok := schemaObjectMembers(branch)
			if !ok {
				return schema, false
			}
			branches = append(branches, schemaBranch{schema: branch, properties: branchProperties, required: branchRequired})
		}
		for _, name := range requiredNames(combinator, branches) {
			if !isRequired[name] {
				isRequired[name] = true
				required = append(required, name)
			}
		}
		for name := range branchPropertyNames(branches) {
			if _, declared := properties[name]; declared {
				continue
			}
			propertySchema, ok := combinatorPropertySchema(combinator, branches, name, &budget)
			if budget -= len(propertySchema); !ok || budget < 0 {
				return schema, false
			}
			constraints[name] = append(constraints[name], propertySchema)
		}
	}
	for _, combinator := range anthropicSchemaCombinators {
		delete(top, combinator)
	}
	if _, typed := top["type"]; !typed {
		top["type"] = json.RawMessage(`"object"`)
	}
	if len(constraints) > 0 {
		if properties == nil {
			properties = map[string]json.RawMessage{}
		}
		for name, schemas := range constraints {
			properties[name] = combinedSchemas("allOf", distinctSchemas(schemas))
		}
		encoded, err := json.Marshal(properties)
		if err != nil {
			return schema, false
		}
		top["properties"] = encoded
	}
	if len(required) > 0 {
		top["required"], _ = json.Marshal(required) // a []string always encodes
	}
	encoded, err := json.Marshal(top)
	if err != nil {
		return schema, false
	}
	return encoded, true
}

// isFoldableBranch reports whether branch says only foldableBranchMembers,
// with an object type if it names one.
func isFoldableBranch(branch map[string]json.RawMessage) bool {
	for member := range branch {
		if !slices.Contains(foldableBranchMembers, member) {
			return false
		}
	}
	raw, typed := branch["type"]
	var kind string
	return !typed || (json.Unmarshal(raw, &kind) == nil && kind == "object")
}

// requiredNames is what a combinator's branches make required: every name an
// allOf branch requires, and for oneOf and anyOf the names every branch
// requires, which hold whichever branch applies.
func requiredNames(combinator string, branches []schemaBranch) []string {
	var names []string
	requiredBy := map[string]int{}
	lastBranch := map[string]int{}
	for i, branch := range branches {
		for _, name := range branch.required {
			if last, counted := lastBranch[name]; counted && last == i {
				continue
			}
			lastBranch[name] = i
			if requiredBy[name] == 0 {
				names = append(names, name)
			}
			requiredBy[name]++
		}
	}
	if combinator == "allOf" {
		return names
	}
	return slices.DeleteFunc(names, func(name string) bool { return requiredBy[name] < len(branches) })
}

// schemaBranch is one resolved branch of a top-level combinator.
type schemaBranch struct {
	schema     map[string]json.RawMessage
	properties map[string]json.RawMessage
	required   []string
}

// combinatorPropertySchema is the schema one combinator's branches give the
// property name. Each branch that declares it gives its own schema; a branch
// that leaves it out gives what its additionalProperties allows. Under oneOf
// and anyOf that is any value when additionalProperties is absent or true,
// and nothing when it is false (another branch may still allow it). Under
// allOf an absent or true additionalProperties adds no constraint, and a
// false one forbids the property outright, which the fold cannot keep. The
// schemas are combined by anyOf for oneOf and anyOf (every alternative stays
// a valid value) and by allOf for allOf. Each branch, and the bytes of each
// alternative, are charged to budget. It reports false when a branch's
// additionalProperties is neither a boolean nor a schema, when an allOf
// branch forbids the property, or when the budget runs out.
func combinatorPropertySchema(combinator string, branches []schemaBranch, name string, budget *int) (json.RawMessage, bool) {
	var alternatives []json.RawMessage
	for _, branch := range branches {
		if *budget--; *budget < 0 {
			return nil, false
		}
		if property, declared := branch.properties[name]; declared {
			alternatives = append(alternatives, property)
			continue
		}
		additional, present := branch.schema["additionalProperties"]
		var additionalSchema map[string]json.RawMessage
		switch trimmed := string(bytes.TrimSpace(additional)); {
		case !present || trimmed == "true":
			if combinator != "allOf" {
				alternatives = append(alternatives, json.RawMessage(`{}`))
			}
		case trimmed == "false":
			if combinator == "allOf" {
				return nil, false
			}
		case json.Unmarshal(additional, &additionalSchema) == nil && additionalSchema != nil:
			alternatives = append(alternatives, additional)
		default:
			return nil, false
		}
	}
	for _, alternative := range alternatives {
		if *budget -= len(alternative); *budget < 0 {
			return nil, false
		}
	}
	nested := "anyOf"
	if combinator == "allOf" {
		nested = "allOf"
	}
	return combinedSchemas(nested, distinctSchemas(alternatives)), true
}

// combinedSchemas is the one schema, or the schemas under combinator.
func combinedSchemas(combinator string, schemas []json.RawMessage) json.RawMessage {
	if len(schemas) == 1 {
		return schemas[0]
	}
	combined, _ := json.Marshal(map[string][]json.RawMessage{combinator: schemas})
	return combined
}

// distinctSchemas is schemas with repeats dropped, the first of each kept.
// Schemas that differ only in member order or spacing are repeats.
func distinctSchemas(schemas []json.RawMessage) []json.RawMessage {
	distinct := make([]json.RawMessage, 0, len(schemas))
	seen := map[string]bool{}
	for _, schema := range schemas {
		key := canonicalSchemaJSON(schema)
		if seen[key] {
			continue
		}
		seen[key] = true
		distinct = append(distinct, schema)
	}
	return distinct
}

// canonicalSchemaJSON is value with its object members sorted and its
// numbers kept as written.
func canonicalSchemaJSON(value json.RawMessage) string {
	decoder := json.NewDecoder(bytes.NewReader(value))
	decoder.UseNumber()
	var decoded any
	if decoder.Decode(&decoded) != nil {
		return string(value)
	}
	encoded, err := json.Marshal(decoded)
	if err != nil {
		return string(value)
	}
	return string(encoded)
}

// branchPropertyNames is the set of property names the branches declare.
func branchPropertyNames(branches []schemaBranch) map[string]bool {
	names := map[string]bool{}
	for _, branch := range branches {
		for name := range branch.properties {
			names[name] = true
		}
	}
	return names
}

// schemaObjectMembers is a schema's properties and required names. It
// reports false when either is present and not the shape JSON Schema gives
// it.
func schemaObjectMembers(schema map[string]json.RawMessage) (map[string]json.RawMessage, []string, bool) {
	var properties map[string]json.RawMessage
	var required []string
	if !decodeSchemaMember(schema, "properties", &properties) || !decodeSchemaMember(schema, "required", &required) {
		return nil, nil, false
	}
	return properties, required, true
}

// resolveSchemaBranch decodes a combinator branch and returns it with the
// bytes it was decoded from. A branch that is a "$ref" into the schema's own
// $defs or definitions (how generated union schemas spell their branches) is
// that definition. The name is a JSON Pointer segment, so "~1" decodes to "/"
// and then "~0" to "~". It reports false for a branch that is not an object
// and for a $ref it cannot resolve: one with sibling members, one outside the
// schema or more than one segment deep, one naming no object definition, or
// one whose definition is itself a $ref.
func resolveSchemaBranch(
	definitions map[string]map[string]json.RawMessage, member json.RawMessage,
) (map[string]json.RawMessage, json.RawMessage, bool) {
	var branch map[string]json.RawMessage
	if json.Unmarshal(member, &branch) != nil || branch == nil {
		return nil, nil, false
	}
	raw, present := branch["$ref"]
	if !present {
		return branch, member, true
	}
	var ref string
	if len(branch) > 1 || json.Unmarshal(raw, &ref) != nil {
		return nil, nil, false
	}
	for _, table := range localDefinitionMembers {
		segment, local := strings.CutPrefix(ref, "#/"+table+"/")
		if !local {
			continue
		}
		if strings.Contains(segment, "/") {
			return nil, nil, false
		}
		name := strings.ReplaceAll(strings.ReplaceAll(segment, "~1", "/"), "~0", "~")
		source, defined := definitions[table][name]
		var definition map[string]json.RawMessage
		if !defined || json.Unmarshal(source, &definition) != nil || definition == nil {
			return nil, nil, false
		}
		if _, nested := definition["$ref"]; nested {
			return nil, nil, false
		}
		return definition, source, true
	}
	return nil, nil, false
}

// keptReferencePrefixes are the local JSON Pointers whose targets the fold
// keeps: the top level's own definitions and properties.
var keptReferencePrefixes = []string{"#/$defs/", "#/definitions/", "#/properties/"}

// hasReferenceTheFoldCouldBreak reports whether schema has a reference the
// fold might leave pointing at nothing: a "$ref" other than the root ("#") or
// a pointer under the top level's $defs, definitions or properties (such as
// one into a removed branch, or to an anchor), or any "$dynamicRef".
// Anything that only looks like a reference, inside an enum or a default, is
// counted too; that schema is just sent unchanged.
func hasReferenceTheFoldCouldBreak(schema json.RawMessage) bool {
	decoder := json.NewDecoder(bytes.NewReader(schema))
	decoder.UseNumber()
	var decoded any
	if decoder.Decode(&decoded) != nil {
		return true
	}
	pending := []any{decoded}
	for len(pending) > 0 {
		value := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		switch value := value.(type) {
		case map[string]any:
			if _, dynamic := value["$dynamicRef"]; dynamic {
				return true
			}
			if ref, isString := value["$ref"].(string); isString && ref != "#" &&
				!slices.ContainsFunc(keptReferencePrefixes, func(prefix string) bool { return strings.HasPrefix(ref, prefix) }) {
				return true
			}
			for _, child := range value {
				pending = append(pending, child)
			}
		case []any:
			pending = append(pending, value...)
		}
	}
	return false
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
