package protocolcodec

import (
	"bytes"
	"encoding/json"
	"net/url"
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
// keep its meaning: an object's properties, which names it requires, whether
// it allows others, and its annotations.
var foldableBranchMembers = []string{"type", "properties", "required", "additionalProperties", "title", "description"}

// localDefinitionMembers hold the definitions a branch's "$ref" can name.
var localDefinitionMembers = []string{"$defs", "definitions"}

// keptReferencePrefixes are the local JSON Pointers whose targets the fold
// keeps: the top level's own definitions and properties.
var keptReferencePrefixes = []string{"#/$defs/", "#/definitions/", "#/properties/"}

// maxFoldPairs bounds the branch-property pairs a fold weighs, so a schema
// with very many branches and properties is sent unchanged rather than
// folded at a cost far beyond its size.
const maxFoldPairs = 1 << 16

// anthropicInputSchema returns schema in the form Messages accepts, and
// whether it changed it. Only a schema with exactly one top-level combinator
// is folded. Its branches are folded into the object:
//
//   - A property the top level declares keeps its own schema.
//   - A property only branches declare gets the schemas the branches give
//     it: under oneOf and anyOf, each declaring branch's schema, and any value
//     from a branch that leaves it out but allows other properties; under
//     allOf, each declaring branch's schema. Several schemas are combined by
//     anyOf for oneOf and anyOf, and by allOf for allOf.
//   - Every name an allOf branch requires stays required, and so does a name
//     every oneOf/anyOf branch requires. What is lost is which oneOf/anyOf
//     branch applies; the tool's own server still validates the arguments.
//
// A branch that is a "$ref" into the schema's own $defs or definitions is
// that definition. Any other schema is returned as it came, since folding it
// would change more than the choice of branch: more than one top-level
// combinator, a top-level unevaluatedProperties, a branch that says more than
// foldableBranchMembers or allows other properties through a schema, a $ref
// branch that cannot be resolved, an allOf branch that forbids a property
// another branch declares, a reference the fold could leave pointing at
// nothing, or a fold bigger than twice the schema. Messages refuses that
// schema either way, and says why. A folded schema has no top-level
// combinator, so folding it again changes nothing.
func anthropicInputSchema(schema json.RawMessage) (json.RawMessage, bool) {
	var top map[string]json.RawMessage
	if json.Unmarshal(schema, &top) != nil {
		return schema, false
	}
	var combinator string
	for _, name := range anthropicSchemaCombinators {
		if _, present := top[name]; present {
			if combinator != "" {
				return schema, false
			}
			combinator = name
		}
	}
	if combinator == "" || hasReferenceTheFoldCouldBreak(schema) {
		return schema, false
	}
	if _, unevaluated := top["unevaluatedProperties"]; unevaluated {
		return schema, false
	}
	properties, required, ok := schemaObjectMembers(top)
	if !ok {
		return schema, false
	}
	definitions := map[string]map[string]json.RawMessage{}
	for _, member := range localDefinitionMembers {
		var table map[string]json.RawMessage
		if !decodeSchemaMember(top, member, &table) {
			return schema, false
		}
		definitions[member] = table
	}
	branches, ok := foldableBranches(top[combinator], definitions)
	if !ok {
		return schema, false
	}
	names := branchPropertyNames(branches)
	if len(names)*len(branches) > maxFoldPairs {
		return schema, false
	}
	added := false
	for _, name := range names {
		if _, declared := properties[name]; declared {
			continue
		}
		propertySchema, ok := combinatorPropertySchema(combinator, branches, name)
		if !ok {
			return schema, false
		}
		if properties == nil {
			properties = map[string]json.RawMessage{}
		}
		properties[name] = propertySchema
		added = true
	}
	isRequired := map[string]bool{}
	for _, name := range required {
		isRequired[name] = true
	}
	for _, name := range requiredNames(combinator, branches) {
		if !isRequired[name] {
			isRequired[name] = true
			required = append(required, name)
		}
	}
	delete(top, combinator)
	if _, typed := top["type"]; !typed {
		top["type"] = json.RawMessage(`"object"`)
	}
	if added {
		top["properties"] = encodeSchemaJSON(properties)
	}
	if len(required) > 0 {
		top["required"] = encodeSchemaJSON(required)
	}
	encoded := encodeSchemaJSON(top)
	if encoded == nil || len(encoded) > 2*len(schema) {
		return schema, false
	}
	return encoded, true
}

// schemaBranch is one resolved branch of the top-level combinator.
type schemaBranch struct {
	properties map[string]json.RawMessage
	required   []string
	closed     bool // additionalProperties is false
}

// foldableBranches decodes the combinator's branches, resolving $ref
// branches and dropping repeats. It reports false when the combinator is not
// a non-empty array or a branch cannot be folded.
func foldableBranches(raw json.RawMessage, definitions map[string]map[string]json.RawMessage) ([]schemaBranch, bool) {
	var members []json.RawMessage
	if json.Unmarshal(raw, &members) != nil || len(members) == 0 {
		return nil, false
	}
	seen := map[string]bool{}
	branches := make([]schemaBranch, 0, len(members))
	for _, member := range members {
		var branch map[string]json.RawMessage
		if json.Unmarshal(member, &branch) != nil || branch == nil {
			return nil, false
		}
		key := string(compactJSON(member))
		if _, isReference := branch["$ref"]; isReference {
			table, name, ok := localDefinitionReference(branch)
			if !ok {
				return nil, false
			}
			// Keyed by the definition, so each is decoded once however
			// the references to it are spelled.
			key = "$ref\x00" + table + "\x00" + name
			if !seen[key] {
				if branch, ok = localDefinition(definitions[table][name]); !ok {
					return nil, false
				}
			}
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		if !isFoldableBranch(branch) {
			return nil, false
		}
		properties, required, ok := schemaObjectMembers(branch)
		if !ok {
			return nil, false
		}
		additional := string(bytes.TrimSpace(branch["additionalProperties"]))
		if additional != "" && additional != "true" && additional != "false" {
			return nil, false
		}
		branches = append(branches, schemaBranch{properties: properties, required: required, closed: additional == "false"})
	}
	return branches, true
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

// combinatorPropertySchema is the schema the branches give the property name
// (see anthropicInputSchema). It reports false when an allOf branch forbids
// the property.
func combinatorPropertySchema(combinator string, branches []schemaBranch, name string) (json.RawMessage, bool) {
	var alternatives []json.RawMessage
	for _, branch := range branches {
		if property, declared := branch.properties[name]; declared {
			alternatives = append(alternatives, property)
			continue
		}
		switch {
		case combinator == "allOf" && branch.closed:
			return nil, false
		case combinator != "allOf" && !branch.closed:
			alternatives = append(alternatives, json.RawMessage(`{}`))
		}
	}
	alternatives = distinctSchemas(alternatives)
	if len(alternatives) == 1 {
		return alternatives[0], true
	}
	nested := "anyOf"
	if combinator == "allOf" {
		nested = "allOf"
	}
	return encodeSchemaJSON(map[string][]json.RawMessage{nested: alternatives}), true
}

// requiredNames is what the branches make required: every name an allOf
// branch requires, and for oneOf and anyOf the names every branch requires,
// which hold whichever branch applies.
func requiredNames(combinator string, branches []schemaBranch) []string {
	var names []string
	requiredBy := map[string]int{}
	for _, branch := range branches {
		for _, name := range slices.Compact(slices.Sorted(slices.Values(branch.required))) {
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

// branchPropertyNames is every property name the branches declare, sorted.
func branchPropertyNames(branches []schemaBranch) []string {
	var names []string
	for _, branch := range branches {
		for name := range branch.properties {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	return slices.Compact(names)
}

// distinctSchemas is schemas with repeats dropped, the first of each kept.
func distinctSchemas(schemas []json.RawMessage) []json.RawMessage {
	distinct := make([]json.RawMessage, 0, len(schemas))
	seen := map[string]bool{}
	for _, schema := range schemas {
		if key := string(compactJSON(schema)); !seen[key] {
			seen[key] = true
			distinct = append(distinct, schema)
		}
	}
	return distinct
}

func compactJSON(value json.RawMessage) []byte {
	var compacted bytes.Buffer
	if json.Compact(&compacted, value) != nil {
		return value
	}
	return compacted.Bytes()
}

// encodeSchemaJSON encodes value without escaping <, > and &, so the
// client's strings keep their bytes. It is nil only for a value JSON cannot
// encode.
func encodeSchemaJSON(value any) json.RawMessage {
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	if encoder.Encode(value) != nil {
		return nil
	}
	return bytes.TrimSuffix(encoded.Bytes(), []byte("\n"))
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

// localDefinitionReference is the definition a branch names with a "$ref"
// into the schema's own $defs or definitions (how generated union schemas
// spell their branches): the table and the definition's name. The reference
// is a URI fragment holding a JSON Pointer, so it is percent-decoded and then
// "~1" becomes "/" and "~0" becomes "~". It reports false for a $ref with
// sibling members, and for one outside the schema or more than one segment
// deep.
func localDefinitionReference(branch map[string]json.RawMessage) (string, string, bool) {
	var ref string
	if len(branch) > 1 || json.Unmarshal(branch["$ref"], &ref) != nil {
		return "", "", false
	}
	pointer, err := url.PathUnescape(ref)
	if err != nil {
		return "", "", false
	}
	for _, table := range localDefinitionMembers {
		segment, local := strings.CutPrefix(pointer, "#/"+table+"/")
		if local && !strings.Contains(segment, "/") {
			return table, strings.ReplaceAll(strings.ReplaceAll(segment, "~1", "/"), "~0", "~"), true
		}
	}
	return "", "", false
}

// localDefinition decodes a definition a branch names. It reports false when
// there is none, it is not an object, or it is itself a $ref.
func localDefinition(raw json.RawMessage) (map[string]json.RawMessage, bool) {
	var definition map[string]json.RawMessage
	if json.Unmarshal(raw, &definition) != nil || definition == nil {
		return nil, false
	}
	_, nested := definition["$ref"]
	return definition, !nested
}

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

// decodeSchemaMember decodes member into target when it is present, and
// reports false only when it is present and not that shape.
func decodeSchemaMember(schema map[string]json.RawMessage, member string, target any) bool {
	raw, present := schema[member]
	return !present || json.Unmarshal(raw, target) == nil
}
