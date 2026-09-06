package protocolcodec

import (
	"encoding/json"
	"reflect"
	"sort"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// This file holds the depth half of accept-by-default. The request carrier in
// unmodeled.go widens the top level of a request object; this widens every
// modelled struct beneath it, with one walk shared by the request envelope,
// each content block and each tool.
//
// The walk descends into a member the struct does name when that member is
// itself a modelled object, and into each element of a member that is an array
// of modelled objects. It stops at json.RawMessage and at an array of anything
// else, because both are already carried verbatim by the member holding them.

// captureUnnamedMembers returns the members of one source object that the wire
// struct does not name, plus the same for each named member that is itself a
// modelled object. It answers nil when the object holds nothing unnamed, so a
// body with no extension decodes exactly as it did before.
func captureUnnamedMembers(
	body []byte,
	targetType reflect.Type,
	format llmprotocol.WireFormat,
) *llmprotocol.UnmodeledFields {
	targetType = dereferenceJSONType(targetType)
	if targetType == nil {
		return nil
	}
	if targetType.Kind() == reflect.Slice || targetType.Kind() == reflect.Array {
		return captureUnnamedElements(body, targetType.Elem(), format)
	}
	if targetType.Kind() != reflect.Struct {
		return nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil
	}
	named := exactJSONStructFields(targetType)
	captured := &llmprotocol.UnmodeledFields{Format: format}
	for name, value := range object {
		fieldType, isNamed := named[name]
		if !isNamed {
			if captured.Fields == nil {
				captured.Fields = make(map[string]json.RawMessage)
			}
			captured.Fields[name] = append(json.RawMessage(nil), value...)
			continue
		}
		child := captureUnnamedMembers(value, fieldType, format)
		if child == nil {
			continue
		}
		if captured.Children == nil {
			captured.Children = make(map[string]*llmprotocol.UnmodeledFields)
		}
		captured.Children[name] = child
	}
	if captured.Empty() {
		return nil
	}
	return captured
}

// captureUnnamedElements walks an array of modelled objects and keeps each
// element's carrier at its own index. An array of anything else -- a string
// list, or a json.RawMessage, which is a byte slice -- is carried whole by the
// member that holds it and stops here.
func captureUnnamedElements(
	body []byte,
	elementType reflect.Type,
	format llmprotocol.WireFormat,
) *llmprotocol.UnmodeledFields {
	element := dereferenceJSONType(elementType)
	if element == nil || element.Kind() != reflect.Struct {
		return nil
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(body, &elements); err != nil {
		return nil
	}
	captured := &llmprotocol.UnmodeledFields{
		Format:   format,
		Elements: make([]*llmprotocol.UnmodeledFields, len(elements)),
	}
	for index, value := range elements {
		captured.Elements[index] = captureUnnamedMembers(value, elementType, format)
	}
	if captured.Empty() {
		return nil
	}
	return captured
}

// mergeUnnamedMembers writes the carried members back into an encoded object.
// A member the encoder already claimed is left alone: the Router may have
// changed it, and the carried copy is what arrived rather than what is being
// sent.
//
// A parent the encoder did not write is rebuilt to hold its carried members.
// The encoder writes a parent only when it has something modelled to put
// there -- Anthropic metadata only for an end-user ID -- so a request whose
// only metadata is a carried one used to encode with no metadata at all, and
// the same-format drop count skips what the format is supposed to carry. The
// member is not invented: nothing is carried under a parent the client did not
// send.
func mergeUnnamedMembers(object map[string]json.RawMessage, carried *llmprotocol.UnmodeledFields) error {
	if carried == nil {
		return nil
	}
	for _, name := range sortedFieldNames(carried.Fields) {
		if _, claimed := object[name]; claimed {
			continue
		}
		object[name] = carried.Fields[name]
	}
	for _, name := range sortedChildNames(carried.Children) {
		if err := mergeCarriedChild(object, name, carried.Children[name]); err != nil {
			return err
		}
	}
	return nil
}

// mergeCarriedChild writes one carried child back into the member holding it,
// which is an object or an array of objects depending on what was captured.
func mergeCarriedChild(
	object map[string]json.RawMessage,
	name string,
	carried *llmprotocol.UnmodeledFields,
) error {
	if len(carried.Elements) > 0 {
		merged, replaced, err := mergeCarriedElements(object[name], carried)
		if err != nil {
			return err
		}
		if replaced {
			object[name] = merged
		}
		return nil
	}
	child, mergeable := carriedParentObject(object[name])
	if !mergeable {
		return nil
	}
	if err := mergeUnnamedMembers(child, carried); err != nil {
		return err
	}
	if len(child) == 0 {
		return nil
	}
	merged, err := marshalWire(child)
	if err != nil {
		return err
	}
	object[name] = merged
	return nil
}

// mergeCarriedElements writes each element's carried members back into the
// array the encoder produced. The indexes line up because a carried member
// re-emits only to the format it arrived in, and a same-format encode writes
// one element per element it read.
//
// An array the encoder did not write is left as it is. Rebuilding one from
// leftovers would produce elements holding no modelled content, which is the
// one thing the carrier must not invent. An element the encoder wrote as null
// or as an empty object is left alone for the same reason.
func mergeCarriedElements(
	claimed json.RawMessage,
	carried *llmprotocol.UnmodeledFields,
) (json.RawMessage, bool, error) {
	if len(claimed) == 0 {
		return nil, false, nil
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(claimed, &elements); err != nil {
		return nil, false, nil
	}
	for index, carriedElement := range carried.Elements {
		if carriedElement.Empty() || index >= len(elements) {
			continue
		}
		element, mergeable := carriedParentObject(elements[index])
		if !mergeable || len(element) == 0 {
			continue
		}
		if err := mergeUnnamedMembers(element, carriedElement); err != nil {
			return nil, false, err
		}
		merged, err := marshalWire(element)
		if err != nil {
			return nil, false, err
		}
		elements[index] = merged
	}
	merged, err := marshalWire(elements)
	if err != nil {
		return nil, false, err
	}
	return merged, true, nil
}

// carriedParentObject returns the object a carried child merges into. An
// absent parent, and a parent the encoder wrote as null, both become a fresh
// object holding the carried members alone.
//
// A parent the encoder wrote as something other than an object has nowhere to
// put them and keeps what the encoder wrote. No wire struct names a member as
// an object on the way in and a scalar on the way out, so that branch is a
// guard rather than a case.
func carriedParentObject(claimed json.RawMessage) (map[string]json.RawMessage, bool) {
	if len(claimed) == 0 {
		return map[string]json.RawMessage{}, true
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(claimed, &object); err != nil {
		return nil, false
	}
	if object == nil {
		return map[string]json.RawMessage{}, true
	}
	return object, true
}

// mergeUnnamedMembersInto re-encodes one already-marshalled object with its
// carried members put back. It is the shape every encoder needs: marshal the
// wire struct, then hand the bytes here.
func mergeUnnamedMembersInto(body []byte, carried *llmprotocol.UnmodeledFields) ([]byte, error) {
	if carried.Empty() {
		return body, nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "encode_wire", "wire request could not be encoded", err)
	}
	if err := mergeUnnamedMembers(object, carried); err != nil {
		return nil, err
	}
	return marshalWire(object)
}

// unnamedMemberPaths names every carried member by its dotted path, so a
// target that cannot express one counts it by the name the client wrote. The
// order is stable because a diagnostics list is compared in goldens.
func unnamedMemberPaths(carried *llmprotocol.UnmodeledFields, prefix string) []string {
	if carried == nil {
		return nil
	}
	paths := make([]string, 0, len(carried.Fields)+len(carried.Children)+len(carried.Elements))
	for _, name := range sortedFieldNames(carried.Fields) {
		paths = append(paths, prefix+name)
	}
	for _, name := range sortedChildNames(carried.Children) {
		paths = append(paths, unnamedMemberPaths(carried.Children[name], prefix+name+".")...)
	}
	// An element is named by its index, so a diagnostic says which message
	// carried the member rather than that some message did.
	for index, element := range carried.Elements {
		paths = append(paths, unnamedMemberPaths(element, prefix+strconv.Itoa(index)+".")...)
	}
	sort.Strings(paths)
	return paths
}

func sortedChildNames(children map[string]*llmprotocol.UnmodeledFields) []string {
	names := make([]string, 0, len(children))
	for name := range children {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// carriedForTarget reports whether a carried set survives to this target. A
// member of one wire contract carries no meaning in another, so only the
// format it arrived in re-emits it.
func carriedForTarget(carried *llmprotocol.UnmodeledFields, target llmprotocol.WireFormat) bool {
	return !carried.Empty() && carried.Format == target
}

// appendUnnamedMemberDrops counts what a target cannot express. It is the half
// of accept-by-default that keeps the loss visible: production carries the
// field, and the diagnostic is what tells an operator it did.
func appendUnnamedMemberDrops(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	carried *llmprotocol.UnmodeledFields,
	target llmprotocol.WireFormat,
	prefix string,
) {
	if carried.Empty() || carried.Format == target {
		return
	}
	for _, path := range unnamedMemberPaths(carried, prefix) {
		appendPresentationDrop(
			diagnostics, policy, carried.Format, target, path,
			"the target wire format does not name this member",
		)
	}
}

// appendContentExtensionDrops counts the carried members of one content list,
// and of the tool results nested inside it, that this target cannot express.
func appendContentExtensionDrops(
	diagnostics *llmprotocol.Diagnostics,
	contents []llmprotocol.Content,
	target llmprotocol.WireFormat,
	policy llmprotocol.Policy,
) {
	for _, content := range contents {
		appendUnnamedMemberDrops(diagnostics, policy, content.Extensions, target, "content.")
		if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil {
			appendContentExtensionDrops(diagnostics, content.ToolResult.Content, target, policy)
		}
	}
}

// appendToolExtensionDrops counts the carried members of each tool definition
// that this target cannot express.
func appendToolExtensionDrops(
	diagnostics *llmprotocol.Diagnostics,
	tools []llmprotocol.Tool,
	target llmprotocol.WireFormat,
	policy llmprotocol.Policy,
) {
	for _, tool := range tools {
		appendUnnamedMemberDrops(diagnostics, policy, tool.Extensions, target, "tools.")
	}
}
