package protocolcodec

import (
	"encoding/json"
	"reflect"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// This file holds the depth half of accept-by-default. The request carrier in
// unmodeled.go widens the top level of a request object; this widens every
// modelled struct beneath it, with one walk shared by the request envelope,
// each content block and each tool.
//
// The walk descends into a member the struct does name only when that member
// is itself a modelled object. It stops at an array and at json.RawMessage,
// because both are already carried verbatim by the member that holds them.

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
	if targetType == nil || targetType.Kind() != reflect.Struct {
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
		child, mergeable := carriedParentObject(object[name])
		if !mergeable {
			continue
		}
		if err := mergeUnnamedMembers(child, carried.Children[name]); err != nil {
			return err
		}
		if len(child) == 0 {
			continue
		}
		merged, err := marshalWire(child)
		if err != nil {
			return err
		}
		object[name] = merged
	}
	return nil
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
	paths := make([]string, 0, len(carried.Fields)+len(carried.Children))
	for _, name := range sortedFieldNames(carried.Fields) {
		paths = append(paths, prefix+name)
	}
	for _, name := range sortedChildNames(carried.Children) {
		paths = append(paths, unnamedMemberPaths(carried.Children[name], prefix+name+".")...)
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
