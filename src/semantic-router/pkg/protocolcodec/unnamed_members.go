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
// sent. A child whose parent the encoder did not write is skipped, because a
// struct built from its leftovers alone would be a member no client sent.
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
		claimed, present := object[name]
		if !present {
			continue
		}
		var child map[string]json.RawMessage
		if err := json.Unmarshal(claimed, &child); err != nil || child == nil {
			continue
		}
		if err := mergeUnnamedMembers(child, carried.Children[name]); err != nil {
			return err
		}
		merged, err := marshalWire(child)
		if err != nil {
			return err
		}
		object[name] = merged
	}
	return nil
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

// appendServerToolDrops counts the tool declarations a target cannot express
// at all. A server tool is run by the source API, so a format with no such
// tool has nowhere to put the declaration and no way to honour it.
func appendServerToolDrops(
	diagnostics *llmprotocol.Diagnostics,
	tools []llmprotocol.Tool,
	source, target llmprotocol.WireFormat,
	policy llmprotocol.Policy,
) {
	for _, tool := range tools {
		if !tool.ServerTool() {
			continue
		}
		appendPresentationDrop(
			diagnostics, policy, source, target, "tools.type",
			"the target wire format has no server-run tool",
		)
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
