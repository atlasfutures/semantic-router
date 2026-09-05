package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// This file holds the one mechanism that keeps a request the Router only routes
// from being refused for a part the neutral contract does not name: the
// carrier. Decoders put source bytes into it, encoders take them out again for
// the format they came from, and every other target drops them and says so.

// anthropicVariantIsUnnamed reports whether a block the contract does name
// holds a variant it does not. Those blocks are carried whole for the same
// reason as an unnamed block type: the source API accepts the body, the Router
// does not read the part in question, and refusing it loses the conversation.
func anthropicVariantIsUnnamed(typeName string, body json.RawMessage) bool {
	var block struct {
		Input  json.RawMessage `json:"input"`
		Source *struct {
			Type string `json:"type"`
		} `json:"source"`
	}
	if err := json.Unmarshal(body, &block); err != nil {
		return false
	}
	switch typeName {
	case "tool_use":
		// The Anthropic API types tool_use.input as arbitrary JSON; the neutral
		// tool call holds one object.
		trimmed := bytes.TrimSpace(block.Input)
		return len(trimmed) > 0 && trimmed[0] != '{'
	case "document":
		// A text or content document source is a document the neutral contract
		// has no media reference for.
		return block.Source != nil && (block.Source.Type == "text" || block.Source.Type == "content")
	default:
		return false
	}
}

// anthropicModelledRequestContent lists the request blocks the neutral contract
// names. Every other block is carried whole, because the Router routes the
// conversation rather than reading it.
var anthropicModelledRequestContent = map[string]bool{
	"text": true, "thinking": true, "image": true,
	"document": true, "tool_use": true, "tool_result": true,
}

func carriedAnthropicBlock(typeName string, body json.RawMessage) llmprotocol.Content {
	return llmprotocol.Content{
		Kind: llmprotocol.ContentUnmodeled,
		Unmodeled: &llmprotocol.UnmodeledBlock{
			Format: llmprotocol.AnthropicMessagesV1,
			Type:   typeName,
			Raw:    append(json.RawMessage(nil), body...),
		},
	}
}

// decodeWireCapturingUnmodeled decodes a client request object and returns the
// members the wire struct does not name instead of refusing them. Every other
// document rule still applies -- size, UTF-8, duplicate keys, depth, trailing
// data -- and the members the struct does name are still decoded strictly,
// including their nested values. Only the top level of the request object is
// widened, and the bytes of every member are left exactly as the client sent
// them.
func decodeWireCapturingUnmodeled(
	body []byte,
	target any,
	policy llmprotocol.Policy,
) (map[string]json.RawMessage, error) {
	if err := validateClientJSONDocument(body, policy, true); err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json", "request JSON is invalid", err)
	}
	named := exactJSONStructFields(dereferenceJSONType(reflect.TypeOf(target)))
	strict := rejectUnknownFields(body, policy)
	var unmodeled map[string]json.RawMessage
	for name, value := range object {
		fieldType, found := named[name]
		if !found {
			if unmodeled == nil {
				unmodeled = make(map[string]json.RawMessage)
			}
			unmodeled[name] = value
			continue
		}
		if !strict {
			continue
		}
		if err := validateExactJSONValue(value, dereferenceJSONType(fieldType)); err != nil {
			return nil, llmprotocol.NewError(
				llmprotocol.ErrorInvalidRequest, "invalid_json",
				"request JSON contains a non-canonical field",
				fmt.Errorf("field %q: %w", name, err),
			)
		}
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	if err := decoder.Decode(target); err != nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json", "request JSON is invalid", err)
	}
	if err := requireEOF(decoder); err != nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "trailing_json", "request body contains trailing JSON", err)
	}
	return unmodeled, nil
}

// unmodeledRequestFields builds the carrier for one decoded request. An empty
// set yields no carrier, so a body with nothing unnamed decodes exactly as it
// did before.
func unmodeledRequestFields(
	format llmprotocol.WireFormat,
	fields map[string]json.RawMessage,
) *llmprotocol.UnmodeledFields {
	if len(fields) == 0 {
		return nil
	}
	return &llmprotocol.UnmodeledFields{Format: format, Fields: fields}
}

// mergeUnmodeledFields re-emits carried members into an encoded request body.
// The carrier survives only to the wire format it came from; any other target
// drops it and records why, because dropping a member the target cannot name
// is the routable outcome and refusing the request is not.
func mergeUnmodeledFields(
	body []byte,
	request llmprotocol.Request,
	target llmprotocol.WireFormat,
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
) ([]byte, error) {
	carrier := request.Unmodeled
	if carrier.Len() == 0 {
		return body, nil
	}
	if carrier.Format != target {
		for _, name := range sortedFieldNames(carrier.Fields) {
			appendUnmodeledDrop(diagnostics, policy, carrier.Format, target, name)
		}
		return body, nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "encode_wire", "wire request could not be encoded", err)
	}
	for name, value := range carrier.Fields {
		if _, claimed := object[name]; claimed {
			continue
		}
		object[name] = value
	}
	return marshalWire(object)
}

func appendUnmodeledDrop(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	source, target llmprotocol.WireFormat,
	field string,
) {
	if diagnostics == nil || len(*diagnostics) >= policy.Limits.Diagnostics {
		return
	}
	*diagnostics = append(*diagnostics, llmprotocol.Diagnostic{
		Source: source, Target: target, Field: field,
		Action: llmprotocol.DiagnosticDropped,
		Reason: "the target wire format does not name this member",
	})
}

func sortedFieldNames(fields map[string]json.RawMessage) []string {
	names := make([]string, 0, len(fields))
	for name := range fields {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// carriedBlockBytes returns the source bytes of a carried block when the target
// wire format is the one it came from. Any other target drops it: the block
// names something only its own contract understands, and losing it is the
// routable outcome where refusing the conversation is not.
func carriedBlockBytes(content llmprotocol.Content, target llmprotocol.WireFormat) (json.RawMessage, bool) {
	carried := content.Unmodeled
	if carried == nil || carried.Format != target || len(carried.Raw) == 0 {
		return nil, false
	}
	return carried.Raw, true
}

// appendCarriedBlockDrops records the carried blocks one encode will drop. It
// reads the block list it is given and does not descend into tool results,
// which carry their own list through the same encoder.
func appendCarriedBlockDrops(
	diagnostics *llmprotocol.Diagnostics,
	contents []llmprotocol.Content,
	target llmprotocol.WireFormat,
	policy llmprotocol.Policy,
) {
	for _, content := range contents {
		if content.Kind != llmprotocol.ContentUnmodeled || content.Unmodeled == nil {
			continue
		}
		if content.Unmodeled.Format == target {
			continue
		}
		appendUnmodeledDrop(
			diagnostics, policy, content.Unmodeled.Format, target,
			"content."+content.Unmodeled.Type,
		)
	}
}

// carriedAnthropicCitations copies the citations member a request block sets,
// treating an absent member and an explicit null alike. The bytes are carried
// verbatim so a Messages target returns exactly what the client sent.
func carriedAnthropicCitations(raw json.RawMessage) json.RawMessage {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}

// appendCitationCarryDrops records the citations member a text or document
// block carries when the target format has no field for it. Claude Code
// echoes the citations of a web-search or document answer back in history on
// every later turn, so refusing them failed real turns on the dev cell on
// 2026-09-05. The member says where an answer came from rather than what the
// model is asked, so the turn runs and the loss is counted.
func appendCitationCarryDrops(
	diagnostics *llmprotocol.Diagnostics,
	contents []llmprotocol.Content,
	source, target llmprotocol.WireFormat,
	policy llmprotocol.Policy,
	reason string,
) {
	for _, content := range contents {
		if len(content.CitationsRaw) == 0 {
			continue
		}
		appendPresentationDrop(diagnostics, policy, source, target, "content.citations", reason)
	}
}

// messageDropsWhole reports whether every block of a message is a carried block
// the target cannot name. Such a message encodes to nothing, so the encoder
// omits the message rather than emitting an empty one the provider refuses.
func messageDropsWhole(contents []llmprotocol.Content, target llmprotocol.WireFormat) bool {
	if len(contents) == 0 {
		return false
	}
	for _, content := range contents {
		if content.Kind != llmprotocol.ContentUnmodeled {
			return false
		}
		if _, kept := carriedBlockBytes(content, target); kept {
			return false
		}
	}
	return true
}

// hasUnnamedMembers reports whether a JSON object holds members the wire struct
// does not name.
func hasUnnamedMembers(body []byte, target any) bool {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return false
	}
	named := exactJSONStructFields(dereferenceJSONType(reflect.TypeOf(target)))
	for name := range object {
		if _, found := named[name]; !found {
			return true
		}
	}
	return false
}

// carriedItemMessage wraps one carried input item as a message. The role never
// reaches a provider: the message is either re-emitted as the item it came
// from, or dropped whole by a target that cannot name it.
func carriedItemMessage(format llmprotocol.WireFormat, itemType string, body json.RawMessage) llmprotocol.Message {
	return llmprotocol.Message{
		Role: llmprotocol.RoleUser,
		Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentUnmodeled,
			Unmodeled: &llmprotocol.UnmodeledBlock{
				Format: format, Type: itemType,
				Raw: append(json.RawMessage(nil), body...),
			},
		}},
	}
}

// carriedItemBytes returns the source item of a message that holds exactly one
// carried block for the target format.
func carriedItemBytes(message llmprotocol.Message, target llmprotocol.WireFormat) (json.RawMessage, bool) {
	if len(message.Content) != 1 {
		return nil, false
	}
	return carriedBlockBytes(message.Content[0], target)
}

// withoutCarriedContent removes carried blocks from a content list. A carrier
// is re-emitted as the item or block it came from, never as a content part of
// another contract, so a content encoder has nothing to do with one.
func withoutCarriedContent(contents []llmprotocol.Content) []llmprotocol.Content {
	kept := make([]llmprotocol.Content, 0, len(contents))
	for _, content := range contents {
		if content.Kind == llmprotocol.ContentUnmodeled {
			continue
		}
		kept = append(kept, content)
	}
	return kept
}
