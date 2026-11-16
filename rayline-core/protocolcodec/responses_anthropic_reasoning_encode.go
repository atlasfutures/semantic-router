package protocolcodec

import (
	"encoding/json"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// appendClaudeReasoningItem adds one of Claude's blocks, as its
// anthropic-claude-v1 reasoning_details item, after the items details already
// holds, numbered in order as OpenRouter numbers them.
func appendClaudeReasoningItem(details json.RawMessage, item map[string]any) (json.RawMessage, error) {
	var items []json.RawMessage
	if hasJSONValue(details) {
		if err := json.Unmarshal(details, &items); err != nil {
			return nil, err
		}
	}
	item["index"] = len(items)
	raw, err := json.Marshal(item)
	if err != nil {
		return nil, err
	}
	return json.Marshal(append(items, raw))
}

// mintedClaudeReasoning is the encrypted_content a Responses client gets for
// one of Claude's blocks: a Router-minted anthropic-claude-v1 reasoning_details
// array holding the single item (see responses_anthropic_reasoning.go).
func mintedClaudeReasoning(item map[string]any) json.RawMessage {
	details, err := appendClaudeReasoningItem(nil, item)
	if err != nil {
		return nil
	}
	return mintReasoningDetails(details)
}

// responsesRedactedThinkingItem is the reasoning item a carried Anthropic
// redacted_thinking block becomes, or false when the content is not one. For
// a client (minted) the block's data rides a minted encrypted_content as a
// reasoning.encrypted item; for a Claude worker it is OpenRouter's shape, the
// data as encrypted_content under the Anthropic format.
func responsesRedactedThinkingItem(content llmprotocol.Content, id string, minted bool) (responsesItemWire, bool) {
	data, ok := OpaqueReasoningData(content)
	if !ok {
		return responsesItemWire{}, false
	}
	item := responsesItemWire{Type: "reasoning", ID: id}
	if minted {
		item.EncryptedContent = mintedClaudeReasoning(claudeReasoningEncryptedItem(data, 0))
		return item, true
	}
	item.EncryptedContent, _ = json.Marshal(data)
	item.ReasoningFormat = responsesAnthropicReasoningFormat
	return item, true
}

// recordReasoningSignature keeps a reasoning event's signature for the item it
// belongs to. Anthropic sends it in one signature_delta after the thinking,
// and a split one is joined in order.
func (encoder *responsesStreamEncoder) recordReasoningSignature(event llmprotocol.Event) {
	if event.Content != nil && event.Content.Signature != "" {
		encoder.reasoningSignatures[event.ItemIndex] += event.Content.Signature
	}
}

// encodeResponsesRedactedThinking writes a completed redacted_thinking block
// as the reasoning item a buffered response gives it, added and done at once:
// the block arrives whole, so there is nothing to stream.
func (encoder *responsesStreamEncoder) encodeResponsesRedactedThinking(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	frames, key, err := encoder.ensureResponsesOutputStarted(event, responsesOutputReasoning)
	if err != nil {
		return nil, nil, err
	}
	item, _ := responsesRedactedThinkingItem(*event.Content, encoder.outputIDs[key], true)
	item.Status = "completed"
	index := encoder.outputIndexes[key]
	wire := responsesEventWire{
		Type: "response.output_item.done", Sequence: encoder.nextWireSequence(),
		OutputIndex: responsesOutputIndex(index), Item: marshalResponsesEventItem(item),
	}
	encoder.recordResponsesCompletedOutput(index, wire.Item)
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	return append(frames, frame), nil, err
}
