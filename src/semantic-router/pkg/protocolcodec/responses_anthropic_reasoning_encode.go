package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// responsesRedactedThinkingItem is the reasoning item a carried Anthropic
// redacted_thinking block becomes for a Responses client, or false when the
// content is not one. See responses_anthropic_reasoning.go for the shape.
func responsesRedactedThinkingItem(content llmprotocol.Content, id string) (responsesItemWire, bool) {
	data, ok := OpaqueReasoningData(content)
	if !ok {
		return responsesItemWire{}, false
	}
	encrypted, _ := json.Marshal(data)
	return responsesItemWire{
		Type: "reasoning", ID: id, EncryptedContent: encrypted,
		ReasoningFormat: responsesAnthropicReasoningFormat,
	}, true
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
	item, _ := responsesRedactedThinkingItem(*event.Content, encoder.outputIDs[key])
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
