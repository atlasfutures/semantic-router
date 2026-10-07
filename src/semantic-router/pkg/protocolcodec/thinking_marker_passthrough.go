package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// thinkingMarkerPassthrough mints Router thinking markers into a same-format
// Messages stream that otherwise travels as the upstream wrote it
// (AnthropicPublicStreamFilter). A thinking block that reaches its
// content_block_stop with text and no signature is given one signature_delta
// carrying the marker, where Anthropic sends a signature: after the thinking,
// before the stop. Every upstream frame still travels byte for byte.
type thinkingMarkerPassthrough struct {
	family string
	blocks map[int]*passthroughThinking
}

type passthroughThinking struct {
	text   strings.Builder
	signed bool
}

type passthroughBlockWire struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
}

type passthroughEventWire struct {
	Type         string                `json:"type"`
	Index        *int                  `json:"index"`
	ContentBlock *passthroughBlockWire `json:"content_block"`
	Delta        *passthroughBlockWire `json:"delta"`
}

// before returns the frames to send ahead of this one: the marker's
// signature_delta when the frame stops an unsigned thinking block.
func (mint *thinkingMarkerPassthrough) before(parsed sseFrame) ([]byte, error) {
	if mint == nil || mint.family == "" || !parsed.HasData || !bytes.Contains(parsed.Data, []byte(`content_block`)) {
		return nil, nil
	}
	var wire passthroughEventWire
	if json.Unmarshal(parsed.Data, &wire) != nil || wire.Index == nil {
		return nil, nil
	}
	eventType := wire.Type
	if eventType == "" {
		eventType = parsed.Event
	}
	index := *wire.Index
	switch eventType {
	case "content_block_start":
		if wire.ContentBlock != nil && wire.ContentBlock.Type == "thinking" {
			block := &passthroughThinking{signed: wire.ContentBlock.Signature != ""}
			block.text.WriteString(wire.ContentBlock.Thinking)
			if mint.blocks == nil {
				mint.blocks = map[int]*passthroughThinking{}
			}
			mint.blocks[index] = block
		}
	case "content_block_delta":
		block := mint.blocks[index]
		if block == nil || wire.Delta == nil {
			return nil, nil
		}
		switch wire.Delta.Type {
		case "thinking_delta":
			block.text.WriteString(wire.Delta.Thinking)
		case "signature_delta":
			block.signed = block.signed || wire.Delta.Signature != ""
		}
	case "content_block_stop":
		block := mint.blocks[index]
		if block == nil {
			return nil, nil
		}
		delete(mint.blocks, index)
		if block.signed || block.text.Len() == 0 {
			return nil, nil
		}
		frames, err := appendAnthropicReasoningSignature(nil, index, &llmprotocol.Content{
			Signature: ThinkingMarker(mint.family, block.text.String()),
		})
		if err != nil {
			return nil, err
		}
		return bytes.Join(frames, nil), nil
	}
	return nil, nil
}
