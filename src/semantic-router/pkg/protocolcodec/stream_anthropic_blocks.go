package protocolcodec

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// anthropicBlockKey identifies one Messages content block. A source content
// position owns one block until the position resumes after that block was
// stopped; the resume counter then carries the position onto a fresh block.
type anthropicBlockKey struct {
	position streamContentKey
	resume   int
}

func (encoder *anthropicStreamEncoder) ensureAnthropicBlockStarted(
	event llmprotocol.Event,
	kind llmprotocol.ContentKind,
) ([][]byte, anthropicBlockKey, error) {
	position := contentKey(event)
	key := encoder.liveBlockKey(position)
	if encoder.blockStarted[key] && !encoder.blockStopped[key] {
		if encoder.blocks[key] != kind {
			return nil, key, llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_content_kind_mismatch",
				"upstream stream changed a content block kind",
				nil,
			)
		}
		return nil, key, nil
	}
	if encoder.blockStarted[key] {
		// Messages cannot reopen a stopped content block. Serve the resumed
		// position as a further block of the same kind rather than failing a
		// stream whose earlier bytes already reached the client.
		key = encoder.resumeAnthropicBlock(position)
	}
	var frames [][]byte
	if encoder.hasActiveBlock && encoder.activeBlock != key {
		stopped, err := encoder.stopAnthropicBlock(encoder.activeBlock)
		if err != nil {
			return nil, key, err
		}
		frames = append(frames, stopped...)
	}
	encoder.blockStarted[key] = true
	encoder.blocks[key] = kind
	encoder.blockIndexes[key] = encoder.nextBlockIndex
	encoder.nextBlockIndex++
	encoder.itemBlockKeys[event.ItemIndex] = append(encoder.itemBlockKeys[event.ItemIndex], key)
	wire := encoder.encodeAnthropicItemStart(event, key, kind)
	frame, err := encodeSSE(wire.Type, wire)
	if err != nil {
		return nil, key, err
	}
	encoder.activeBlock = key
	encoder.hasActiveBlock = true
	return append(frames, frame), key, nil
}

// liveBlockKey names the block that currently carries a source content
// position. It is the position itself until the position has been resumed.
func (encoder *anthropicStreamEncoder) liveBlockKey(position streamContentKey) anthropicBlockKey {
	return anthropicBlockKey{position: position, resume: encoder.blockResumes[position]}
}

// resumeAnthropicBlock moves a source content position onto a fresh block.
func (encoder *anthropicStreamEncoder) resumeAnthropicBlock(position streamContentKey) anthropicBlockKey {
	encoder.blockResumes[position]++
	return encoder.liveBlockKey(position)
}

func (encoder *anthropicStreamEncoder) encodeAnthropicItemStart(
	event llmprotocol.Event,
	key anthropicBlockKey,
	kind llmprotocol.ContentKind,
) anthropicEventWire {
	block := &anthropicContentWire{Type: "text", Text: ""}
	if kind == llmprotocol.ContentToolCall {
		block.Type, block.ID, block.Name, block.Input = "tool_use", event.ToolCall.ID, event.ToolCall.Name, json.RawMessage(`{}`)
	} else if kind == llmprotocol.ContentReasoning {
		signature := ""
		if event.Content != nil {
			signature = event.Content.Signature
		}
		block.Type, block.Text, block.Thinking, block.Signature = "thinking", "", "", signature
	}
	return anthropicEventWire{Type: "content_block_start", Index: anthropicIndex(encoder.blockIndexes[key]), ContentBlock: block}
}

func (encoder *anthropicStreamEncoder) completeAnthropicItem(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	keys := append([]anthropicBlockKey(nil), encoder.itemBlockKeys[event.ItemIndex]...)
	var frames [][]byte
	if event.ToolCall != nil && len(keys) == 0 {
		started, key, err := encoder.ensureAnthropicBlockStarted(event, llmprotocol.ContentToolCall)
		if err != nil {
			return nil, nil, err
		}
		frames = append(frames, started...)
		delta := anthropicEventWire{
			Type: "content_block_delta", Index: anthropicIndex(encoder.blockIndexes[key]),
			Delta: &anthropicDeltaWire{Type: "input_json_delta", PartialJSON: event.ToolCall.Arguments},
		}
		deltaFrames, _, err := encodeAnthropicWireFrame(delta)
		if err != nil {
			return nil, nil, err
		}
		frames = append(frames, deltaFrames...)
		keys = append(keys, key)
	}
	if len(keys) == 0 {
		kind := llmprotocol.ContentText
		if event.ToolCall != nil {
			kind = llmprotocol.ContentToolCall
		} else if event.Content != nil && event.Content.Kind != "" {
			kind = event.Content.Kind
		}
		started, key, err := encoder.ensureAnthropicBlockStarted(event, kind)
		if err != nil {
			return nil, nil, err
		}
		frames = append(frames, started...)
		keys = append(keys, key)
	}
	for _, key := range keys {
		stopped, err := encoder.stopAnthropicBlock(key)
		if err != nil {
			return nil, nil, err
		}
		frames = append(frames, stopped...)
	}
	return frames, nil, nil
}

func (encoder *anthropicStreamEncoder) stopAnthropicBlock(key anthropicBlockKey) ([][]byte, error) {
	if !encoder.blockStarted[key] || encoder.blockStopped[key] {
		return nil, nil
	}
	wire := anthropicEventWire{Type: "content_block_stop", Index: anthropicIndex(encoder.blockIndexes[key])}
	frame, err := encodeSSE(wire.Type, wire)
	if err != nil {
		return nil, err
	}
	encoder.blockStopped[key] = true
	if encoder.hasActiveBlock && encoder.activeBlock == key {
		encoder.hasActiveBlock = false
	}
	return [][]byte{frame}, nil
}
