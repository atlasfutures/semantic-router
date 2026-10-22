package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strconv"
)

// anthropicBlockOrder puts a same-format Anthropic stream's content blocks
// back into the order Messages defines: one block open at a time, indexes
// contiguous from zero, each block stopped before the next starts.
//
// OpenRouter's Messages surface does not keep that order for OpenAI models
// (measured on openai/gpt-6-luna, 2026-10-22). It opens a thinking block for
// the reasoning summary, streams the reasoning's encrypted_content as a
// redacted_thinking block and the answer as a text block while the thinking
// block is still open, and stops the thinking block last:
//
//	start[0] thinking{"",""}  start[1] redacted_thinking  stop[1]
//	start[2] text  delta[2] "ok"  stop[2]  stop[0]
//
// A client that assembles blocks as they stop (Claude Code) then ends the
// turn on an empty thinking block, and its result is "" though the model
// answered. So this pass:
//
//   - holds a thinking block that starts empty and unsigned until its first
//     delta, and drops it, with its late stop, when another block starts, the
//     block stops, or the message ends first. Such a block carries nothing:
//     the reasoning a later turn needs travels in the redacted_thinking block,
//     and unsigned thinking is dropped for a Messages target on the next turn
//     anyway (CarryReasoningTo).
//   - renumbers every later start, delta and stop frame so the indexes the
//     client sees stay contiguous.
//   - stops a block still open when another block starts, and swallows that
//     block's late stop. A delta that arrives later for a stopped block
//     reopens it as a new block at the next index, started as it first was.
//
// Every frame the pass does not have to change travels byte for byte.
type anthropicBlockOrder struct {
	next int
	// live maps an upstream index to the client index of its open block.
	live map[int]int
	open *int
	held *heldAnthropicBlock
	// starts keeps each upstream block's start data, for a reopen.
	starts map[int][]byte
	// gone holds upstream blocks this pass stopped or dropped whose own stop
	// is still to come, and is swallowed.
	gone map[int]bool
}

type heldAnthropicBlock struct {
	index int
	raw   []byte
	frame sseFrame
}

type anthropicBlockOrderWire struct {
	Type         string `json:"type"`
	Index        *int   `json:"index"`
	ContentBlock *struct {
		Type      string  `json:"type"`
		Thinking  *string `json:"thinking"`
		Signature *string `json:"signature"`
	} `json:"content_block"`
}

func newAnthropicBlockOrder() *anthropicBlockOrder {
	return &anthropicBlockOrder{live: map[int]int{}, starts: map[int][]byte{}, gone: map[int]bool{}}
}

// push returns the frames the client receives for one upstream frame.
func (order *anthropicBlockOrder) push(raw []byte, parsed sseFrame) ([][]byte, error) {
	if !parsed.HasData {
		return [][]byte{raw}, nil
	}
	var wire anthropicBlockOrderWire
	if json.Unmarshal(parsed.Data, &wire) != nil {
		return [][]byte{raw}, nil
	}
	eventType := wire.Type
	if eventType == "" {
		eventType = parsed.Event
	}
	switch eventType {
	case "content_block_start":
		if wire.Index == nil {
			return [][]byte{raw}, nil
		}
		return order.start(raw, parsed, *wire.Index, emptyUnsignedThinking(wire))
	case "content_block_delta":
		if wire.Index == nil {
			return [][]byte{raw}, nil
		}
		return order.delta(raw, parsed, *wire.Index)
	case "content_block_stop":
		if wire.Index == nil {
			return [][]byte{raw}, nil
		}
		return order.stop(raw, parsed, *wire.Index)
	case "message_delta", "message_stop":
		// The message ends: a thinking block still held never had anything.
		order.dropHeld()
	}
	return [][]byte{raw}, nil
}

func emptyUnsignedThinking(wire anthropicBlockOrderWire) bool {
	block := wire.ContentBlock
	return block != nil && block.Type == "thinking" &&
		(block.Thinking == nil || *block.Thinking == "") &&
		(block.Signature == nil || *block.Signature == "")
}

func (order *anthropicBlockOrder) start(raw []byte, parsed sseFrame, index int, holdable bool) ([][]byte, error) {
	order.dropHeld()
	frames, err := order.stopOpen()
	if err != nil {
		return nil, err
	}
	order.starts[index] = append([]byte(nil), parsed.Data...)
	delete(order.gone, index)
	if holdable {
		order.held = &heldAnthropicBlock{index: index, raw: raw, frame: parsed}
		return frames, nil
	}
	started, err := order.begin(raw, parsed, index)
	if err != nil {
		return nil, err
	}
	return append(frames, started), nil
}

func (order *anthropicBlockOrder) delta(raw []byte, parsed sseFrame, index int) ([][]byte, error) {
	var frames [][]byte
	switch {
	case order.held != nil && order.held.index == index:
		held := order.held
		order.held = nil
		started, err := order.begin(held.raw, held.frame, index)
		if err != nil {
			return nil, err
		}
		frames = append(frames, started)
	case order.isOpen(index):
	case order.gone[index]:
		// A delta for a block already stopped: Messages cannot reopen it,
		// so it continues as a new block at the next index.
		order.dropHeld()
		stopped, err := order.stopOpen()
		if err != nil {
			return nil, err
		}
		frames = append(frames, stopped...)
		delete(order.gone, index)
		started, err := order.begin(nil, sseFrame{Event: "content_block_start", Data: order.starts[index], HasData: true}, index)
		if err != nil {
			return nil, err
		}
		frames = append(frames, started)
	default:
		// A block this pass never saw start travels as it is.
		return [][]byte{raw}, nil
	}
	frame, err := order.renumbered(raw, parsed, index)
	if err != nil {
		return nil, err
	}
	return append(frames, frame), nil
}

func (order *anthropicBlockOrder) stop(raw []byte, parsed sseFrame, index int) ([][]byte, error) {
	switch {
	case order.held != nil && order.held.index == index:
		// An empty unsigned thinking block that stopped with nothing in it.
		order.held = nil
		return nil, nil
	case order.isOpen(index):
		frame, err := order.renumbered(raw, parsed, index)
		if err != nil {
			return nil, err
		}
		delete(order.live, index)
		order.open = nil
		return [][]byte{frame}, nil
	case order.gone[index]:
		delete(order.gone, index)
		return nil, nil
	default:
		return [][]byte{raw}, nil
	}
}

// begin assigns an upstream block the next client index and returns its
// start frame: raw itself when the index is unchanged.
func (order *anthropicBlockOrder) begin(raw []byte, parsed sseFrame, index int) ([]byte, error) {
	order.live[index] = order.next
	order.next++
	upstream := index
	order.open = &upstream
	return order.renumbered(raw, parsed, index)
}

func (order *anthropicBlockOrder) isOpen(index int) bool {
	return order.open != nil && *order.open == index
}

// stopOpen stops the open block, whose own stop is then swallowed.
func (order *anthropicBlockOrder) stopOpen() ([][]byte, error) {
	if order.open == nil {
		return nil, nil
	}
	index := *order.open
	frame, err := encodeSSE("content_block_stop", anthropicEventWire{Type: "content_block_stop", Index: anthropicIndex(order.live[index])})
	if err != nil {
		return nil, err
	}
	delete(order.live, index)
	order.open = nil
	order.gone[index] = true
	return [][]byte{frame}, nil
}

func (order *anthropicBlockOrder) dropHeld() {
	if order.held == nil {
		return
	}
	order.gone[order.held.index] = true
	order.held = nil
}

// renumbered is the frame with its index set to the block's client index:
// raw, byte for byte, when that index is the upstream's own.
func (order *anthropicBlockOrder) renumbered(raw []byte, parsed sseFrame, index int) ([]byte, error) {
	client := order.live[index]
	if raw != nil && client == index {
		return raw, nil
	}
	data, ok := withTopLevelIndex(parsed.Data, client)
	if !ok {
		return nil, invalidProviderResponse("stream_output_index_order", "Anthropic content block frame has no index to renumber")
	}
	var buffer bytes.Buffer
	if parsed.Event != "" {
		buffer.WriteString("event: ")
		buffer.WriteString(parsed.Event)
		buffer.WriteByte('\n')
	}
	buffer.WriteString("data: ")
	buffer.Write(data)
	buffer.WriteString("\n\n")
	return buffer.Bytes(), nil
}

// withTopLevelIndex replaces the value of an object's top-level "index"
// member, leaving every other byte of the object as it was.
func withTopLevelIndex(data []byte, index int) ([]byte, bool) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil, false
	}
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return nil, false
		}
		name, _ := key.(string)
		valueStart := decoder.InputOffset()
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, false
		}
		if name != "index" {
			continue
		}
		valueEnd := decoder.InputOffset()
		// The value read starts after the colon and any space before it.
		valueStart += int64(bytes.Index(data[valueStart:valueEnd], value))
		var out bytes.Buffer
		out.Write(data[:valueStart])
		out.WriteString(strconv.Itoa(index))
		out.Write(data[valueStart+int64(len(value)):])
		return out.Bytes(), true
	}
	return nil, false
}
