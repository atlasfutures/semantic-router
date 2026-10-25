package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strconv"
)

// anthropicBlockOrderBufferLimit bounds the frames anthropicBlockOrder holds
// back while it waits for an open block to stop.
const anthropicBlockOrderBufferLimit = 32 << 20

// anthropicBlockOrder puts a same-format Anthropic stream's content blocks
// back into the order Messages defines: one block open at a time, each block
// stopped before the next starts, indexes contiguous from zero in start order.
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
// answered.
//
// The pass serializes blocks. The first block to start is live and streams
// as it arrives. A block that starts while another is live is held, with its
// frames and every frame after it that belongs to no live block, until the
// live block's own stop; then the held blocks follow in start order. No block
// is ever stopped for the upstream, so a late delta (a thinking block's
// signature_delta, a tool_use block's input_json_delta) still lands in the
// block it belongs to. A client index is given when a block goes live, so
// indexes follow start order; a frame whose index needs no change, and every
// frame of a stream already in order, travels byte for byte.
type anthropicBlockOrder struct {
	next   int
	live   int // upstream index of the live block, or -1
	blocks map[int]*anthropicOrderBlock
	held   []anthropicOrderFrame
	bytes  int
}

type anthropicOrderBlock struct {
	client int
	state  anthropicOrderState
}

type anthropicOrderState int

const (
	anthropicOrderHeld anthropicOrderState = iota
	anthropicOrderLive
	anthropicOrderDone
)

type anthropicOrderFrame struct {
	raw    []byte
	parsed sseFrame
	// block is the upstream index of the content block the frame belongs
	// to, or -1 for a frame that belongs to none.
	block int
	stop  bool
}

type anthropicBlockOrderWire struct {
	Type  string `json:"type"`
	Index *int   `json:"index"`
}

func newAnthropicBlockOrder() *anthropicBlockOrder {
	return &anthropicBlockOrder{live: -1, blocks: map[int]*anthropicOrderBlock{}}
}

// push returns the frames the client receives for one upstream frame.
func (order *anthropicBlockOrder) push(raw []byte, parsed sseFrame) ([][]byte, error) {
	frame := anthropicOrderFrame{raw: raw, parsed: parsed, block: -1}
	var wire anthropicBlockOrderWire
	if parsed.HasData && json.Unmarshal(parsed.Data, &wire) == nil {
		eventType := wire.Type
		if eventType == "" {
			eventType = parsed.Event
		}
		switch eventType {
		case "content_block_start", "content_block_delta", "content_block_stop":
			if wire.Index != nil {
				frame.block = *wire.Index
				frame.stop = eventType == "content_block_stop"
				if eventType == "content_block_start" {
					if block := order.blocks[frame.block]; block == nil || block.state == anthropicOrderDone {
						order.blocks[frame.block] = &anthropicOrderBlock{state: anthropicOrderHeld}
					}
				}
			}
		case "error":
			// Nothing valid follows an error: what is held goes first.
			out, err := order.drain()
			if err != nil {
				return nil, err
			}
			return append(out, raw), nil
		}
	}
	if block := order.blocks[frame.block]; frame.block >= 0 && block != nil && block.state == anthropicOrderHeld {
		return order.hold(frame)
	}
	if frame.block < 0 || order.blocks[frame.block] == nil {
		// A frame of no block, or of a block never started: it keeps its
		// place behind anything held.
		if len(order.held) > 0 {
			return order.hold(frame)
		}
		return [][]byte{raw}, nil
	}
	// A frame of the live block, or a late one of a block already done.
	out, err := order.emit(frame)
	if err != nil {
		return nil, err
	}
	if order.live >= 0 {
		return out, nil
	}
	rest, err := order.release()
	if err != nil {
		return nil, err
	}
	return append(out, rest...), nil
}

// finish returns the frames still held when the stream ends: the upstream
// stopped without stopping the live block.
func (order *anthropicBlockOrder) finish() ([][]byte, error) {
	return order.drain()
}

func (order *anthropicBlockOrder) hold(frame anthropicOrderFrame) ([][]byte, error) {
	if order.live < 0 && len(order.held) == 0 {
		// Nothing is live: a held block's start goes live now.
		order.held = append(order.held, frame)
		order.bytes += len(frame.raw)
		return order.release()
	}
	order.bytes += len(frame.raw)
	if order.bytes > anthropicBlockOrderBufferLimit {
		return nil, invalidProviderResponse("stream_block_order_buffer", "Anthropic stream holds too much behind an open content block")
	}
	order.held = append(order.held, frame)
	return nil, nil
}

// emit returns one frame of a live or done block, renumbered, and ends the
// live block on its stop.
func (order *anthropicBlockOrder) emit(frame anthropicOrderFrame) ([][]byte, error) {
	block := order.blocks[frame.block]
	out, err := order.renumbered(frame, block.client)
	if err != nil {
		return nil, err
	}
	if frame.stop && block.state == anthropicOrderLive {
		block.state = anthropicOrderDone
		order.live = -1
	}
	return [][]byte{out}, nil
}

// release sends what is held once no block is live: frames of no block up to
// the next held block, that block (which goes live), and its frames so far,
// until a block stays live or nothing is held.
func (order *anthropicBlockOrder) release() ([][]byte, error) {
	var out [][]byte
	for len(order.held) > 0 {
		if order.live < 0 {
			frame := order.held[0]
			order.held = order.held[1:]
			order.bytes -= len(frame.raw)
			block := order.blocks[frame.block]
			switch {
			case frame.block < 0 || block == nil:
				out = append(out, frame.raw)
				continue
			case block.state == anthropicOrderHeld:
				block.state = anthropicOrderLive
				block.client = order.next
				order.next++
				order.live = frame.block
			}
			emitted, err := order.emit(frame)
			if err != nil {
				return nil, err
			}
			out = append(out, emitted...)
			continue
		}
		kept := order.held[:0]
		for position, frame := range order.held {
			if order.live < 0 {
				kept = append(kept, order.held[position:]...)
				break
			}
			if frame.block != order.live {
				kept = append(kept, frame)
				continue
			}
			order.bytes -= len(frame.raw)
			emitted, err := order.emit(frame)
			if err != nil {
				return nil, err
			}
			out = append(out, emitted...)
		}
		order.held = kept
		if order.live >= 0 {
			break
		}
	}
	return out, nil
}

// drain sends everything held, block by block in start order, without
// waiting for the live block's stop: the stream ends or fails here.
func (order *anthropicBlockOrder) drain() ([][]byte, error) {
	var out [][]byte
	for {
		released, err := order.release()
		if err != nil {
			return nil, err
		}
		out = append(out, released...)
		if len(order.held) == 0 {
			return out, nil
		}
		// The live block never stopped: the next held block follows it.
		order.blocks[order.live].state = anthropicOrderDone
		order.live = -1
	}
}

// renumbered is the frame with its index set to the block's client index:
// raw, byte for byte, when that index is the upstream's own.
func (order *anthropicBlockOrder) renumbered(frame anthropicOrderFrame, client int) ([]byte, error) {
	if client == frame.block {
		return frame.raw, nil
	}
	data, ok := withTopLevelIndex(frame.parsed.Data, client)
	if !ok {
		return nil, invalidProviderResponse("stream_output_index_order", "Anthropic content block frame has no index to renumber")
	}
	var buffer bytes.Buffer
	if frame.parsed.Event != "" {
		buffer.WriteString("event: ")
		buffer.WriteString(frame.parsed.Event)
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
