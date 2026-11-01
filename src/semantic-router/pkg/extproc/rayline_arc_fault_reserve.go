package extproc

import (
	"bytes"
	"encoding/json"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// The dev fault refusal_prelude_all refuses the re-served attempt too, so the
// exhaustion path of a refusal re-serve can be exercised on a cell: the first
// attempt refuses at its first visible content (as refusal_prelude), the turn
// is re-served once, and the re-served arm's stream is rewritten into a
// refusal before any content. It then passes through to the client as any
// re-served refusal does, and each attempt is settled on its own usage.
//
// The rewrite works on the upstream's bytes, before the cell reads them: the
// re-served call's text and tool calls are dropped and its stop is made a
// refusal (content_filter on Chat, refusal on Messages). Its reasoning and its
// usage are kept, so the refused attempt is billed what the provider charged.
// Like every fault it is inert unless the cell enables fault_injection.

// faultsFirstAttempt reports a fault that refuses a held turn's first
// attempt at its first visible content.
func faultsFirstAttempt(ctx *RequestContext) bool {
	return ctx.InjectedFault == headers.FaultRefusalPrelude || ctx.InjectedFault == headers.FaultRefusalPreludeAll
}

// reserveFaultActive reports a re-served call whose bytes the fault rewrites.
func reserveFaultActive(ctx *RequestContext) bool {
	return ctx != nil && ctx.InjectedFault == headers.FaultRefusalPreludeAll && refusalReserveServing(ctx) &&
		(ctx.TargetFormat == llmprotocol.OpenAIChatV1 || ctx.TargetFormat == llmprotocol.AnthropicMessagesV1)
}

// reserveFaultStream is the rewrite's state across the call's chunks.
type reserveFaultStream struct {
	// pending is an event not yet terminated by a blank line.
	pending []byte
	// started says the stream's first bytes were read, so a leading byte
	// order mark has been stripped.
	started bool
	// dropped are the Messages content blocks dropped, by upstream index;
	// kept maps each block kept to its index in the rewritten stream, which
	// the decoder reads as contiguous from zero.
	dropped map[float64]bool
	kept    map[float64]int
	// droppedContent says the last rewrite dropped text or a tool call.
	droppedContent bool
}

func (plan *refusalReservePlan) faultStream() *reserveFaultStream {
	if plan.fault == nil {
		plan.fault = &reserveFaultStream{dropped: map[float64]bool{}, kept: map[float64]int{}}
	}
	return plan.fault
}

// utf8BOM is the byte order mark an SSE reader strips from a stream's start.
var utf8BOM = []byte("\xef\xbb\xbf")

// faultReservedChunk is the re-served call's chunk as the fault rewrites it.
// An event split across chunks is rewritten once it is whole. Content it
// drops still counts as the call's activity, so the silence clock does not
// cut a call that is answering.
func faultReservedChunk(ctx *RequestContext, body []byte) []byte {
	fault := ctx.RefusalReserve.faultStream()
	data := append(fault.pending, body...)
	if !fault.started {
		if len(data) < len(utf8BOM) && bytes.HasPrefix(utf8BOM, data) {
			fault.pending = append([]byte(nil), data...)
			return nil
		}
		fault.started = true
		data = bytes.TrimPrefix(data, utf8BOM)
	}
	// A CR that ends the chunk may be the first half of a CRLF: it waits.
	var heldCR bool
	if n := len(data); n > 0 && data[n-1] == '\r' {
		data, heldCR = data[:n-1], true
	}
	data = bytes.ReplaceAll(bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n")), []byte("\r"), []byte("\n"))
	fault.droppedContent = false
	var out bytes.Buffer
	for {
		end := bytes.Index(data, []byte("\n\n"))
		if end < 0 {
			break
		}
		event := data[:end+2]
		data = data[end+2:]
		out.Write(fault.rewrite(event, ctx.TargetFormat))
	}
	fault.pending = append([]byte(nil), data...)
	if heldCR {
		fault.pending = append(fault.pending, '\r')
	}
	if fault.droppedContent {
		ctx.StreamContentSince = time.Now()
	}
	return out.Bytes()
}

// flushReservedFault is the call's last event when the stream ended without
// the blank line that terminates it: the codec reads such an event, so the
// rewrite does too.
func flushReservedFault(ctx *RequestContext) []byte {
	fault := ctx.RefusalReserve.faultStream()
	if len(bytes.TrimSpace(fault.pending)) == 0 {
		fault.pending = nil
		return nil
	}
	tail := append(bytes.TrimRight(fault.pending, "\r\n"), '\n', '\n')
	fault.pending = nil
	return faultReservedChunk(ctx, tail)
}

// rewrite is one whole SSE event as the fault rewrites it; nil drops it. Its
// data lines are read joined, as an SSE reader joins them, and written back
// as one.
func (fault *reserveFaultStream) rewrite(event []byte, format llmprotocol.WireFormat) []byte {
	lines := bytes.Split(bytes.TrimRight(event, "\n"), []byte("\n"))
	var fields [][]byte
	var others [][]byte
	dataAt := -1
	for _, line := range lines {
		if !bytes.HasPrefix(line, []byte("data:")) {
			others = append(others, line)
			continue
		}
		if dataAt < 0 {
			dataAt = len(others)
		}
		field := line[len("data:"):]
		if len(field) > 0 && field[0] == ' ' {
			field = field[1:]
		}
		fields = append(fields, field)
	}
	if dataAt < 0 {
		return event
	}
	decoder := json.NewDecoder(bytes.NewReader(bytes.Join(fields, []byte("\n"))))
	decoder.UseNumber()
	var frame map[string]interface{}
	if decoder.Decode(&frame) != nil {
		// [DONE], or a frame the cell reads as it stands.
		return event
	}
	keep := true
	if format == llmprotocol.AnthropicMessagesV1 {
		keep = fault.rewriteMessages(frame)
	} else {
		fault.droppedContent = rewriteChat(frame) || fault.droppedContent
	}
	if !keep {
		return nil
	}
	rewritten, err := json.Marshal(frame)
	if err != nil {
		return event
	}
	out := append([][]byte{}, others[:dataAt]...)
	out = append(out, append([]byte("data: "), rewritten...))
	out = append(out, others[dataAt:]...)
	return append(bytes.Join(out, []byte("\n")), '\n', '\n')
}

// rewriteChat drops a Chat chunk's text and tool calls and makes its stop a
// content_filter refusal. It reports whether it dropped content.
func rewriteChat(frame map[string]interface{}) bool {
	dropped := false
	choices, _ := frame["choices"].([]interface{})
	for _, raw := range choices {
		choice, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		if delta, ok := choice["delta"].(map[string]interface{}); ok {
			for _, key := range []string{"content", "tool_calls", "function_call"} {
				if value, present := delta[key]; present {
					if text, isText := value.(string); !isText || text != "" {
						dropped = true
					}
					delete(delta, key)
				}
			}
		}
		if reason, ok := choice["finish_reason"].(string); ok && reason != "" {
			choice["finish_reason"] = string(llmprotocol.StopContentFilter)
		}
	}
	return dropped
}

// rewriteMessages drops a Messages stream's text and tool blocks, keeping its
// thinking renumbered from zero, and makes its stop a refusal. It reports
// whether the event stays.
func (fault *reserveFaultStream) rewriteMessages(frame map[string]interface{}) bool {
	index, _ := frame["index"].(json.Number)
	at, _ := index.Float64()
	switch frame["type"] {
	case "content_block_start":
		block, _ := frame["content_block"].(map[string]interface{})
		switch block["type"] {
		case "thinking", "redacted_thinking":
			fault.kept[at] = len(fault.kept)
			frame["index"] = fault.kept[at]
			return true
		}
		fault.dropped[at] = true
		fault.droppedContent = true
		return false
	case "content_block_delta", "content_block_stop":
		if fault.dropped[at] {
			fault.droppedContent = true
			return false
		}
		if renumbered, ok := fault.kept[at]; ok {
			frame["index"] = renumbered
		}
	case "message_delta":
		if delta, ok := frame["delta"].(map[string]interface{}); ok {
			if reason, ok := delta["stop_reason"].(string); ok && reason != "" {
				delta["stop_reason"] = "refusal"
			}
		}
	}
	return true
}
