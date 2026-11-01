package extproc

import (
	"bytes"
	"encoding/json"

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
	// dropped are the Messages content blocks dropped, by index.
	dropped map[float64]bool
}

// faultReservedChunk is the re-served call's chunk as the fault rewrites it.
// An event split across chunks is rewritten once it is whole; one the stream
// never terminates is dropped, as an SSE reader drops it.
func faultReservedChunk(ctx *RequestContext, body []byte) []byte {
	plan := ctx.RefusalReserve
	if plan.fault == nil {
		plan.fault = &reserveFaultStream{dropped: map[float64]bool{}}
	}
	fault := plan.fault
	data := append(fault.pending, body...)
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
	return out.Bytes()
}

// rewrite is one whole SSE event as the fault rewrites it; nil drops it.
func (fault *reserveFaultStream) rewrite(event []byte, format llmprotocol.WireFormat) []byte {
	lines := bytes.Split(bytes.TrimRight(event, "\n"), []byte("\n"))
	dataAt := -1
	for i, line := range lines {
		if bytes.HasPrefix(line, []byte("data:")) {
			dataAt = i
		}
	}
	if dataAt < 0 {
		return event
	}
	payload := bytes.TrimSpace(lines[dataAt][len("data:"):])
	decoder := json.NewDecoder(bytes.NewReader(payload))
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
		rewriteChat(frame)
	}
	if !keep {
		return nil
	}
	rewritten, err := json.Marshal(frame)
	if err != nil {
		return event
	}
	lines[dataAt] = append([]byte("data: "), rewritten...)
	return append(bytes.Join(lines, []byte("\n")), '\n', '\n')
}

// rewriteChat drops a Chat chunk's text and tool calls and makes its stop a
// content_filter refusal.
func rewriteChat(frame map[string]interface{}) {
	choices, _ := frame["choices"].([]interface{})
	for _, raw := range choices {
		choice, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		if delta, ok := choice["delta"].(map[string]interface{}); ok {
			delete(delta, "content")
			delete(delta, "tool_calls")
			delete(delta, "function_call")
		}
		if reason, ok := choice["finish_reason"].(string); ok && reason != "" {
			choice["finish_reason"] = string(llmprotocol.StopContentFilter)
		}
	}
}

// rewriteMessages drops a Messages stream's text and tool blocks, keeping its
// thinking, and makes its stop a refusal. It reports whether the event stays.
func (fault *reserveFaultStream) rewriteMessages(frame map[string]interface{}) bool {
	index, _ := frame["index"].(json.Number)
	at, _ := index.Float64()
	switch frame["type"] {
	case "content_block_start":
		block, _ := frame["content_block"].(map[string]interface{})
		switch block["type"] {
		case "thinking", "redacted_thinking":
			return true
		}
		fault.dropped[at] = true
		return false
	case "content_block_delta", "content_block_stop":
		return !fault.dropped[at]
	case "message_delta":
		if delta, ok := frame["delta"].(map[string]interface{}); ok {
			if reason, ok := delta["stop_reason"].(string); ok && reason != "" {
				delta["stop_reason"] = "refusal"
			}
		}
	}
	return true
}
