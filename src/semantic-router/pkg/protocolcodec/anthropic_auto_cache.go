package protocolcodec

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// anthropicBreakpointLimit is the most cache_control breakpoints Anthropic
// accepts in one request, across tools, system and messages.
const anthropicBreakpointLimit = 4

// placeAutoCacheBreakpoint turns a request-level automatic-cache directive
// into the one breakpoint Anthropic's automatic caching would set: on the
// last block of the conversation that can carry cache_control. A translated
// request has no top-level member to send, and a Messages provider caches
// only at breakpoints, so without this the dispatched turn asks for no
// caching at all. A Messages client's own top-level member is re-emitted
// instead (encodeAnthropicBaseRequest), which keeps its bytes replayable.
//
// When no message block can hold a breakpoint -- a conversation of signed
// reasoning only, say -- it goes on the last system block, else the last
// tool (see autoCacheBreakpointPlace).
//
// The place that already holds a breakpoint keeps it. A request whose
// client breakpoints already fill Anthropic's limit gets none added, and a
// request with no place for one gets none either; both drops are counted.
func placeAutoCacheBreakpoint(
	request llmprotocol.Request,
	policy llmprotocol.Policy,
) (llmprotocol.Request, llmprotocol.Diagnostics) {
	if request.AutoCache == nil || request.Trusted.SourceFormat == llmprotocol.AnthropicMessagesV1 {
		return request, nil
	}
	var diagnostics llmprotocol.Diagnostics
	drop := func(reason string) {
		appendPresentationDrop(&diagnostics, policy, request.Trusted.SourceFormat,
			llmprotocol.AnthropicMessagesV1, "cache_control", reason)
	}
	held, place, found := autoCacheBreakpointPlace(request)
	if !found {
		drop("no message block, system block or tool can carry a Messages cache breakpoint")
		return request, diagnostics
	}
	if held {
		return request, nil
	}
	if countCacheBreakpoints(request) >= anthropicBreakpointLimit {
		drop("the client's own breakpoints already use every Messages cache breakpoint")
		return request, diagnostics
	}
	directive := *request.AutoCache
	place(&request, &directive)
	diagnostics = appendDiagnostics(diagnostics, llmprotocol.Diagnostics{{
		Source: request.Trusted.SourceFormat, Target: llmprotocol.AnthropicMessagesV1,
		Field: "cache_control", Action: llmprotocol.DiagnosticApproximated,
		Reason: "automatic caching becomes a breakpoint on the last cacheable block",
	}}, policy.Limits.Diagnostics)
	return request, diagnostics
}

// autoCacheBreakpointPlace finds where the breakpoint goes: the last
// cacheable message block, else the last cacheable system block, else the
// last tool. Anthropic caches the prefix in the order tools, system,
// messages, so each rung is the longest prefix left when the one before it
// has no place: a breakpoint on the last system block caches the tools and
// the system prompt. It reports whether that place already holds a
// breakpoint, and returns a func that sets one there on copies of the
// slices it changes, so the caller's request is never mutated.
func autoCacheBreakpointPlace(
	request llmprotocol.Request,
) (bool, func(*llmprotocol.Request, *llmprotocol.CacheDirective), bool) {
	if messageIndex, contentIndex, found := lastAnthropicCacheableBlock(request.Messages); found {
		held := request.Messages[messageIndex].Content[contentIndex].Cache != nil
		return held, func(request *llmprotocol.Request, directive *llmprotocol.CacheDirective) {
			messages := append([]llmprotocol.Message(nil), request.Messages...)
			contents := append([]llmprotocol.Content(nil), messages[messageIndex].Content...)
			contents[contentIndex].Cache = directive
			messages[messageIndex].Content = contents
			request.Messages = messages
		}, true
	}
	for instructionIndex := len(request.Instructions) - 1; instructionIndex >= 0; instructionIndex-- {
		contents := request.Instructions[instructionIndex].Content
		for contentIndex := len(contents) - 1; contentIndex >= 0; contentIndex-- {
			if !anthropicCacheableBlock(contents[contentIndex]) {
				continue
			}
			held := contents[contentIndex].Cache != nil
			return held, func(request *llmprotocol.Request, directive *llmprotocol.CacheDirective) {
				instructions := append([]llmprotocol.InstructionBlock(nil), request.Instructions...)
				blocks := append([]llmprotocol.Content(nil), instructions[instructionIndex].Content...)
				blocks[contentIndex].Cache = directive
				instructions[instructionIndex].Content = blocks
				request.Instructions = instructions
			}, true
		}
	}
	if last := len(request.Tools) - 1; last >= 0 {
		return request.Tools[last].Cache != nil, func(request *llmprotocol.Request, directive *llmprotocol.CacheDirective) {
			tools := append([]llmprotocol.Tool(nil), request.Tools...)
			tools[last].Cache = directive
			request.Tools = tools
		}, true
	}
	return false, nil, false
}

// lastAnthropicCacheableBlock finds the last block the Messages encoder will
// send with a cache_control member. Thinking, refusal and carried blocks
// cannot hold one, and Anthropic refuses one on empty text.
func lastAnthropicCacheableBlock(messages []llmprotocol.Message) (int, int, bool) {
	for messageIndex := len(messages) - 1; messageIndex >= 0; messageIndex-- {
		contents := messages[messageIndex].Content
		if messageDropsWhole(contents, llmprotocol.AnthropicMessagesV1) {
			continue
		}
		for contentIndex := len(contents) - 1; contentIndex >= 0; contentIndex-- {
			if anthropicCacheableBlock(contents[contentIndex]) {
				return messageIndex, contentIndex, true
			}
		}
	}
	return 0, 0, false
}

func anthropicCacheableBlock(content llmprotocol.Content) bool {
	switch content.Kind {
	case llmprotocol.ContentText:
		return content.Text != ""
	case llmprotocol.ContentImage, llmprotocol.ContentFile:
		return true
	case llmprotocol.ContentToolCall:
		return content.ToolCall != nil
	case llmprotocol.ContentToolResult:
		return content.ToolResult != nil
	default:
		return false
	}
}

// CountCacheBreakpoints is countCacheBreakpoints for a caller that decides,
// before encoding, whether the client stated its own cache intent.
func CountCacheBreakpoints(request llmprotocol.Request) int {
	return countCacheBreakpoints(request)
}

// countCacheBreakpoints counts the breakpoints the request already states:
// on tools, on system blocks, and on message blocks, including the blocks
// inside a tool result.
func countCacheBreakpoints(request llmprotocol.Request) int {
	count := 0
	for _, tool := range request.Tools {
		if tool.Cache != nil {
			count++
		}
	}
	for _, instruction := range request.Instructions {
		count += countContentBreakpoints(instruction.Content)
	}
	for _, message := range request.Messages {
		count += countContentBreakpoints(message.Content)
	}
	return count
}

func countContentBreakpoints(contents []llmprotocol.Content) int {
	count := 0
	for _, content := range contents {
		if content.Cache != nil {
			count++
		}
		if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil {
			count += countContentBreakpoints(content.ToolResult.Content)
		}
	}
	return count
}
