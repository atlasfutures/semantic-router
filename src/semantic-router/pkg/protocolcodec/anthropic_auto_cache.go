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
// The block that already holds a breakpoint keeps it. A request whose
// client breakpoints already fill Anthropic's limit gets none added, and a
// conversation with no cacheable block gets none either; both drops are
// counted.
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
	messageIndex, contentIndex, found := lastAnthropicCacheableBlock(request.Messages)
	if !found {
		drop("no block of the conversation can carry a Messages cache breakpoint")
		return request, diagnostics
	}
	if request.Messages[messageIndex].Content[contentIndex].Cache != nil {
		return request, nil
	}
	if countCacheBreakpoints(request) >= anthropicBreakpointLimit {
		drop("the client's own breakpoints already use every Messages cache breakpoint")
		return request, diagnostics
	}
	messages := append([]llmprotocol.Message(nil), request.Messages...)
	contents := append([]llmprotocol.Content(nil), messages[messageIndex].Content...)
	directive := *request.AutoCache
	contents[contentIndex].Cache = &directive
	messages[messageIndex].Content = contents
	request.Messages = messages
	diagnostics = appendDiagnostics(diagnostics, llmprotocol.Diagnostics{{
		Source: request.Trusted.SourceFormat, Target: llmprotocol.AnthropicMessagesV1,
		Field: "cache_control", Action: llmprotocol.DiagnosticApproximated,
		Reason: "automatic caching becomes a breakpoint on the last cacheable block",
	}}, policy.Limits.Diagnostics)
	return request, diagnostics
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
