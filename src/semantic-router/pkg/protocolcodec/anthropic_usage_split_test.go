package protocolcodec

import (
	"context"
	"strconv"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An Anthropic-shaped provider can state a breakdown that cannot be true, the
// same way an OpenAI-shaped one can: qwen reported reasoning_tokens 79 against
// completion_tokens 64 on 2026-09-03. The Chat decoder answers that by keeping
// the totals and leaving the split unknown.
//
// The Anthropic decoder answers it by clamping the remainder to zero and
// calling the result authoritative. That is worse than the wrong number it
// looks like: reasoning 79 and other 0 do not sum to the output total 64, so
// the response validator rejects the whole response and the client loses its
// answer. It is the defect the Chat decoder was already fixed for.
const (
	anthropicOverThinkingOutput   = 64
	anthropicOverThinkingThinking = 79
)

func anthropicUsageBody(outputTokens, thinkingTokens int64) []byte {
	return []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"provider-model",` +
		`"content":[{"type":"text","text":"hi"}],"stop_reason":"max_tokens","stop_sequence":null,` +
		`"usage":{"input_tokens":20,"output_tokens":` + strconv.FormatInt(outputTokens, 10) +
		`,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,` +
		`"output_tokens_details":{"thinking_tokens":` + strconv.FormatInt(thinkingTokens, 10) + `}}}`)
}

// A thinking count larger than the output count leaves the split unknown. It
// must not become a zero, which would read as "no ordinary output tokens".
func TestAnthropicUnreconcilableThinkingLeavesTheSplitUnknown(t *testing.T) {
	response, _, _, err := NewBuiltinEngine().DecodeResponse(
		llmprotocol.AnthropicMessagesV1,
		anthropicUsageBody(anthropicOverThinkingOutput, anthropicOverThinkingThinking),
	)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	assertTokenValue(t, "input total", response.Usage.InputTotal, 20)
	assertTokenValue(t, "output total", response.Usage.OutputTotal, anthropicOverThinkingOutput)
	assertTokenUnknown(t, "output reasoning", response.Usage.OutputReasoning)
	assertTokenUnknown(t, "output other", response.Usage.OutputOther)
}

// A breakdown that does reconcile is still split exactly as before.
func TestAnthropicReconcilableThinkingKeepsItsSplit(t *testing.T) {
	response, _, _, err := NewBuiltinEngine().DecodeResponse(
		llmprotocol.AnthropicMessagesV1, anthropicUsageBody(64, 40),
	)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	assertTokenValue(t, "output reasoning", response.Usage.OutputReasoning, 40)
	assertTokenValue(t, "output other", response.Usage.OutputOther, 24)
	assertTokenValue(t, "output total", response.Usage.OutputTotal, 64)
}

// The stream decoder reads the same breakdown off message_delta and has to
// answer it the same way.
func TestAnthropicStreamUnreconcilableThinkingLeavesTheSplitUnknown(t *testing.T) {
	decoder := AnthropicMessagesCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
		llmprotocol.DefaultPolicy(),
	)
	payload := []byte(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":20,\"output_tokens\":0,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":20,\"output_tokens\":64,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens_details\":{\"thinking_tokens\":79}}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
	events, _, err := decoder.Push(payload)
	if err != nil {
		t.Fatalf("push: %v", err)
	}
	assertUnreconciledThinkingUsage(t, events)
}

func assertUnreconciledThinkingUsage(t *testing.T, events []llmprotocol.Event) {
	t.Helper()
	for index := len(events) - 1; index >= 0; index-- {
		usage := events[index].Usage
		if usage == nil || usage.OutputTotal.Value == nil {
			continue
		}
		assertTokenValue(t, "output total", usage.OutputTotal, anthropicOverThinkingOutput)
		assertTokenUnknown(t, "output reasoning", usage.OutputReasoning)
		assertTokenUnknown(t, "output other", usage.OutputOther)
		return
	}
	t.Fatalf("no event carried an output total: %+v", events)
}
