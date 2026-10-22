package protocolcodec

import (
	"strings"
	"testing"
)

func sse(event, data string) string {
	return "event: " + event + "\ndata: " + data + "\n\n"
}

const (
	orderMessageStart  = `{"type":"message_start","message":{"id":"msg_lp9etdqZjJXqfATH5AR8","type":"message","role":"assistant","container":null,"content":[],"model":"openai/gpt-6-luna","stop_reason":null,"stop_details":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0},"provider":"OpenAI"}}`
	orderMessageDelta  = `{"type":"message_delta","delta":{"container":null,"stop_details":null,"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":3,"output_tokens":48,"output_tokens_details":{"thinking_tokens":41}}}`
	orderEmptyThinking = `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`
)

func orderStart(index, block string) string {
	return sse("content_block_start", `{"type":"content_block_start","index":`+index+`,"content_block":`+block+`}`)
}

func orderDelta(index, delta string) string {
	return sse("content_block_delta", `{"type":"content_block_delta","index":`+index+`,"delta":`+delta+`}`)
}

func orderStop(index string) string {
	return sse("content_block_stop", `{"type":"content_block_stop","index":`+index+`}`)
}

func orderOpen() string {
	return sse("message_start", orderMessageStart)
}

func orderClose() string {
	return sse("message_delta", orderMessageDelta) + sse("message_stop", `{"type":"message_stop"}`)
}

const (
	orderRedacted = `{"type":"redacted_thinking","data":"gAAAAABq2pNtnfA9KXMr3PRAf9EddSBk92tK"}`
	orderText     = `{"type":"text","text":"","citations":[]}`
	orderOK       = `{"type":"text_delta","text":"ok"}`
)

func filterAnthropicOrder(t *testing.T, input string) string {
	t.Helper()
	return string(pushInSmallChunks(t, NewAnthropicPublicStreamFilter(1<<20, ""), []byte(input)))
}

// OpenRouter's Messages stream for openai/gpt-6-luna, as the dev cell served
// it on 2026-10-22 (trailing3-f545174cfef7): an empty unsigned thinking block
// opened first and stopped last, around the redacted reasoning and the answer.
func TestAnthropicBlockOrderDropsEmptyThinkingOpenAroundAnswer(t *testing.T) {
	input := orderOpen() +
		sse("content_block_start", orderEmptyThinking) +
		orderStart("1", orderRedacted) + orderStop("1") +
		orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
		orderStop("0") +
		orderClose()
	want := orderOpen() +
		orderStart("0", orderRedacted) + orderStop("0") +
		orderStart("1", orderText) + orderDelta("1", orderOK) + orderStop("1") +
		orderClose()
	if got := filterAnthropicOrder(t, input); got != want {
		t.Fatalf("stream =\n%s\nwant\n%s", got, want)
	}
}

// The same thinking block, when the message ends before anything else
// starts, is dropped too.
func TestAnthropicBlockOrderDropsEmptyThinkingAtMessageEnd(t *testing.T) {
	input := orderOpen() +
		orderStart("0", orderText) + orderDelta("0", orderOK) + orderStop("0") +
		sse("content_block_start", strings.Replace(orderEmptyThinking, `"index":0`, `"index":1`, 1)) +
		orderClose() + orderStop("1")
	want := orderOpen() +
		orderStart("0", orderText) + orderDelta("0", orderOK) + orderStop("0") +
		orderClose()
	if got := filterAnthropicOrder(t, input); got != want {
		t.Fatalf("stream =\n%s\nwant\n%s", got, want)
	}
}

// A thinking block with summary text, left open while the redacted reasoning
// streamed inside it (trailing-0b3a3d71e393), is stopped before the redacted
// block starts, and its late stop is swallowed.
func TestAnthropicBlockOrderStopsOpenBlockBeforeNested(t *testing.T) {
	summary := `{"type":"thinking_delta","thinking":"**Interpreting the request**"}`
	input := orderOpen() +
		sse("content_block_start", orderEmptyThinking) + orderDelta("0", summary) +
		orderStart("1", orderRedacted) + orderStop("1") +
		orderStop("0") +
		orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
		orderClose()
	want := orderOpen() +
		sse("content_block_start", orderEmptyThinking) + orderDelta("0", summary) + orderStop("0") +
		orderStart("1", orderRedacted) + orderStop("1") +
		orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
		orderClose()
	if got := filterAnthropicOrder(t, input); got != want {
		t.Fatalf("stream =\n%s\nwant\n%s", got, want)
	}
}

// A delta for a block already stopped continues as a new block of the same
// start at the next index.
func TestAnthropicBlockOrderReopensLateDeltaAsNewBlock(t *testing.T) {
	summary := `{"type":"thinking_delta","thinking":"first"}`
	late := `{"type":"thinking_delta","thinking":"late"}`
	input := orderOpen() +
		sse("content_block_start", orderEmptyThinking) + orderDelta("0", summary) +
		orderStart("1", orderText) + orderDelta("1", orderOK) +
		orderDelta("0", late) + orderStop("0") +
		orderStop("1") +
		orderClose()
	want := orderOpen() +
		sse("content_block_start", orderEmptyThinking) + orderDelta("0", summary) + orderStop("0") +
		orderStart("1", orderText) + orderDelta("1", orderOK) + orderStop("1") +
		sse("content_block_start", strings.Replace(orderEmptyThinking, `"index":0`, `"index":2`, 1)) +
		orderDelta("2", late) + orderStop("2") +
		orderClose()
	if got := filterAnthropicOrder(t, input); got != want {
		t.Fatalf("stream =\n%s\nwant\n%s", got, want)
	}
}

// Streams already in Messages order travel byte for byte.
func TestAnthropicBlockOrderLeavesOrderedStreamsUnchanged(t *testing.T) {
	for name, input := range map[string]string{
		// OpenRouter's luna stream when the summary has text
		// (trailing3-2d9aa023d8ed).
		"summary_text": orderOpen() +
			sse("content_block_start", orderEmptyThinking) +
			orderDelta("0", `{"type":"thinking_delta","thinking":"**Interpreting**"}`) +
			orderDelta("0", `{"type":"thinking_delta","thinking":" the request."}`) + orderStop("0") +
			orderStart("1", orderRedacted) + orderStop("1") +
			orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
			orderClose(),
		// Claude: thinking, its signature_delta, then text.
		"claude_thinking": orderOpen() +
			sse("content_block_start", orderEmptyThinking) +
			orderDelta("0", `{"type":"thinking_delta","thinking":"Plan."}`) +
			orderDelta("0", `{"type":"signature_delta","signature":"EqQBCkYIBxgCKkA"}`) + orderStop("0") +
			orderStart("1", `{"type":"text","text":""}`) + orderDelta("1", orderOK) + orderStop("1") +
			orderClose(),
		// Claude's omitted thinking: a signature and no text.
		"claude_signature_only": orderOpen() +
			sse("content_block_start", orderEmptyThinking) +
			orderDelta("0", `{"type":"signature_delta","signature":"EqQBCkYIBxgCKkA"}`) + orderStop("0") +
			orderStart("1", `{"type":"text","text":""}`) + orderDelta("1", orderOK) + orderStop("1") +
			orderClose(),
		"tool_use_after_text": orderOpen() +
			orderStart("0", `{"type":"text","text":""}`) + orderDelta("0", `{"type":"text_delta","text":"Reading."}`) + orderStop("0") +
			orderStart("1", `{"type":"tool_use","id":"toolu_1","name":"Read","input":{}}`) +
			orderDelta("1", `{"type":"input_json_delta","partial_json":"{\"file_path\":\"/a\"}"}`) + orderStop("1") +
			sse("ping", `{"type":"ping"}`) +
			orderClose(),
	} {
		t.Run(name, func(t *testing.T) {
			if got := filterAnthropicOrder(t, input); got != input {
				t.Fatalf("stream =\n%s\nwant\n%s", got, input)
			}
		})
	}
}
