package protocolcodec

import (
	"encoding/json"
	"strconv"
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

const (
	orderSummary   = `{"type":"thinking_delta","thinking":"**Interpreting the request**"}`
	orderSignature = `{"type":"signature_delta","signature":"provider-signature"}`
	orderPing      = "event: ping\ndata: {\"type\":\"ping\"}\n\n"
	orderToolA     = `{"type":"tool_use","id":"toolu_A","name":"Read","input":{}}`
	orderToolB     = `{"type":"tool_use","id":"toolu_B","name":"Read","input":{}}`
)

func orderJSON(fragment string) string {
	encoded, _ := json.Marshal(fragment)
	return `{"type":"input_json_delta","partial_json":` + string(encoded) + `}`
}

func orderEmptyThinkingAt(index string) string {
	return sse("content_block_start", strings.Replace(orderEmptyThinking, `"index":0`, `"index":`+index, 1))
}

// Streams whose blocks overlap come out one block at a time, each block's
// frames together, in start order.
func TestAnthropicBlockOrderSerializesOverlappingBlocks(t *testing.T) {
	for name, tc := range map[string]struct{ input, want string }{
		// OpenRouter's luna stream as the dev cell served it on 2026-10-22
		// (trailing3-f545174cfef7): an empty unsigned thinking block opened
		// first and stopped last, around the redacted reasoning and the
		// answer. The answer is now the last block.
		"luna_empty_thinking_around_answer": {
			input: orderOpen() + orderEmptyThinkingAt("0") +
				orderStart("1", orderRedacted) + orderStop("1") +
				orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
				orderStop("0") + orderClose(),
			want: orderOpen() + orderEmptyThinkingAt("0") + orderStop("0") +
				orderStart("1", orderRedacted) + orderStop("1") +
				orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
				orderClose(),
		},
		// A thinking block with summary text, left open while the redacted
		// reasoning streamed inside it (trailing-0b3a3d71e393).
		"luna_redacted_nested_in_summary": {
			input: orderOpen() + orderEmptyThinkingAt("0") + orderDelta("0", orderSummary) +
				orderStart("1", orderRedacted) + orderStop("1") + orderStop("0") +
				orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
				orderClose(),
			want: orderOpen() + orderEmptyThinkingAt("0") + orderDelta("0", orderSummary) + orderStop("0") +
				orderStart("1", orderRedacted) + orderStop("1") +
				orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
				orderClose(),
		},
		// A signature that arrives after the answer stays in the thinking
		// block it signs, and the thinking block goes before the answer.
		"late_signature": {
			input: orderOpen() + orderEmptyThinkingAt("0") + orderDelta("0", orderSummary) +
				orderStart("1", orderText) + orderDelta("1", orderOK) + orderStop("1") +
				orderDelta("0", orderSignature) + orderStop("0") + orderClose(),
			want: orderOpen() + orderEmptyThinkingAt("0") + orderDelta("0", orderSummary) +
				orderDelta("0", orderSignature) + orderStop("0") +
				orderStart("1", orderText) + orderDelta("1", orderOK) + orderStop("1") +
				orderClose(),
		},
		// Interleaved tool calls: neither is stopped early, so each keeps
		// its id and all of its arguments.
		"interleaved_tool_use": {
			input: orderOpen() +
				orderStart("0", orderToolA) + orderDelta("0", orderJSON(`{"path":`)) +
				orderStart("1", orderToolB) + orderDelta("1", orderJSON(`{"path":"/b"}`)) + orderStop("1") +
				orderDelta("0", orderJSON(`"/a"}`)) + orderStop("0") + orderClose(),
			want: orderOpen() +
				orderStart("0", orderToolA) + orderDelta("0", orderJSON(`{"path":`)) +
				orderDelta("0", orderJSON(`"/a"}`)) + orderStop("0") +
				orderStart("1", orderToolB) + orderDelta("1", orderJSON(`{"path":"/b"}`)) + orderStop("1") +
				orderClose(),
		},
		// A ping among held frames keeps its place among them.
		"ping_while_held": {
			input: orderOpen() + orderEmptyThinkingAt("0") +
				orderStart("1", orderRedacted) + orderPing + orderStop("1") +
				orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
				orderStop("0") + orderClose(),
			want: orderOpen() + orderEmptyThinkingAt("0") + orderStop("0") +
				orderStart("1", orderRedacted) + orderStop("1") + orderPing +
				orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
				orderClose(),
		},
		// A message_delta before the live block's stop waits for it.
		"message_delta_before_stop": {
			input: orderOpen() + orderEmptyThinkingAt("0") +
				orderStart("1", orderText) + orderDelta("1", orderOK) + orderStop("1") +
				orderClose() + orderStop("0"),
			want: orderOpen() + orderEmptyThinkingAt("0") + orderStop("0") +
				orderStart("1", orderText) + orderDelta("1", orderOK) + orderStop("1") +
				orderClose(),
		},
		// Client indexes follow start order and stay contiguous.
		"renumbered_by_start_order": {
			input: orderOpen() + orderEmptyThinkingAt("0") +
				orderStart("3", orderText) + orderDelta("3", orderOK) +
				orderStart("1", orderRedacted) + orderStop("1") + orderStop("3") +
				orderStop("0") + orderClose(),
			want: orderOpen() + orderEmptyThinkingAt("0") + orderStop("0") +
				orderStart("1", orderText) + orderDelta("1", orderOK) + orderStop("1") +
				orderStart("2", orderRedacted) + orderStop("2") +
				orderClose(),
		},
		// The stream ends with the live block never stopped: what is held
		// still goes out, in start order, and nothing is stopped for it.
		"truncated": {
			input: orderOpen() + orderEmptyThinkingAt("0") +
				orderStart("1", orderText) + orderDelta("1", orderOK),
			want: orderOpen() + orderEmptyThinkingAt("0") +
				orderStart("1", orderText) + orderDelta("1", orderOK),
		},
	} {
		t.Run(name, func(t *testing.T) {
			if got := filterAnthropicOrder(t, tc.input); got != tc.want {
				t.Fatalf("stream =\n%s\nwant\n%s", got, tc.want)
			}
		})
	}
}

// An error event sends what is held first, then the error in its public form.
func TestAnthropicBlockOrderErrorReleasesHeldBlocks(t *testing.T) {
	failure := sse("error", `{"type":"error","error":{"type":"overloaded_error","message":"upstream detail"}}`)
	publicFailure := filterAnthropicOrder(t, failure)
	if !strings.HasPrefix(publicFailure, "event: error\n") {
		t.Fatalf("error frame = %q", publicFailure)
	}
	input := orderOpen() + orderEmptyThinkingAt("0") +
		orderStart("1", orderRedacted) + orderStop("1") +
		orderStart("2", orderText) + orderDelta("2", orderOK) + failure
	want := orderOpen() + orderEmptyThinkingAt("0") +
		orderStart("1", orderRedacted) + orderStop("1") +
		orderStart("2", orderText) + orderDelta("2", orderOK) + publicFailure
	if got := filterAnthropicOrder(t, input); got != want {
		t.Fatalf("stream =\n%s\nwant\n%s", got, want)
	}
}

// Streams already in Messages order travel byte for byte.
func TestAnthropicBlockOrderLeavesOrderedStreamsUnchanged(t *testing.T) {
	var long strings.Builder
	long.WriteString(orderOpen() + orderEmptyThinkingAt("0"))
	for range 2000 {
		long.WriteString(orderDelta("0", orderSummary))
	}
	long.WriteString(orderStop("0") + orderStart("1", orderText))
	for range 5000 {
		long.WriteString(orderDelta("1", orderOK))
	}
	long.WriteString(orderStop("1") + orderClose())
	for name, input := range map[string]string{
		// OpenRouter's luna stream when the summary has text
		// (trailing3-2d9aa023d8ed).
		"summary_text": orderOpen() + orderEmptyThinkingAt("0") +
			orderDelta("0", `{"type":"thinking_delta","thinking":"**Interpreting**"}`) +
			orderDelta("0", `{"type":"thinking_delta","thinking":" the request."}`) + orderStop("0") +
			orderStart("1", orderRedacted) + orderStop("1") +
			orderStart("2", orderText) + orderDelta("2", orderOK) + orderStop("2") +
			orderClose(),
		// Claude: thinking, its signature_delta, then text.
		"claude_thinking": orderOpen() + orderEmptyThinkingAt("0") +
			orderDelta("0", `{"type":"thinking_delta","thinking":"Plan."}`) +
			orderDelta("0", `{"type":"signature_delta","signature":"EqQBCkYIBxgCKkA"}`) + orderStop("0") +
			orderStart("1", `{"type":"text","text":""}`) + orderDelta("1", orderOK) + orderStop("1") +
			orderClose(),
		// Claude's omitted thinking: a signature and no text.
		"claude_signature_only": orderOpen() + orderEmptyThinkingAt("0") +
			orderDelta("0", `{"type":"signature_delta","signature":"EqQBCkYIBxgCKkA"}`) + orderStop("0") +
			orderStart("1", `{"type":"text","text":""}`) + orderDelta("1", orderOK) + orderStop("1") +
			orderClose(),
		"empty_thinking_then_text": orderOpen() + orderEmptyThinkingAt("0") + orderStop("0") +
			orderStart("1", orderText) + orderDelta("1", orderOK) + orderStop("1") +
			orderClose(),
		"ping_inside_thinking": orderOpen() + orderEmptyThinkingAt("0") + orderPing +
			orderDelta("0", orderSummary) + orderPing + orderDelta("0", orderSignature) + orderStop("0") +
			orderClose(),
		"tool_use_after_text": orderOpen() +
			orderStart("0", `{"type":"text","text":""}`) + orderDelta("0", `{"type":"text_delta","text":"Reading."}`) + orderStop("0") +
			orderStart("1", `{"type":"tool_use","id":"toolu_1","name":"Read","input":{}}`) +
			orderDelta("1", `{"type":"input_json_delta","partial_json":"{\"file_path\":\"/a\"}"}`) + orderStop("1") +
			orderPing + orderClose(),
		"parallel_tool_use": orderOpen() +
			orderStart("0", orderToolA) + orderDelta("0", orderJSON(`{"path":"/a"}`)) + orderStop("0") +
			orderStart("1", orderToolB) + orderDelta("1", orderJSON(`{"path":"/b"}`)) + orderStop("1") +
			orderClose(),
		"long": long.String(),
	} {
		t.Run(name, func(t *testing.T) {
			if got := filterAnthropicOrder(t, input); got != input {
				t.Fatalf("stream =\n%s\nwant\n%s", got, input)
			}
		})
	}
}

// A long answer streamed inside the open thinking block is held and then
// sent whole, in order.
func TestAnthropicBlockOrderLongHeldAnswer(t *testing.T) {
	var input, want, answer strings.Builder
	input.WriteString(orderOpen() + orderEmptyThinkingAt("0") + orderStart("1", orderText))
	want.WriteString(orderOpen() + orderEmptyThinkingAt("0") + orderStop("0") + orderStart("1", orderText))
	for position := range 5000 {
		delta := orderDelta("1", `{"type":"text_delta","text":"chunk `+strconv.Itoa(position)+` "}`)
		answer.WriteString(delta)
		input.WriteString(delta)
	}
	input.WriteString(orderStop("1") + orderStop("0") + orderClose())
	want.WriteString(answer.String() + orderStop("1") + orderClose())
	filter := NewAnthropicPublicStreamFilter(1<<20, "")
	var got []byte
	for _, frame := range strings.SplitAfter(input.String(), "\n\n") {
		out, err := filter.Push([]byte(frame))
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, out...)
	}
	out, err := filter.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	if got = append(got, out...); string(got) != want.String() {
		t.Fatalf("held answer came out changed (%d bytes, want %d)", len(got), want.Len())
	}
}

// What is held behind an open block is bounded.
func TestAnthropicBlockOrderBoundsHeldFrames(t *testing.T) {
	filter := NewAnthropicPublicStreamFilter(1<<20, "")
	if _, err := filter.Push([]byte(orderOpen() + orderEmptyThinkingAt("0") + orderStart("1", orderText))); err != nil {
		t.Fatal(err)
	}
	delta := []byte(orderDelta("1", `{"type":"text_delta","text":"`+strings.Repeat("x", 512<<10)+`"}`))
	for range anthropicBlockOrderBufferLimit/len(delta) + 1 {
		if _, err := filter.Push(delta); err != nil {
			if !strings.Contains(err.Error(), "too much") {
				t.Fatalf("error = %v", err)
			}
			return
		}
	}
	t.Fatal("held frames grew past the limit")
}
