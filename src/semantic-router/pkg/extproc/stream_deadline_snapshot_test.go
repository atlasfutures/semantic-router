package extproc

import (
	"bytes"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// A count the upstream stated before the output is a snapshot, and the row
// says so.
//
// Anthropic's message_start states output_tokens 1 before the first delta.
// Measured live 2026-10-26 with a 20 s deadline: after 781 content deltas the
// usage row said completion_tokens 1, usage_source authoritative, and so did
// the frame the Messages client received. The turn was billed as one token.
// Once output has run past the statement the output leg takes the larger of
// the snapshot and the Router's estimate of what it forwarded, and the row is
// marked an estimate so reconciliation can tell it from a settlement.
func TestACountStatedBeforeTheOutputIsRestatedOnTheRow(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1
	ctx.StartTime = time.Now()

	// Inside the budget: message_start with the snapshot, then real output.
	streamDeadlineResponseBody(t, router, ctx, anthropicMessageStart(37, 1))
	streamDeadlineResponseBody(t, router, ctx, anthropicTextDelta(strings.Repeat("one two three four ", 60)))

	// Then the deadline falls.
	ctx.StartTime = time.Now().Add(-600 * time.Second)
	body := streamDeadlineResponseBody(t, router, ctx, anthropicTextDelta("five"))

	fields := findLogEvent(t, logs, "llm_usage")
	if truncated, _ := fields["truncated"].(bool); !truncated {
		t.Fatalf("a turn the Router cut was counted as a whole one: %v", fields)
	}
	if got, _ := fields["completion_tokens"].(int64); got <= 1 {
		t.Fatalf("completion_tokens = %v: the snapshot was billed as the turn's output", fields["completion_tokens"])
	}
	if got, _ := fields["usage_source"].(string); got != "stream_estimate" {
		t.Fatalf("usage_source = %q, want stream_estimate for a restated count", got)
	}
	if got, _ := fields["prompt_tokens"].(int64); got != 37 {
		t.Fatalf("prompt_tokens = %v, want the stated 37: a prompt does not grow", fields["prompt_tokens"])
	}
	// The frame the client receives says the same.
	if bytes.Contains(body, []byte(`"output_tokens":1,`)) {
		t.Fatalf("the client was told the turn cost one token:\n%s", body)
	}
	if !bytes.Contains(body, []byte(`"usage_source":"stream_estimate"`)) {
		t.Fatalf("the client was not told the count is an estimate:\n%s", body)
	}
}

func anthropicMessageStart(input, output int) string {
	return "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",` +
		`"model":"claude-haiku-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":` +
		itoa(input) + `,"output_tokens":` + itoa(output) + `}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n"
}

func anthropicTextDelta(text string) string {
	return "event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + text + `"}}` + "\n\n"
}

func itoa(value int) string {
	return strconv.Itoa(value)
}

// A snapshot that happens to be larger than the estimate still does not pass
// as settled. The turn ran past it, so the row is an estimate whichever count
// it carries; a reconciler summing settlements must not count this one.
func TestASnapshotThatWinsIsStillAnEstimate(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1
	ctx.StartTime = time.Now()

	// A generous snapshot, then a little real output past it.
	streamDeadlineResponseBody(t, router, ctx, anthropicMessageStart(37, 100))
	streamDeadlineResponseBody(t, router, ctx, anthropicTextDelta("one two"))

	ctx.StartTime = time.Now().Add(-600 * time.Second)
	streamDeadlineResponseBody(t, router, ctx, anthropicTextDelta("three"))

	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["completion_tokens"].(int64); got != 100 {
		t.Fatalf("completion_tokens = %v, want the larger snapshot kept", fields["completion_tokens"])
	}
	if got, _ := fields["usage_source"].(string); got != "stream_estimate" {
		t.Fatalf("usage_source = %q: a count the turn ran past was presented as settled", got)
	}
}

// A cut before any output reached the client ends on the error terminal even
// when the upstream had already stated a count: Anthropic's message_start
// prices the prompt before the first delta, and that count used to make the
// cut look like a turn to close as a length stop.
func TestACutBeforeAnyOutputKeepsTheErrorTerminalDespiteAStatedCount(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1
	ctx.SemanticRequest = streamUsageRequest()

	// message_start with its count, then the deadline, with no delta between.
	body := streamDeadlineResponseBody(t, router, ctx, anthropicMessageStart(37, 1))

	if !bytes.Contains(body, []byte(`"code":"stream_truncated"`)) {
		t.Fatalf("a cut before any output did not end on the error terminal:\n%s", body)
	}
	if bytes.Contains(body, []byte(`"finish_reason":"length"`)) || bytes.Contains(body, []byte("data: [DONE]")) {
		t.Fatalf("a turn that delivered nothing was closed as a length stop:\n%s", body)
	}
	// The prompt the upstream priced still reaches the usage row.
	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["prompt_tokens"].(int64); got != 37 {
		t.Fatalf("prompt_tokens = %v, want the stated 37 on the row", fields["prompt_tokens"])
	}
}

// A call that has only started is output the client holds, with no bytes to
// count. A cut then still closes the turn in the format's own shape rather
// than on the error terminal reserved for a turn that delivered nothing.
func TestACallThatOnlyStartedIsOutputTheClientHolds(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.SemanticRequest = streamUsageRequest()

	body := streamDeadlineResponseBody(t, router, ctx,
		`data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,`+
			`"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function",`+
			`"function":{"name":"lookup","arguments":""}}]},"finish_reason":null}]}`+"\n\n")

	if !bytes.Contains(body, []byte(`"finish_reason":"length"`)) || !bytes.Contains(body, []byte("data: [DONE]")) {
		t.Fatalf("a turn whose call had started was not closed for the client:\n%s", body)
	}
	if bytes.Contains(body, []byte(`"code":"stream_truncated"`)) {
		t.Fatalf("a turn with output the client holds ended on the error terminal:\n%s", body)
	}
}

// The same call's start that is output for a Responses client is nothing for
// a Chat client, whose role chunk already went. A cut after it ends on the
// error terminal for Chat, as a turn that delivered nothing does.
func TestACallStartIsNotOutputForAChatClient(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1
	ctx.SemanticRequest = streamUsageRequest()

	body := streamDeadlineResponseBody(t, router, ctx, anthropicMessageStart(37, 1)+
		"event: content_block_start\n"+
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}`+"\n\n")

	if !bytes.Contains(body, []byte(`"code":"stream_truncated"`)) {
		t.Fatalf("a Chat client with no output was not told the turn was cut:\n%s", body)
	}
	if bytes.Contains(body, []byte(`"finish_reason":"length"`)) {
		t.Fatalf("a Chat client that held nothing was closed as a length stop:\n%s", body)
	}
}
