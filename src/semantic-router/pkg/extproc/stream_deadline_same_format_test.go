package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"go.uber.org/zap/zaptest/observer"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// On a same-format route the upstream's own frames reach the client through
// the public filter, and the encoder defers a tool block's frames until the
// block closes. So the encoder's answer to "has the client seen output" was no
// even once a tool call had been forwarded, and the cut billed the turn as
// nothing -- while the upstream charged for the generation either way.
//
// What the client receives is left to the writer that was already writing it.
// This is the count: a turn whose output the filter forwarded is a turn with a
// count, and the Messages terminal is where that count travels.
func TestSameFormatMessagesCutBillsOutputTheFilterForwarded(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1

	body := streamDeadlineResponseBody(t, router, ctx, sameFormatToolCallChunk())

	if !bytes.Contains(body, []byte(`"usage_source":"stream_estimate"`)) {
		t.Fatalf("a turn whose output reached the client was billed as nothing:\n%s", body)
	}
	if !bytes.Contains(body, []byte(`"input_tokens":37`)) {
		t.Fatalf("the prompt the stream priced was not billed:\n%s", body)
	}
	// Messages keeps its published terminal, which is where its count travels.
	if !bytes.Contains(body, []byte(`"type":"error"`)) {
		t.Fatalf("the Messages contract stopped naming the failure:\n%s", body)
	}
	// The Router states its own terminal once; nothing of the upstream's is
	// restated beside it.
	if opens := bytes.Count(body, []byte(`"type":"message_start"`)); opens > 1 {
		t.Fatalf("the message opened %d times:\n%s", opens, body)
	}
}

// The same route, asked the same question through the state the usage row is
// built from. A turn the client has output for is a turn with a count, whoever
// wrote the bytes.
func TestSameFormatStreamRecordsOutputTheFilterForwarded(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIChatV1,
	} {
		router := &OpenAIRouter{}
		ctx := overrunningStreamContext(t)
		ctx.SourceFormat = format
		ctx.TargetFormat = format
		ctx.SemanticRequest = streamUsageRequest()
		router.initializeSemanticResponseStream(ctx)
		if ctx.PublicStreamFilter == nil {
			t.Fatalf("%s: this route does not forward the upstream's own frames", format)
		}

		chunk := sameFormatToolCallChunk()
		if format == llmprotocol.OpenAIChatV1 {
			chunk = endlessUpstreamChunk(1)
		}
		buffers := semanticStreamBuffers{}
		buffers.push([]byte(chunk), ctx)

		if !ctx.SemanticStreamState.outputForwarded {
			t.Fatalf("%s: output the filter forwarded was not recorded as forwarded", format)
		}
	}
}

// A same-format turn cut before any output still has nothing to bill. The
// fix reads the decoder's answer rather than the encoder's, and a prompt the
// upstream priced is not output.
func TestSameFormatCutBeforeOutputIsNotBilled(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1
	router.initializeSemanticResponseStream(ctx)

	buffers := semanticStreamBuffers{}
	buffers.push([]byte(sameFormatMessageStart()), ctx)

	if ctx.SemanticStreamState.outputForwarded {
		t.Fatal("a priced prompt was mistaken for output the client holds")
	}
}

// sameFormatMessageStart opens a message and prices its prompt, which is not
// output.
func sameFormatMessageStart() string {
	return "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message",` +
		`"role":"assistant","model":"claude-opus-4","content":[],"stop_reason":null,` +
		`"usage":{"input_tokens":37,"output_tokens":1}}}` + "\n\n"
}

// sameFormatToolCallChunk opens a tool call and streams arguments into it:
// output the client receives that carries no text to count.
func sameFormatToolCallChunk() string {
	return sameFormatMessageStart() +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":` +
		`{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":` +
		`{"type":"input_json_delta","partial_json":"{\"q\":\"par"}}` + "\n\n"
}

// A block the contract cannot name is generation too. An Anthropic redacted
// thinking block is a payload and nothing else -- no text, no reasoning, no
// arguments -- so the host's byte count saw a turn that had generated nothing,
// and the provisional count message_start states before the first delta stood
// as the settlement. The client received the block and was billed one token.
//
// Two checkpoints had to agree for that: the codec's own snapshot and this
// one. Fixing only the codec left the row and the client's terminal both
// stating the snapshot.
func TestABlockWithOnlyAPayloadIsCounted(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1

	body := streamDeadlineResponseBody(t, router, ctx, sameFormatMessageStart()+
		"event: content_block_start\n"+
		`data: {"type":"content_block_start","index":0,"content_block":`+
		`{"type":"redacted_thinking","data":"EroCCoYBGAIiQL2/hJREDACTEDPAYLOADHERE"}}`+"\n\n"+
		"event: content_block_stop\n"+`data: {"type":"content_block_stop","index":0}`+"\n\n")

	// The client holds the block, so the turn is billed past the snapshot.
	if !bytes.Contains(body, []byte("REDACTEDPAYLOADHERE")) {
		t.Fatalf("the block never reached the client, so this case is not being tested:\n%s", body)
	}
	// The terminal is where the cut's count travels; message_start's own
	// provisional count is not the settlement.
	_, terminal, found := bytes.Cut(body, []byte("event: error"))
	if !found {
		t.Fatalf("the turn did not reach a terminal:\n%s", body)
	}
	if bytes.Contains(terminal, []byte(`"output_tokens":1,`)) {
		t.Fatalf("a turn that delivered a payload was billed as one token:\n%s", terminal)
	}
	// And the row agrees with what the client was told.
	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["completion_tokens"].(int64); got <= 1 {
		t.Fatalf("the usage row billed %v for a turn that delivered a payload", fields["completion_tokens"])
	}
}

// A thinking block whose text the provider omits still signs itself. The
// signature is the block's whole substance -- the client has to send it back
// on the next turn -- and it is output the client holds, so the turn is
// billed. But how much was decided by a byte count that never saw the
// signature: it stayed at zero, the snapshot epoch never advanced, and the
// provisional count message_start states before the first delta stood as the
// settlement on both the client's terminal and the usage row. Whether to bill
// and how much have to come to the same answer.
func TestASignatureOnlyThinkingBlockIsCounted(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1

	body := streamDeadlineResponseBody(t, router, ctx, sameFormatMessageStart()+
		"event: content_block_start\n"+
		`data: {"type":"content_block_start","index":0,"content_block":`+
		`{"type":"thinking","thinking":"","signature":""}}`+"\n\n"+
		"event: content_block_delta\n"+
		`data: {"type":"content_block_delta","index":0,"delta":`+
		`{"type":"signature_delta","signature":"ErUBCkYIBRgCIkDaSIGNATUREONLYPAYLOADWITHNOTEXT"}}`+"\n\n"+
		"event: content_block_stop\n"+`data: {"type":"content_block_stop","index":0}`+"\n\n")

	// The client holds the signature, so the turn is billed past the snapshot.
	if !bytes.Contains(body, []byte("SIGNATUREONLYPAYLOADWITHNOTEXT")) {
		t.Fatalf("the signature never reached the client, so this case is not being tested:\n%s", body)
	}
	_, terminal, found := bytes.Cut(body, []byte("event: error"))
	if !found {
		t.Fatalf("the turn did not reach a terminal:\n%s", body)
	}
	if bytes.Contains(terminal, []byte(`"output_tokens":1,`)) {
		t.Fatalf("a turn that delivered a signature was billed as one token:\n%s", terminal)
	}
	// And the row agrees with what the client was told.
	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["completion_tokens"].(int64); got <= 1 {
		t.Fatalf("the usage row billed %v for a turn that delivered a signature", fields["completion_tokens"])
	}
}

// The same signature, sent in two deltas. The codec's encoders join a split
// signature in order; the host kept the last fragment, so a long signature
// whose final fragment was two bytes was stored as those two bytes and
// counted as them -- one token, which is the snapshot's own number, so the
// snapshot stood.
func TestASplitSignatureIsCountedWhole(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1

	body := streamDeadlineResponseBody(t, router, ctx, sameFormatMessageStart()+
		"event: content_block_start\n"+
		`data: {"type":"content_block_start","index":0,"content_block":`+
		`{"type":"thinking","thinking":"","signature":""}}`+"\n\n"+
		"event: content_block_delta\n"+
		`data: {"type":"content_block_delta","index":0,"delta":`+
		`{"type":"signature_delta","signature":"ErUBCkYIBRgCIkDaFIRSTFRAGMENTOFTHESIGNATURE"}}`+"\n\n"+
		"event: content_block_delta\n"+
		`data: {"type":"content_block_delta","index":0,"delta":`+
		`{"type":"signature_delta","signature":"Zz"}}`+"\n\n"+
		"event: content_block_stop\n"+`data: {"type":"content_block_stop","index":0}`+"\n\n")

	if !bytes.Contains(body, []byte("FIRSTFRAGMENTOFTHESIGNATURE")) {
		t.Fatalf("the first fragment never reached the client, so this case is not being tested:\n%s", body)
	}
	if got, want := ctx.SemanticStreamState.items[0].reasoningSig, "ErUBCkYIBRgCIkDaFIRSTFRAGMENTOFTHESIGNATUREZz"; got != want {
		t.Fatalf("the signature was stored as %q, not joined as %q", got, want)
	}
	_, terminal, found := bytes.Cut(body, []byte("event: error"))
	if !found {
		t.Fatalf("the turn did not reach a terminal:\n%s", body)
	}
	if bytes.Contains(terminal, []byte(`"output_tokens":1,`)) {
		t.Fatalf("a turn that delivered a split signature was billed as one token:\n%s", terminal)
	}
	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["completion_tokens"].(int64); got <= 1 {
		t.Fatalf("the usage row billed %v for a turn that delivered a split signature", fields["completion_tokens"])
	}
}

// A same-format Messages stream whose blocks overlap reaches the client one
// block at a time: the filter holds a block that starts while another is
// still open, until that one stops. A cut used to switch the client to the
// encoder's frames for its last chunk, so a block the filter still held was
// never sent -- and was billed, since the decoder had it. The cut now stays on
// the filter's frames and sends what it held before the terminal, so the
// client holds everything it is billed for.
//
// Two cuts are checked: one on a chunk that releases nothing, and one on the
// chunk that stops the open block and so releases the held ones itself.
func TestACutSendsWhatTheFilterHeld(t *testing.T) {
	for name, cutChunk := range map[string]string{
		"while held": "event: ping\ndata: {\"type\":\"ping\"}\n\n",
		"on release": "event: content_block_stop\n" + `data: {"type":"content_block_stop","index":0}` + "\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			router := &OpenAIRouter{}
			ctx := sameFormatMessagesContext(t)

			client := streamDeadlineResponseBody(t, router, ctx, sameFormatMessageStart()+
				overlappingThinkingChunk(strings.Repeat("thinking ", 40)))
			ctx.StartTime = time.Now().Add(-600 * time.Second)
			client = append(client, streamDeadlineResponseBody(t, router, ctx, cutChunk)...)

			delivered := wellFormedMessagesOutput(t, client)
			if delivered.redacted != 33 || delivered.text != 30 || delivered.thinking != 360 {
				t.Fatalf("the client did not receive every block the turn is billed for: %+v\n%s", delivered, client)
			}
			// 360 bytes of thinking, 30 of text, and the redacted block by its
			// raw form (71 bytes): 461 bytes, 116 tokens. The count is the
			// one the turn was billed before; the client now holds it.
			assertCutBilled(t, logs, client, 116)
		})
	}
}

// A tool_use block the filter held, whose arguments and stop arrive on the
// chunk the Router cuts on, reaches the client once, from the filter, and is
// billed. The encoder holds a tool call until it closes, so on that chunk it
// would have written the whole call -- under its own index.
func TestAHeldToolCallCompletedOnTheCutChunkIsSentOnceAndBilled(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := sameFormatMessagesContext(t)
	args := strings.Repeat("DELIVERED", 40)

	client := streamDeadlineResponseBody(t, router, ctx, sameFormatMessageStart()+
		"event: content_block_start\n"+
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`+"\n\n"+
		"event: content_block_delta\n"+
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thinking"}}`+"\n\n"+
		"event: content_block_start\n"+
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}`+"\n\n")
	ctx.StartTime = time.Now().Add(-600 * time.Second)
	client = append(client, streamDeadlineResponseBody(t, router, ctx,
		"event: content_block_delta\n"+
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"`+args+`\"}"}}`+"\n\n"+
			"event: content_block_stop\n"+`data: {"type":"content_block_stop","index":1}`+"\n\n")...)

	if got := bytes.Count(client, []byte(args)); got != 1 {
		t.Fatalf("the call's arguments reached the client %d times:\n%s", got, client)
	}
	delivered := wellFormedMessagesOutput(t, client)
	// The client holds 8 bytes of thinking and 368 of arguments. The count
	// also takes the start's own input, {}: 378 bytes, 95 tokens.
	if delivered.thinking+delivered.arguments != 376 {
		t.Fatalf("the client did not receive the call: %+v\n%s", delivered, client)
	}
	assertCutBilled(t, logs, client, 95)
}

// A turn cut while every block is in order ends on the filter's frames too:
// both deltas, the open block stopped, then the terminal.
func TestACutInOrderStreamEndsOnTheFiltersFrames(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := sameFormatMessagesContext(t)
	text := strings.Repeat("answer ", 20)

	client := streamDeadlineResponseBody(t, router, ctx, sameFormatMessageStart()+
		"event: content_block_start\n"+
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n"+
		"event: content_block_delta\n"+
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"`+text+`"}}`+"\n\n")
	ctx.StartTime = time.Now().Add(-600 * time.Second)
	client = append(client, streamDeadlineResponseBody(t, router, ctx,
		"event: content_block_delta\n"+
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"`+text+`"}}`+"\n\n")...)

	if delivered := wellFormedMessagesOutput(t, client); delivered.text != 2*len(text) {
		t.Fatalf("the client did not receive both deltas: %+v\n%s", delivered, client)
	}
	assertCutBilled(t, logs, client, 70)
}

// A cut chunk that also carries a frame the codec refuses, or an error the
// upstream sent, is not ended by the filter: the filter would publish the
// malformed frame the codec dropped, or end a provider's failure as the
// Router's cut. The codec's terminal ends it, as before.
func TestACutChunkThatFailsIsEndedByTheCodec(t *testing.T) {
	for name, tc := range map[string]struct{ chunk, absent, terminal string }{
		"refused frame": {
			chunk:    "event: content_block_delta\ndata: {]\n\n",
			absent:   "{]",
			terminal: `"type":"api_error"`,
		},
		"upstream error": {
			chunk: "event: error\n" +
				`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}` + "\n\n",
			absent:   "timeout_error",
			terminal: `"type":"overloaded_error"`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := sameFormatMessagesContext(t)
			client := streamDeadlineResponseBody(t, router, ctx, sameFormatMessageStart()+
				overlappingThinkingChunk(strings.Repeat("thinking ", 4)))
			ctx.StartTime = time.Now().Add(-600 * time.Second)
			client = append(client, streamDeadlineResponseBody(t, router, ctx, tc.chunk)...)

			if bytes.Contains(client, []byte(tc.absent)) {
				t.Fatalf("the client received %s:\n%s", tc.absent, client)
			}
			if !bytes.Contains(client, []byte(tc.terminal)) {
				t.Fatalf("the turn did not end on %s:\n%s", tc.terminal, client)
			}
		})
	}
}

// A cut that lands on the chunk where the upstream finished drops the
// upstream's own terminal frames, the [DONE] some providers send after
// message_stop included: a client that honours it would stop before the
// Router's terminal.
func TestACutOnTheUpstreamsLastChunkEndsOnOneTerminal(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := sameFormatMessagesContext(t)
	client := streamDeadlineResponseBody(t, router, ctx, sameFormatMessageStart()+
		overlappingThinkingChunk(strings.Repeat("thinking ", 4)))
	ctx.StartTime = time.Now().Add(-600 * time.Second)
	client = append(client, streamDeadlineResponseBody(t, router, ctx,
		"event: content_block_stop\n"+`data: {"type":"content_block_stop","index":0}`+"\n\n"+
			"event: content_block_stop\n"+`data: {"type":"content_block_stop","index":2}`+"\n\n"+
			"event: message_delta\n"+
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":100}}`+"\n\n"+
			"event: message_stop\n"+`data: {"type":"message_stop"}`+"\n\n"+
			"data: [DONE]\n\n")...)

	if bytes.Contains(client, []byte("[DONE]")) {
		t.Fatalf("the upstream's [DONE] reached the client beside the Router's terminal:\n%s", client)
	}
	wellFormedMessagesOutput(t, client)
}

func sameFormatMessagesContext(t *testing.T) *RequestContext {
	t.Helper()
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.TargetFormat = llmprotocol.AnthropicMessagesV1
	ctx.StartTime = time.Now()
	return ctx
}

// messagesOutput is the output a Messages client assembled, in bytes.
type messagesOutput struct {
	thinking, text, arguments, redacted int
}

// wellFormedMessagesOutput reads a Messages stream as a client that assembles
// blocks as they stop would: one block open at a time, each started once and
// stopped before the next starts, deltas and a stop only while it is open,
// every block stopped before the stream ends on one terminal, and nothing
// after that.
func wellFormedMessagesOutput(t *testing.T, body []byte) messagesOutput {
	t.Helper()
	var output messagesOutput
	open := map[int]bool{}
	started := map[int]bool{}
	terminals := 0
	for _, frame := range strings.Split(string(body), "\n\n") {
		_, data, found := strings.Cut(frame, "data: ")
		if !found {
			continue
		}
		if terminals > 0 {
			t.Fatalf("a frame followed the terminal:\n%s", body)
		}
		var event struct {
			Type         string `json:"type"`
			Index        int    `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
				Data string `json:"data"`
			} `json:"content_block"`
			Delta struct {
				Thinking    string `json:"thinking"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			t.Fatalf("frame is not JSON: %v\n%s", err, frame)
		}
		switch event.Type {
		case "content_block_start":
			if started[event.Index] {
				t.Fatalf("block %d started twice:\n%s", event.Index, body)
			}
			for index, isOpen := range open {
				if isOpen {
					t.Fatalf("block %d started while block %d was open:\n%s", event.Index, index, body)
				}
			}
			started[event.Index], open[event.Index] = true, true
			output.redacted += len(event.ContentBlock.Data)
		case "content_block_delta":
			if !open[event.Index] {
				t.Fatalf("a delta for block %d, which is not open:\n%s", event.Index, body)
			}
			output.thinking += len(event.Delta.Thinking)
			output.text += len(event.Delta.Text)
			output.arguments += len(event.Delta.PartialJSON)
		case "content_block_stop":
			if !open[event.Index] {
				t.Fatalf("a stop for block %d, which is not open:\n%s", event.Index, body)
			}
			open[event.Index] = false
		case "error", "message_stop":
			for index, isOpen := range open {
				if isOpen {
					t.Fatalf("the stream ended with block %d open:\n%s", index, body)
				}
			}
			terminals++
		}
	}
	if terminals != 1 {
		t.Fatalf("the stream ended on %d terminals:\n%s", terminals, body)
	}
	return output
}

// assertCutBilled checks the count on the cut's terminal and on its usage row.
func assertCutBilled(t *testing.T, logs *observer.ObservedLogs, client []byte, tokens int) {
	t.Helper()
	_, terminal, found := bytes.Cut(client, []byte("event: error"))
	if !found {
		t.Fatalf("the turn did not end on the cut's terminal:\n%s", client)
	}
	if want := fmt.Sprintf(`"output_tokens":%d,`, tokens); !bytes.Contains(terminal, []byte(want)) {
		t.Fatalf("the terminal did not bill %d tokens:\n%s", tokens, terminal)
	}
	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["completion_tokens"].(int64); got != int64(tokens) {
		t.Fatalf("the usage row billed %v, the terminal %d", fields["completion_tokens"], tokens)
	}
}

// overlappingThinkingChunk opens a thinking block and streams it, then starts
// a redacted thinking block and a text block while it is still open.
func overlappingThinkingChunk(thinking string) string {
	return "event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":` +
		`{"type":"thinking","thinking":"","signature":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":` +
		`{"type":"thinking_delta","thinking":"` + thinking + `"}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":` +
		`{"type":"redacted_thinking","data":"HELDPAYLOADHELDPAYLOADHELDPAYLOAD"}}` + "\n\n" +
		"event: content_block_stop\n" + `data: {"type":"content_block_stop","index":1}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":2,"delta":` +
		`{"type":"text_delta","text":"HELDANSWERHELDANSWERHELDANSWER"}}` + "\n\n"
}
