package extproc

import (
	"bytes"
	"testing"

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
