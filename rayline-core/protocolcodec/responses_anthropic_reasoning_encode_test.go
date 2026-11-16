package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// What claude-opus-5 on a Messages arm streams for a turn that thinks: signed
// thinking, a redacted block, then the answer.
func claudeSignedThinkingStream() string {
	frame := func(event, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	return frame("message_start", `{"type":"message_start","message":{"id":"msg_01XFDUDYJgAACzvnptvVoYEL","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":472,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":3}}}`) +
		frame("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`) +
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The test fails because the fixture "}}`) +
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"path is relative."}}`) +
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"`+claudeThinkingSignature+`"}}`) +
		frame("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		frame("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"`+claudeRedactedThinkingData+`"}}`) +
		frame("content_block_stop", `{"type":"content_block_stop","index":1}`) +
		frame("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`) +
		frame("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Make the fixture path absolute."}}`) +
		frame("content_block_stop", `{"type":"content_block_stop","index":2}`) +
		frame("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":87}}`) +
		frame("message_stop", `{"type":"message_stop"}`)
}

// The same turn, buffered.
const claudeSignedThinkingResponse = `{"id":"msg_01XFDUDYJgAACzvnptvVoYEL","type":"message","role":"assistant","model":"claude-opus-5",` +
	`"content":[` +
	`{"type":"thinking","thinking":"The test fails because the fixture path is relative.","signature":"` + claudeThinkingSignature + `"},` +
	`{"type":"redacted_thinking","data":"` + claudeRedactedThinkingData + `"},` +
	`{"type":"text","text":"Make the fixture path absolute."}],` +
	`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":87}}`

type responsesReasoningItemWire struct {
	Type             string          `json:"type"`
	Content          json.RawMessage `json:"content"`
	Signature        *string         `json:"signature"`
	Format           *string         `json:"format"`
	EncryptedContent *string         `json:"encrypted_content"`
}

func reasoningItemsOf(t *testing.T, output json.RawMessage) []responsesReasoningItemWire {
	t.Helper()
	var items []responsesReasoningItemWire
	if err := json.Unmarshal(output, &items); err != nil {
		t.Fatalf("output is not an item list: %v\n%s", err, output)
	}
	reasoning := items[:0]
	for _, item := range items {
		if item.Type == "reasoning" {
			reasoning = append(reasoning, item)
		}
	}
	return reasoning
}

// mintedClaudeItems decodes the Router-minted encrypted_content a Responses
// client got for one of Claude's blocks, and fails for any other value.
func mintedClaudeItems(t *testing.T, encrypted *string) []claudeReasoningItemWire {
	t.Helper()
	if encrypted == nil {
		t.Fatal("reasoning item carries no encrypted_content")
	}
	quoted, _ := json.Marshal(*encrypted)
	details, minted := mintedReasoningDetails(quoted)
	if !minted {
		t.Fatalf("encrypted_content %q is not Router-minted", *encrypted)
	}
	var items []claudeReasoningItemWire
	if err := json.Unmarshal(details, &items); err != nil {
		t.Fatalf("minted details: %v\n%s", err, details)
	}
	return items
}

// mintedSignaturesIn decodes every Router-minted encrypted_content in a wire
// body or stream and returns the signatures its items hold.
func mintedSignaturesIn(t *testing.T, wire string) []string {
	t.Helper()
	var signatures []string
	for _, blob := range regexp.MustCompile(`vsr\.reasoning_details\.v1\.[A-Za-z0-9_-]+`).FindAllString(wire, -1) {
		for _, item := range mintedClaudeItems(t, &blob) {
			signatures = append(signatures, item.Signature)
		}
	}
	return signatures
}

// assertClaudeReasoningItems checks the two reasoning items a Responses client
// gets for claude's turn. Each carries its block as the anthropic-claude-v1
// reasoning_details item in a Router-minted encrypted_content, the one member
// codex resends: the signed thinking with its text and signature, and the
// redacted block's data. Neither names OpenRouter's signature or format member.
func assertClaudeReasoningItems(t *testing.T, items []responsesReasoningItemWire) {
	t.Helper()
	if len(items) != 2 {
		t.Fatalf("reasoning items = %d, want 2 (signed thinking, redacted thinking): %+v", len(items), items)
	}
	signed, redacted := items[0], items[1]
	if signed.Signature != nil || signed.Format != nil || redacted.Signature != nil || redacted.Format != nil {
		t.Fatalf("a client was given members codex drops: %+v", items)
	}
	if !bytes.Contains(signed.Content, []byte(`"text":"The test fails because the fixture path is relative."`)) ||
		!bytes.Contains(signed.Content, []byte(`"type":"reasoning_text"`)) {
		t.Fatalf("signed thinking item content = %s, want the whole thinking as reasoning_text", signed.Content)
	}
	if carried := mintedClaudeItems(t, signed.EncryptedContent); len(carried) != 1 || carried[0] != (claudeReasoningItemWire{
		Type: "reasoning.text", Text: "The test fails because the fixture path is relative.",
		Signature: claudeThinkingSignature, Format: "anthropic-claude-v1",
	}) {
		t.Fatalf("signed thinking item carries %+v, want the thinking and its signature as one reasoning.text item", carried)
	}
	if carried := mintedClaudeItems(t, redacted.EncryptedContent); len(carried) != 1 || carried[0] != (claudeReasoningItemWire{
		Type: "reasoning.encrypted", Data: claudeRedactedThinkingData, Format: "anthropic-claude-v1",
	}) {
		t.Fatalf("redacted thinking item carries %+v, want the block's data as one reasoning.encrypted item", carried)
	}
}

// asCodexResends is an output item as codex resends it on its next turn: its
// ResponseItem::Reasoning keeps id, summary, content and encrypted_content
// and drops every other member (memex-desktop#8316).
func asCodexResends(t *testing.T, item string) string {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal([]byte(item), &object); err != nil {
		t.Fatalf("item: %v\n%s", err, item)
	}
	if string(object["type"]) != `"reasoning"` {
		return item
	}
	kept := map[string]json.RawMessage{}
	for _, name := range []string{"type", "id", "summary", "content", "encrypted_content"} {
		if value, present := object[name]; present {
			kept[name] = value
		}
	}
	resent, err := json.Marshal(kept)
	if err != nil {
		t.Fatal(err)
	}
	return string(resent)
}

// bufferedClaudeTurnItems is what a Responses client gets for the buffered
// turn, item by item.
func bufferedClaudeTurnItems(t *testing.T) []string {
	t.Helper()
	var response struct {
		Output []json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(translateAnthropicResponse(t, claudeSignedThinkingResponse, llmprotocol.OpenAIResponsesV1), &response); err != nil {
		t.Fatal(err)
	}
	items := make([]string, 0, len(response.Output))
	for _, item := range response.Output {
		items = append(items, string(item))
	}
	return items
}

// claudeTurnStreamFrames is the Responses stream a client gets for the live
// turn.
func claudeTurnStreamFrames(t *testing.T) [][]byte {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "auto", ProviderModel: "claude-opus-5",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(claudeSignedThinkingStream()))
	if err != nil {
		t.Fatalf("push: %v", err)
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	return append(frames, final...)
}

// streamedClaudeTurnItems is what a Responses client gets for the live turn:
// each output_item.done item, in order.
func streamedClaudeTurnItems(t *testing.T) []string {
	t.Helper()
	var items []string
	for _, event := range responsesStreamEvents(t, claudeTurnStreamFrames(t)) {
		if event.Type == "response.output_item.done" {
			items = append(items, string(event.Item))
		}
	}
	return items
}

// responsesInputReasoningItems are the reasoning items in a Responses
// worker's request.
func responsesInputReasoningItems(t *testing.T, body []byte) []responsesReasoningItemWire {
	t.Helper()
	var wire struct {
		Input json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("Responses body: %v\n%s", err, body)
	}
	return reasoningItemsOf(t, wire.Input)
}

// assertClaudeWorkerReasoningItems checks what a Claude worker over Responses
// is sent for the replayed turn: OpenRouter's shape, the signature and format
// on the thinking item and the redacted block's data as encrypted_content
// under the format (responses_anthropic_reasoning.go), and no minted blob.
func assertClaudeWorkerReasoningItems(t *testing.T, items []responsesReasoningItemWire) {
	t.Helper()
	if len(items) != 2 {
		t.Fatalf("reasoning items = %d, want 2 (signed thinking, redacted thinking): %+v", len(items), items)
	}
	signed, redacted := items[0], items[1]
	if signed.Signature == nil || *signed.Signature != claudeThinkingSignature || signed.Format == nil || *signed.Format != "anthropic-claude-v1" ||
		signed.EncryptedContent != nil || !bytes.Contains(signed.Content, []byte(`"text":"The test fails because the fixture path is relative."`)) {
		t.Fatalf("signed thinking item = %+v, want the text with the signature and format members", signed)
	}
	if redacted.EncryptedContent == nil || *redacted.EncryptedContent != claudeRedactedThinkingData ||
		redacted.Format == nil || *redacted.Format != "anthropic-claude-v1" || redacted.Signature != nil {
		t.Fatalf("redacted thinking item = %+v, want the block's data under the format and no signature", redacted)
	}
}

type responsesStreamEventWire struct {
	Type     string          `json:"type"`
	Item     json.RawMessage `json:"item"`
	Response struct {
		Status string          `json:"status"`
		Output json.RawMessage `json:"output"`
	} `json:"response"`
}

func responsesStreamEvents(t *testing.T, frames [][]byte) []responsesStreamEventWire {
	t.Helper()
	var events []responsesStreamEventWire
	for _, frame := range frames {
		text := string(frame)
		start := strings.Index(text, "data: ")
		if start < 0 {
			continue
		}
		var event responsesStreamEventWire
		if err := json.Unmarshal([]byte(strings.TrimSpace(text[start+len("data: "):])), &event); err != nil {
			t.Fatalf("frame is not JSON: %v\n%s", err, text)
		}
		events = append(events, event)
	}
	return events
}

// A Responses client routed to a Messages arm that thinks gets the turn as a
// stream that completes, with Claude's signature in each reasoning item's
// minted encrypted_content, on the item's done event and in the completed
// response.
func TestClaudeSignedThinkingStreamsToAResponsesClient(t *testing.T) {
	frames := claudeTurnStreamFrames(t)
	var done []responsesReasoningItemWire
	var completed json.RawMessage
	for _, event := range responsesStreamEvents(t, frames) {
		switch event.Type {
		case "response.output_item.done":
			done = append(done, reasoningItemsOf(t, json.RawMessage("["+string(event.Item)+"]"))...)
		case "response.completed":
			completed = event.Response.Output
		case "response.failed", "error":
			t.Fatalf("stream failed: %s", bytes.Join(frames, nil))
		}
	}
	if completed == nil {
		t.Fatalf("stream did not complete:\n%s", bytes.Join(frames, nil))
	}
	assertClaudeReasoningItems(t, done)
	assertClaudeReasoningItems(t, reasoningItemsOf(t, completed))
	if !bytes.Contains(completed, []byte("Make the fixture path absolute.")) {
		t.Fatalf("the answer was lost: %s", completed)
	}
}

// The buffered turn gives a Responses client the same items.
func TestClaudeSignedThinkingResponseToAResponsesClient(t *testing.T) {
	body := translateAnthropicResponse(t, claudeSignedThinkingResponse, llmprotocol.OpenAIResponsesV1)
	var response struct {
		Output json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		t.Fatalf("response: %v\n%s", err, body)
	}
	assertClaudeReasoningItems(t, reasoningItemsOf(t, response.Output))
	if !bytes.Contains(response.Output, []byte("Make the fixture path absolute.")) {
		t.Fatalf("the answer was lost: %s", response.Output)
	}
}

// Two signed thinking blocks in one message are two items: one signature
// proves one block, so merging them would resend text it does not sign.
func TestEachSignedThinkingBlockIsItsOwnResponsesItem(t *testing.T) {
	body := `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5","content":[` +
		`{"type":"thinking","thinking":"first","signature":"sig-first"},` +
		`{"type":"thinking","thinking":"second","signature":"sig-second"},` +
		`{"type":"text","text":"answer"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`
	var response struct {
		Output json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(translateAnthropicResponse(t, body, llmprotocol.OpenAIResponsesV1), &response); err != nil {
		t.Fatal(err)
	}
	items := reasoningItemsOf(t, response.Output)
	if len(items) != 2 || !bytes.Contains(items[0].Content, []byte(`"first"`)) || bytes.Contains(items[0].Content, []byte(`"second"`)) {
		t.Fatalf("reasoning items = %s, want one per signed block", response.Output)
	}
	first, second := mintedClaudeItems(t, items[0].EncryptedContent), mintedClaudeItems(t, items[1].EncryptedContent)
	if len(first) != 1 || first[0].Signature != "sig-first" || first[0].Text != "first" ||
		len(second) != 1 || second[0].Signature != "sig-second" || second[0].Text != "second" {
		t.Fatalf("carried items = %+v and %+v, want each block under its own signature", first, second)
	}
}

// Control: OpenAI's own encrypted reasoning reaches a Responses client as
// before, without the Anthropic format tag or a signature.
func TestOpenAIEncryptedReasoningKeepsItsShape(t *testing.T) {
	var response struct {
		Output json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(translateResponsesResponse(t, encryptedReasoningResponse, llmprotocol.OpenAIResponsesV1), &response); err != nil {
		t.Fatal(err)
	}
	assertOpenAIEncryptedReasoning(t, reasoningItemsOf(t, response.Output))

	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "gpt-5-codex",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(encryptedReasoningStream()))
	if err != nil {
		t.Fatalf("push: %v", err)
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	for _, event := range responsesStreamEvents(t, append(frames, final...)) {
		if event.Type == "response.completed" {
			assertOpenAIEncryptedReasoning(t, reasoningItemsOf(t, event.Response.Output))
			return
		}
	}
	t.Fatal("stream did not complete")
}

func assertOpenAIEncryptedReasoning(t *testing.T, items []responsesReasoningItemWire) {
	t.Helper()
	if len(items) != 1 || items[0].EncryptedContent == nil || *items[0].EncryptedContent != "gAAAAB-issued" {
		t.Fatalf("reasoning items = %+v, want the one encrypted item", items)
	}
	if items[0].Format != nil || items[0].Signature != nil {
		t.Fatalf("OpenAI's encrypted reasoning gained format %v / signature %v", items[0].Format, items[0].Signature)
	}
}

// The whole loop, from the buffered turn and from the stream: the items a
// Responses client is given for claude's turn, resent as codex resends them,
// reach the next Claude worker over Messages as the identical thinking and
// redacted_thinking blocks, and over Responses as the signature, format and
// data OpenRouter's /responses takes back.
func TestClaudeTurnRoundTripsThroughAResponsesClient(t *testing.T) {
	for name, given := range map[string][]string{"buffered": bufferedClaudeTurnItems(t), "streamed": streamedClaudeTurnItems(t)} {
		items := make([]string, 0, len(given))
		for _, item := range given {
			items = append(items, asCodexResends(t, item))
		}
		t.Run(name+"/messages worker", func(t *testing.T) {
			_, body := dispatchResponsesRequestToClaude(t, replayedTurn(items), llmprotocol.AnthropicMessagesV1)
			var thinking, redacted bool
			for _, block := range anthropicAssistantBlocks(t, body) {
				switch block.Type {
				case "thinking":
					thinking = block.Signature == claudeThinkingSignature && block.Thinking == "The test fails because the fixture path is relative."
				case "redacted_thinking":
					redacted = block.Data == claudeRedactedThinkingData
				}
			}
			if !thinking || !redacted {
				t.Fatalf("the replayed turn lost Claude's reasoning (thinking=%v redacted=%v):\n%s", thinking, redacted, body)
			}
		})
		t.Run(name+"/responses worker", func(t *testing.T) {
			_, body := dispatchResponsesRequestToClaude(t, replayedTurn(items), llmprotocol.OpenAIResponsesV1)
			assertClaudeWorkerReasoningItems(t, responsesInputReasoningItems(t, body))
			if bytes.Contains(body, []byte(mintedReasoningDetailsPrefix)) {
				t.Fatalf("a Claude worker was sent the Router's minted blob:\n%s", body)
			}
		})
	}
}

// A buffered Claude answer with two consecutive signed thinking blocks, as
// the response cache holds it, replays as a Responses stream with one
// reasoning item per block, each under its own signature: grouped, the two
// signatures would concatenate into one no provider accepts.
func TestCachedSignedThinkingReplaysOneItemPerBlock(t *testing.T) {
	engine := NewBuiltinEngine()
	response, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5",
		"content":[
			{"type":"thinking","thinking":"first","signature":"sigA"},
			{"type":"thinking","thinking":"second","signature":"sigB"},
			{"type":"text","text":"done"}
		],
		"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":3}}`))
	if err != nil {
		t.Fatal(err)
	}
	wire, _, err := engine.EncodeResponseStream(llmprotocol.OpenAIResponsesV1, response, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	var signatures []string
	for _, event := range responsesStreamEvents(t, sseFrames(wire)) {
		if event.Type != "response.output_item.done" {
			continue
		}
		for _, item := range reasoningItemsOf(t, json.RawMessage("["+string(event.Item)+"]")) {
			for _, carried := range mintedClaudeItems(t, item.EncryptedContent) {
				signatures = append(signatures, carried.Signature)
			}
		}
	}
	if len(signatures) != 2 || signatures[0] != "sigA" || signatures[1] != "sigB" {
		t.Fatalf("the signatures were not kept apart: %q\n%s", signatures, wire)
	}
}

// An unsigned reasoning block right before a signed one stays out of the
// signed block's item, buffered and replayed as a stream: the signature
// covers only the block it was issued for.
func TestSignedThinkingStartsAFreshResponsesItem(t *testing.T) {
	engine := NewBuiltinEngine()
	response, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5",
		"content":[
			{"type":"thinking","thinking":"unsigned","signature":"dropme"},
			{"type":"thinking","thinking":"signed","signature":"sigB"},
			{"type":"text","text":"done"}
		],
		"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":3}}`))
	if err != nil {
		t.Fatal(err)
	}
	// The first block carried no signature, as a cached or synthetic answer may.
	response.Output[0].Content[0].Signature = ""
	buffered, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, response, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	streamed, _, err := engine.EncodeResponseStream(llmprotocol.OpenAIResponsesV1, response, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	var bufferedWire struct {
		Output json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(buffered.Body, &bufferedWire); err != nil {
		t.Fatal(err)
	}
	var streamedItems []responsesReasoningItemWire
	for _, event := range responsesStreamEvents(t, sseFrames(streamed)) {
		if event.Type == "response.output_item.done" {
			streamedItems = append(streamedItems, reasoningItemsOf(t, json.RawMessage("["+string(event.Item)+"]"))...)
		}
	}
	for name, items := range map[string][]responsesReasoningItemWire{"buffered": reasoningItemsOf(t, bufferedWire.Output), "streamed": streamedItems} {
		// The signed block's item holds its own text only.
		signed := -1
		for index, item := range items {
			if item.EncryptedContent != nil && mintedClaudeItems(t, item.EncryptedContent)[0].Signature == "sigB" {
				signed = index
			}
		}
		if signed < 0 {
			t.Fatalf("%s: no signed item: %+v", name, items)
		}
		if bytes.Contains(items[signed].Content, []byte("unsigned")) || !bytes.Contains(items[signed].Content, []byte(`"signed"`)) {
			t.Fatalf("%s: the unsigned block is inside the signed item: %s", name, items[signed].Content)
		}
	}
}

// A cached Claude answer with signed and redacted thinking replays as a
// stream, as the live provider stream does: the redacted block arrives as
// one item started and completed, carrying its data. Checked for a Responses
// client and for an Anthropic one.
func TestCachedRedactedThinkingReplaysAsAStream(t *testing.T) {
	engine := NewBuiltinEngine()
	response, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5",
		"content":[
			{"type":"thinking","thinking":"plan","signature":"sigA"},
			{"type":"redacted_thinking","data":"REDACTED_DATA_1"},
			{"type":"text","text":"done"}
		],
		"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":3}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		wire, _, err := engine.EncodeResponseStream(target, response, llmprotocol.StreamContext{
			Context: context.Background(), PublicModel: "public-model",
		})
		if err != nil {
			t.Fatalf("%s: replay: %v", target, err)
		}
		if target == llmprotocol.OpenAIResponsesV1 {
			var carried []claudeReasoningItemWire
			for _, event := range responsesStreamEvents(t, sseFrames(wire)) {
				if event.Type != "response.output_item.done" {
					continue
				}
				for _, item := range reasoningItemsOf(t, json.RawMessage("["+string(event.Item)+"]")) {
					carried = append(carried, mintedClaudeItems(t, item.EncryptedContent)...)
				}
			}
			if len(carried) != 2 || carried[0].Signature != "sigA" || carried[0].Text != "plan" ||
				carried[1].Type != "reasoning.encrypted" || carried[1].Data != "REDACTED_DATA_1" || !strings.Contains(string(wire), "done") {
				t.Fatalf("%s: the replay lost a block: %+v\n%s", target, carried, wire)
			}
			continue
		}
		if !strings.Contains(string(wire), "REDACTED_DATA_1") || !strings.Contains(string(wire), "sigA") || !strings.Contains(string(wire), "done") {
			t.Fatalf("%s: the replay lost a block:\n%s", target, wire)
		}
	}
}
