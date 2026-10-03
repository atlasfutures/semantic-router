package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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

// assertClaudeReasoningItems checks the two reasoning items a Responses client
// gets for claude's turn: the signed thinking with its text, and the redacted
// block's data, both under the Anthropic format tag.
func assertClaudeReasoningItems(t *testing.T, items []responsesReasoningItemWire) {
	t.Helper()
	if len(items) != 2 {
		t.Fatalf("reasoning items = %d, want 2 (signed thinking, redacted thinking): %+v", len(items), items)
	}
	signed, redacted := items[0], items[1]
	if signed.Signature == nil || *signed.Signature != claudeThinkingSignature {
		t.Fatalf("signed thinking item signature = %v, want the provider's", signed.Signature)
	}
	if signed.Format == nil || *signed.Format != "anthropic-claude-v1" {
		t.Fatalf("signed thinking item format = %v, want anthropic-claude-v1", signed.Format)
	}
	if !bytes.Contains(signed.Content, []byte(`"text":"The test fails because the fixture path is relative."`)) ||
		!bytes.Contains(signed.Content, []byte(`"type":"reasoning_text"`)) {
		t.Fatalf("signed thinking item content = %s, want the whole thinking as reasoning_text", signed.Content)
	}
	if signed.EncryptedContent != nil {
		t.Fatalf("signed thinking item carries encrypted_content %q", *signed.EncryptedContent)
	}
	if redacted.EncryptedContent == nil || *redacted.EncryptedContent != claudeRedactedThinkingData {
		t.Fatalf("redacted thinking item encrypted_content = %v, want the block's data", redacted.EncryptedContent)
	}
	if redacted.Format == nil || *redacted.Format != "anthropic-claude-v1" || redacted.Signature != nil {
		t.Fatalf("redacted thinking item format = %v signature = %v, want the Anthropic format and no signature", redacted.Format, redacted.Signature)
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

// A Responses client routed to a Messages arm that thinks: the stream used to
// fail after its 200 went out ("translation would lose reasoning.signature").
// It now completes, and the client holds the signature to resend.
func TestClaudeSignedThinkingStreamsToAResponsesClient(t *testing.T) {
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
	var done []responsesReasoningItemWire
	var completed json.RawMessage
	for _, event := range responsesStreamEvents(t, append(frames, final...)) {
		switch event.Type {
		case "response.output_item.done":
			done = append(done, reasoningItemsOf(t, json.RawMessage("["+string(event.Item)+"]"))...)
		case "response.completed":
			completed = event.Response.Output
		case "response.failed", "error":
			t.Fatalf("stream failed: %s", bytes.Join(append(frames, final...), nil))
		}
	}
	if completed == nil {
		t.Fatalf("stream did not complete:\n%s", bytes.Join(append(frames, final...), nil))
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
	if len(items) != 2 || items[0].Signature == nil || *items[0].Signature != "sig-first" ||
		items[1].Signature == nil || *items[1].Signature != "sig-second" ||
		!bytes.Contains(items[0].Content, []byte(`"first"`)) || bytes.Contains(items[0].Content, []byte(`"second"`)) {
		t.Fatalf("reasoning items = %s, want one per signed block", response.Output)
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

// The whole loop: the items a Responses client is given for claude's turn,
// resent verbatim, reach the next Messages worker as the identical thinking
// and redacted_thinking blocks.
func TestClaudeTurnRoundTripsThroughAResponsesClient(t *testing.T) {
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
	_, body := dispatchResponsesRequest(t, replayedTurn(items), llmprotocol.AnthropicMessagesV1)
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
	stream := string(wire)
	if strings.Contains(stream, "sigAsigB") || strings.Count(stream, `"signature":"sigA"`) == 0 || strings.Count(stream, `"signature":"sigB"`) == 0 {
		t.Fatalf("the signatures were not kept apart:\n%s", stream)
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
	for name, wire := range map[string][]byte{"buffered": buffered.Body, "streamed": streamed} {
		// The signed block's item holds its own text only.
		index := strings.Index(string(wire), `"signature":"sigB"`)
		if index < 0 {
			t.Fatalf("%s: no signed item:\n%s", name, wire)
		}
		start := strings.LastIndex(string(wire[:index]), `"type":"reasoning"`)
		if start < 0 || strings.Contains(string(wire[start:index]), "unsigned") {
			t.Fatalf("%s: the unsigned block is inside the signed item:\n%s", name, wire)
		}
	}
}
