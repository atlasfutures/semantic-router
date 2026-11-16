package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// OpenRouter streams Claude's thinking over Responses as a reasoning item
// whose reasoning_text part is announced with content_part.added and closed
// with content_part.done, the same events a message uses for output_text:
//
//	output_item.added    {item: {type: reasoning}}
//	content_part.added   {part: {type: reasoning_text, text: ""}}
//	reasoning_text.delta / reasoning_text.done
//	content_part.done    {part: {type: reasoning_text, text}}
//	output_item.done     {item: {type: reasoning, content: [reasoning_text], signature, format}}
//
// The decoder took every content_part event for a message part and aborted
// the stream with stream_item_kind_mismatch, so every Claude turn that
// thought before calling a tool failed. The fixture is a captured stream of
// that shape with its ids, text, signature and tools replaced.
const openRouterClaudeReasoningStream = "responses-stream-claude-reasoning.sse"

const openRouterClaudeReasoningText = "The user wants 6*7 evaluated with the js tool, so I will call it."

func pushResponsesStream(body []byte, client llmprotocol.WireFormat) ([]byte, error) {
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, client, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "anthropic/claude-sonnet-5.5",
		ProviderModel: "anthropic/claude-sonnet-5.5", WorkerFamily: "anthropic",
	})
	if err != nil {
		return nil, err
	}
	frames, _, _, err := stream.Push(body)
	if err != nil {
		return nil, err
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		return nil, err
	}
	return bytes.Join(append(frames, final...), nil), nil
}

func TestOpenRouterClaudeReasoningStreamReachesEveryClient(t *testing.T) {
	body := loadProviderFixture(t, openRouterClaudeReasoningStream)
	for _, client := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(client), func(t *testing.T) {
			routed, err := pushResponsesStream(body, client)
			if err != nil {
				t.Fatalf("stream failed: %v", err)
			}
			for _, want := range []string{"6*7 evaluated ", `\"code\":`} {
				if !bytes.Contains(routed, []byte(want)) {
					t.Fatalf("routed stream lost %q:\n%s", want, routed)
				}
			}
		})
	}
}

var responsesSequenceNumber = regexp.MustCompile(`"sequence_number":(\d+)`)

// dropResponsesStreamEvent removes the event with sequence number `sequence`
// from an SSE body and renumbers the events after it, since the decoder
// requires contiguous sequence numbers.
func dropResponsesStreamEvent(t *testing.T, body string, sequence int) string {
	t.Helper()
	events := strings.SplitAfter(body, "\n\n")
	kept := events[:0]
	for _, event := range events {
		match := responsesSequenceNumber.FindStringSubmatch(event)
		if match != nil && match[1] == strconv.Itoa(sequence) && strings.HasPrefix(event, "data: ") {
			continue
		}
		kept = append(kept, event)
	}
	if len(kept) != len(events)-1 {
		t.Fatalf("the fixture has no single event with sequence number %d", sequence)
	}
	return responsesSequenceNumber.ReplaceAllStringFunc(strings.Join(kept, ""), func(field string) string {
		number, _ := strconv.Atoi(responsesSequenceNumber.FindStringSubmatch(field)[1])
		if number > sequence {
			number--
		}
		return `"sequence_number":` + strconv.Itoa(number)
	})
}

// A reasoning_text part may open with text of its own, which is the first
// reasoning delta and reaches every client like the deltas after it.
func TestOpenRouterClaudeReasoningInitialPartTextReachesEveryClient(t *testing.T) {
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	if !strings.Contains(body, `"delta":"The user wants ","sequence_number":4}`) {
		t.Fatal("the fixture's first reasoning delta is no longer event 4")
	}
	mutated := strings.Replace(body,
		`"part":{"type":"reasoning_text","text":""}`, `"part":{"type":"reasoning_text","text":"The user wants "}`, 1)
	if mutated == body {
		t.Fatal("the fixture no longer opens an empty reasoning_text part")
	}
	mutated = dropResponsesStreamEvent(t, mutated, 4)
	for _, client := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(client), func(t *testing.T) {
			routed, err := pushResponsesStream([]byte(mutated), client)
			if err != nil {
				t.Fatalf("stream failed: %v", err)
			}
			if !bytes.Contains(routed, []byte(`"The user wants "`)) {
				t.Fatalf("routed stream lost the part's initial text:\n%s", routed)
			}
		})
	}
}

// A reasoning_text part that content_part.added opened must be closed by its
// content_part.done, and a content_part.done must close a part that
// content_part.added opened, not reasoning text that started with a delta.
func TestResponsesReasoningTextPartMustCloseAsItOpened(t *testing.T) {
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	for _, check := range []struct {
		name     string
		sequence int
		event    string
		code     string
	}{
		{"part opened and never closed", 9, "response.content_part.done", "stream_item_content_incomplete"},
		{"part closed and never opened", 3, "response.content_part.added", "stream_content_part_start_missing"},
	} {
		t.Run(check.name, func(t *testing.T) {
			event := regexp.MustCompile(`(?m)^data: \{"type":"` + regexp.QuoteMeta(check.event) +
				`",.*"sequence_number":` + strconv.Itoa(check.sequence) + `\}$`)
			if !event.MatchString(body) {
				t.Fatalf("the fixture's event %d is no longer %s", check.sequence, check.event)
			}
			_, err := pushResponsesStream([]byte(dropResponsesStreamEvent(t, body, check.sequence)), llmprotocol.OpenAIResponsesV1)
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Code != check.code {
				t.Fatalf("got %v, want %s", err, check.code)
			}
		})
	}
}

// A reasoning_text part is reasoning content: it belongs to a reasoning item
// and to no other, and a message part still belongs only to a message.
func TestResponsesContentPartMustMatchItsItem(t *testing.T) {
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	for name, mutated := range map[string]string{
		"reasoning_text part on a message": strings.Replace(body,
			`"item":{"id":"rs_fixture_1","type":"reasoning","status":"in_progress","summary":[]}`,
			`"item":{"id":"rs_fixture_1","type":"message","role":"assistant","status":"in_progress","content":[]}`, 1),
		"output_text part on a reasoning item": strings.Replace(body,
			`"part":{"type":"reasoning_text","text":""}`,
			`"part":{"type":"output_text","text":"","annotations":[]}`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			if mutated == body {
				t.Fatal("the fixture no longer has the event this case rewrites")
			}
			_, err := pushResponsesStream([]byte(mutated), llmprotocol.OpenAIResponsesV1)
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Code != "stream_item_kind_mismatch" {
				t.Fatalf("got %v, want stream_item_kind_mismatch", err)
			}
		})
	}
}

// The fixture's reasoning item is signed, as a Claude thinking block is, and
// its done event names the signature and format OpenRouter gives Claude's
// thinking over Responses (responsesAnthropicReasoning).
const openRouterClaudeReasoningSignature = "RXJmaXh0dXJlLXNpZ25hdHVyZS1ub3QtYS1yZWFsLW9uZQ=="

const openRouterClaudeReasoningItem = `{"id":"rs_fixture_1","type":"reasoning","status":"completed",` +
	`"content":[{"type":"reasoning_text","text":"` + openRouterClaudeReasoningText + `"}],"summary":[],` +
	`"signature":"` + openRouterClaudeReasoningSignature + `","format":"anthropic-claude-v1"}`

// openRouterClaudeRedactedStream is the fixture with its thinking redacted: the
// reasoning item streams no content, and its done event carries the block's
// data as encrypted_content under the same format, the shape OpenRouter's
// buffered /responses gives a redacted_thinking block. No stream of one has
// been captured; this is the streamed form of that item.
func openRouterClaudeRedactedStream(t *testing.T) string {
	t.Helper()
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	for range 7 { // content_part.added through content_part.done, events 3 to 9
		body = dropResponsesStreamEvent(t, body, 3)
	}
	redacted := `{"id":"rs_fixture_1","type":"reasoning","status":"completed","summary":[],` +
		`"encrypted_content":"` + claudeRedactedThinkingData + `","format":"anthropic-claude-v1"}`
	if strings.Count(body, openRouterClaudeReasoningItem) != 2 {
		t.Fatal("the fixture's reasoning item is no longer on its done event and the completed output")
	}
	return strings.ReplaceAll(body, openRouterClaudeReasoningItem, redacted)
}

func sseFrames(routed []byte) [][]byte {
	var frames [][]byte
	for _, frame := range bytes.Split(routed, []byte("\n\n")) {
		if len(bytes.TrimSpace(frame)) > 0 {
			frames = append(frames, frame)
		}
	}
	return frames
}

// responsesStreamReasoning returns the reasoning items a Responses client got
// on output_item.done, and those in the completed response's output.
func responsesStreamReasoning(t *testing.T, routed []byte) (done, completed []responsesReasoningItemWire) {
	t.Helper()
	for _, event := range responsesStreamEvents(t, sseFrames(routed)) {
		switch event.Type {
		case "response.output_item.done":
			var item responsesReasoningItemWire
			if err := json.Unmarshal(event.Item, &item); err != nil {
				t.Fatalf("done item: %v\n%s", err, event.Item)
			}
			if item.Type == "reasoning" {
				done = append(done, item)
			}
		case "response.completed":
			completed = reasoningItemsOf(t, event.Response.Output)
		}
	}
	return done, completed
}

type anthropicStreamEventWire struct {
	Type         string              `json:"type"`
	Index        int                 `json:"index"`
	ContentBlock *anthropicBlockWire `json:"content_block"`
	Delta        *struct {
		Type      string `json:"type"`
		Thinking  string `json:"thinking"`
		Signature string `json:"signature"`
	} `json:"delta"`
}

func anthropicStreamEvents(t *testing.T, routed []byte) []anthropicStreamEventWire {
	t.Helper()
	var events []anthropicStreamEventWire
	for _, frame := range sseFrames(routed) {
		text := string(frame)
		start := strings.Index(text, "data: ")
		if start < 0 {
			continue
		}
		var event anthropicStreamEventWire
		if err := json.Unmarshal([]byte(strings.TrimSpace(text[start+len("data: "):])), &event); err != nil {
			t.Fatalf("frame is not JSON: %v\n%s", err, text)
		}
		events = append(events, event)
	}
	return events
}

// replayStreamToMessagesWorker resends the items a Responses client got from
// the stream, as codex does on its next turn, and returns the assistant blocks
// a Messages worker is then sent.
func replayStreamToMessagesWorker(t *testing.T, routed []byte) []anthropicBlockWire {
	t.Helper()
	var items []string
	for _, event := range responsesStreamEvents(t, sseFrames(routed)) {
		if event.Type == "response.output_item.done" {
			items = append(items, string(event.Item))
		}
	}
	items = append(items, `{"type":"function_call_output","call_id":"toolu_fixture_1","output":"42"}`)
	_, body := dispatchResponsesRequestToClaude(t, replayedTurn(items), llmprotocol.AnthropicMessagesV1)
	return anthropicAssistantBlocks(t, body)
}

// The signature and format on the item's done event reach each client the
// way a Messages arm's signature_delta does: a Responses client gets the
// signed thinking as a minted encrypted_content, the one member codex resends
// (memex-desktop#8316: codex drops signature and format on replay), and the
// client's replay reaches a Messages worker as signed thinking.
func TestOpenRouterClaudeReasoningSignatureReachesEveryClient(t *testing.T) {
	body := loadProviderFixture(t, openRouterClaudeReasoningStream)
	t.Run(string(llmprotocol.OpenAIResponsesV1), func(t *testing.T) {
		routed, err := pushResponsesStream(body, llmprotocol.OpenAIResponsesV1)
		if err != nil {
			t.Fatalf("stream failed: %v", err)
		}
		done, completed := responsesStreamReasoning(t, routed)
		for label, items := range map[string][]responsesReasoningItemWire{"output_item.done": done, "response.completed": completed} {
			if len(items) != 1 {
				t.Fatalf("%s: %d reasoning items, want 1:\n%s", label, len(items), routed)
			}
			item := items[0]
			if item.Signature != nil || item.Format != nil || !bytes.Contains(item.Content, []byte(openRouterClaudeReasoningText)) {
				t.Fatalf("%s: reasoning item = %+v, want the fixture's text and no member codex drops", label, item)
			}
			if carried := mintedClaudeItems(t, item.EncryptedContent); len(carried) != 1 || carried[0] != (claudeReasoningItemWire{
				Type: "reasoning.text", Text: openRouterClaudeReasoningText,
				Signature: openRouterClaudeReasoningSignature, Format: "anthropic-claude-v1",
			}) {
				t.Fatalf("%s: encrypted_content carries %+v, want the fixture's text and signature as one reasoning.text item", label, carried)
			}
		}
		blocks := replayStreamToMessagesWorker(t, routed)
		if len(blocks) == 0 || blocks[0].Type != "thinking" || blocks[0].Signature != openRouterClaudeReasoningSignature ||
			blocks[0].Thinking != openRouterClaudeReasoningText {
			t.Fatalf("a Messages worker is sent %+v, want the signed thinking first", blocks)
		}
	})
	t.Run(string(llmprotocol.OpenAIChatV1), func(t *testing.T) {
		routed, err := pushResponsesStream(body, llmprotocol.OpenAIChatV1)
		if err != nil {
			t.Fatalf("stream failed: %v", err)
		}
		merged, _, _ := chatStreamReasoningDetails(t, string(routed))
		assertEqualReasoningJSON(t, "reasoning_details", merged, json.RawMessage(`[{"type":"reasoning.text",`+
			`"text":"`+openRouterClaudeReasoningText+`","signature":"`+openRouterClaudeReasoningSignature+`",`+
			`"format":"anthropic-claude-v1","index":0}]`))
	})
	t.Run(string(llmprotocol.AnthropicMessagesV1), func(t *testing.T) {
		routed, err := pushResponsesStream(body, llmprotocol.AnthropicMessagesV1)
		if err != nil {
			t.Fatalf("stream failed: %v", err)
		}
		var thinking, signatures []string
		for _, event := range anthropicStreamEvents(t, routed) {
			switch {
			case event.Type == "content_block_start" && event.ContentBlock != nil && event.ContentBlock.Type == "thinking":
				thinking = append(thinking, event.ContentBlock.Signature)
			case event.Type == "content_block_delta" && event.Delta != nil && event.Delta.Type == "signature_delta":
				signatures = append(signatures, event.Delta.Signature)
			}
		}
		if len(thinking) != 1 || len(signatures) != 1 || signatures[0] != openRouterClaudeReasoningSignature {
			t.Fatalf("thinking blocks %q, signature deltas %q; want one block and the fixture's signature:\n%s", thinking, signatures, routed)
		}
	})
}

// A redacted item is the carried redacted_thinking block to every client, as
// it is from a Messages arm and on the buffered path, and replays as one. A
// Messages client gets no thinking block for it: the item's start is held
// until the item shows what it is.
func TestOpenRouterClaudeRedactedThinkingStreamReachesEveryClient(t *testing.T) {
	body := []byte(openRouterClaudeRedactedStream(t))
	t.Run(string(llmprotocol.OpenAIResponsesV1), func(t *testing.T) {
		routed, err := pushResponsesStream(body, llmprotocol.OpenAIResponsesV1)
		if err != nil {
			t.Fatalf("stream failed: %v", err)
		}
		done, completed := responsesStreamReasoning(t, routed)
		for label, items := range map[string][]responsesReasoningItemWire{"output_item.done": done, "response.completed": completed} {
			if len(items) != 1 {
				t.Fatalf("%s: %d reasoning items, want 1:\n%s", label, len(items), routed)
			}
			item := items[0]
			if item.Signature != nil || item.Format != nil {
				t.Fatalf("%s: reasoning item = %+v, want no member codex drops", label, item)
			}
			if carried := mintedClaudeItems(t, item.EncryptedContent); len(carried) != 1 || carried[0] != (claudeReasoningItemWire{
				Type: "reasoning.encrypted", Data: claudeRedactedThinkingData, Format: "anthropic-claude-v1",
			}) {
				t.Fatalf("%s: encrypted_content carries %+v, want the block's data as one reasoning.encrypted item", label, carried)
			}
		}
		blocks := replayStreamToMessagesWorker(t, routed)
		if len(blocks) == 0 || blocks[0].Type != "redacted_thinking" || blocks[0].Data != claudeRedactedThinkingData {
			t.Fatalf("a Messages worker is sent %+v, want the redacted_thinking block first", blocks)
		}
	})
	t.Run(string(llmprotocol.OpenAIChatV1), func(t *testing.T) {
		routed, err := pushResponsesStream(body, llmprotocol.OpenAIChatV1)
		if err != nil {
			t.Fatalf("stream failed: %v", err)
		}
		merged, _, _ := chatStreamReasoningDetails(t, string(routed))
		assertEqualReasoningJSON(t, "reasoning_details", merged, json.RawMessage(`[{"type":"reasoning.encrypted",`+
			`"data":"`+claudeRedactedThinkingData+`","format":"anthropic-claude-v1","index":0}]`))
	})
	t.Run(string(llmprotocol.AnthropicMessagesV1), func(t *testing.T) {
		routed, err := pushResponsesStream(body, llmprotocol.AnthropicMessagesV1)
		if err != nil {
			t.Fatalf("stream failed: %v", err)
		}
		var blocks []anthropicBlockWire
		for _, event := range anthropicStreamEvents(t, routed) {
			if event.Type == "content_block_start" && event.ContentBlock != nil {
				blocks = append(blocks, *event.ContentBlock)
			}
		}
		if len(blocks) != 2 || blocks[0].Type != "redacted_thinking" || blocks[0].Data != claudeRedactedThinkingData || blocks[1].Type != "tool_use" {
			t.Fatalf("content blocks = %+v, want the redacted_thinking block and the tool call:\n%s", blocks, routed)
		}
	})
}

// One Anthropic block is one item: a reasoning item that streamed its text, or
// that is signed, has no encrypted_content. Its start has gone out as thinking,
// so the stream is refused rather than losing either member.
func TestResponsesReasoningItemWithContentRefusesEncryptedContent(t *testing.T) {
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	mutated := strings.Replace(body, `"summary":[],"signature":"`, `"summary":[],"encrypted_content":"`+claudeRedactedThinkingData+`","signature":"`, 1)
	if mutated == body {
		t.Fatal("the fixture's done event no longer has the item this case rewrites")
	}
	_, err := pushResponsesStream([]byte(mutated), llmprotocol.OpenAIResponsesV1)
	var protocolErr *llmprotocol.ProtocolError
	if !errors.As(err, &protocolErr) || protocolErr.Code != "stream_reasoning_item_shape" {
		t.Fatalf("got %v, want stream_reasoning_item_shape", err)
	}
}

// A reasoning item's held start still reaches the client when the stream is
// cut before the item shows anything.
func TestResponsesReasoningItemStartReachesAClientWhenCut(t *testing.T) {
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	added := strings.Index(body, `data: {"type":"response.content_part.added"`)
	if added < 0 {
		t.Fatal("the fixture no longer opens a reasoning_text part")
	}
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "anthropic/claude-sonnet-5.5", ProviderModel: "anthropic/claude-sonnet-5.5",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(body[:added]))
	if err != nil {
		t.Fatalf("push: %v", err)
	}
	if bytes.Contains(bytes.Join(frames, nil), []byte("response.output_item.added")) {
		t.Fatalf("the reasoning item's start went out before the item showed what it is:\n%s", bytes.Join(frames, nil))
	}
	final, _, _, err := stream.Finalize(errors.New("upstream closed"))
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	var types []string
	for _, event := range responsesStreamEvents(t, final) {
		types = append(types, event.Type)
	}
	if len(types) != 2 || types[0] != "response.output_item.added" || types[1] != "error" {
		t.Fatalf("final events = %v, want the item's start and then the error", types)
	}
}

// A signature proves one thinking block. A signed reasoning item with two
// reasoning_text parts is refused: its parts have already gone out as separate
// contents, and signing the first alone would replay thinking the provider
// rejects. The same item unsigned streams, as reasoning text with several parts
// has nothing to prove.
func TestResponsesSignedReasoningItemHasOneTextPart(t *testing.T) {
	body := string(loadProviderFixture(t, openRouterClaudeReasoningStream))
	const second = "Then I report the result."
	events := strings.SplitAfter(body, "\n\n")
	var mutated []string
	for _, event := range events {
		mutated = append(mutated, event)
		if strings.Contains(event, `"type":"response.content_part.done"`) && strings.Contains(event, `"content_index":0`) {
			mutated = append(mutated,
				"event: response.content_part.added\ndata: "+`{"type":"response.content_part.added","output_index":0,"item_id":"rs_fixture_1","content_index":1,"part":{"type":"reasoning_text","text":""},"sequence_number":0}`+"\n\n",
				"event: response.reasoning_text.delta\ndata: "+`{"type":"response.reasoning_text.delta","output_index":0,"item_id":"rs_fixture_1","content_index":1,"delta":"`+second+`","sequence_number":0}`+"\n\n",
				"event: response.reasoning_text.done\ndata: "+`{"type":"response.reasoning_text.done","output_index":0,"item_id":"rs_fixture_1","content_index":1,"text":"`+second+`","sequence_number":0}`+"\n\n",
				"event: response.content_part.done\ndata: "+`{"type":"response.content_part.done","output_index":0,"item_id":"rs_fixture_1","content_index":1,"part":{"type":"reasoning_text","text":"`+second+`"},"sequence_number":0}`+"\n\n",
			)
		}
	}
	if len(mutated) != len(events)+4 {
		t.Fatal("the fixture no longer closes its reasoning_text part with content_part.done")
	}
	sequence := 0
	twoParts := responsesSequenceNumber.ReplaceAllStringFunc(strings.Join(mutated, ""), func(string) string {
		field := `"sequence_number":` + strconv.Itoa(sequence)
		sequence++
		return field
	})
	onePart := `"content":[{"type":"reasoning_text","text":"` + openRouterClaudeReasoningText + `"}]`
	if strings.Count(twoParts, onePart) != 2 {
		t.Fatal("the fixture's reasoning item is no longer on its done event and the completed output")
	}
	twoParts = strings.ReplaceAll(twoParts, onePart,
		`"content":[{"type":"reasoning_text","text":"`+openRouterClaudeReasoningText+`"},{"type":"reasoning_text","text":"`+second+`"}]`)

	_, err := pushResponsesStream([]byte(twoParts), llmprotocol.OpenAIResponsesV1)
	var protocolErr *llmprotocol.ProtocolError
	if !errors.As(err, &protocolErr) || protocolErr.Code != "stream_reasoning_item_shape" {
		t.Fatalf("signed: got %v, want stream_reasoning_item_shape", err)
	}

	tag := `,"signature":"` + openRouterClaudeReasoningSignature + `","format":"anthropic-claude-v1"`
	unsigned := strings.ReplaceAll(twoParts, tag, "")
	if unsigned == twoParts {
		t.Fatal("the fixture's reasoning item is no longer signed")
	}
	routed, err := pushResponsesStream([]byte(unsigned), llmprotocol.AnthropicMessagesV1)
	if err != nil {
		t.Fatalf("unsigned: %v", err)
	}
	var thinking []string
	for _, event := range anthropicStreamEvents(t, routed) {
		if event.Type == "content_block_start" && event.ContentBlock != nil && event.ContentBlock.Type == "thinking" {
			thinking = append(thinking, event.ContentBlock.Type)
		}
	}
	if len(thinking) != 2 {
		t.Fatalf("unsigned: %d thinking blocks, want one per part:\n%s", len(thinking), routed)
	}
}

// A keepalive between a reasoning item's held start and its first content
// reaches the client at once, as every keepalive does (stream_keepalive_test);
// the held start follows with the content, and the content frames are those of
// the stream without the keepalive. Holding the keepalive would delay it for
// nothing, and releasing the start on it would make a redacted item's done
// event find no start to become the carried block.
func TestKeepaliveBetweenAHeldReasoningStartAndItsContentGoesOutAtOnce(t *testing.T) {
	frames := strings.SplitAfter(string(loadProviderFixture(t, openRouterClaudeReasoningStream)), "\n\n")
	added := -1
	for index, frame := range frames {
		if strings.Contains(frame, `"type":"response.output_item.added"`) && strings.Contains(frame, `"type":"reasoning"`) {
			added = index
		}
	}
	if added < 0 {
		t.Fatal("the fixture no longer starts a reasoning item")
	}
	withKeepalive := append(append(append([]string{}, frames[:added+1]...), openRouterCommentFrame), frames[added+1:]...)
	for _, client := range builtinFormats {
		t.Run(string(client), func(t *testing.T) {
			plain := runKeepaliveStream(t, llmprotocol.OpenAIResponsesV1, client, frames)
			kept := runKeepaliveStream(t, llmprotocol.OpenAIResponsesV1, client, withKeepalive)
			if len(plain.pushed[added]) != 0 {
				t.Fatalf("the reasoning item's start went out before the item showed what it is:\n%q", plain.pushed[added])
			}
			if pushed := kept.pushed[added+1]; len(pushed) != 1 || !isTargetKeepalive(pushed[0]) {
				t.Fatalf("the keepalive became %q, want one keepalive frame at once", pushed)
			}
			content := append(append([][][]byte{}, kept.pushed[:added+1]...), kept.pushed[added+2:]...)
			if !reflect.DeepEqual(content, plain.pushed) {
				t.Fatalf("content frames changed around the keepalive:\n got %q\nwant %q", content, plain.pushed)
			}
			if !reflect.DeepEqual(kept.final, plain.final) {
				t.Fatalf("terminal frames changed:\n got %q\nwant %q", kept.final, plain.final)
			}
		})
	}
}
