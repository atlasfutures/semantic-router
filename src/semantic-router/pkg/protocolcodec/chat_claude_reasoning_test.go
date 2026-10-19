package protocolcodec

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// claudeTurnBlock is one block of a Messages assistant turn as a test states it
// and reads it back: a thinking block with its signature, a redacted_thinking
// block with its data, a text block, or a tool_use block.
type claudeTurnBlock struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
	Text      string `json:"text,omitempty"`
	ID        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
}

func anthropicTurn(blocks []claudeTurnBlock) string {
	var content []string
	for _, block := range blocks {
		switch block.Type {
		case "tool_use":
			content = append(content, fmt.Sprintf(`{"type":"tool_use","id":%q,"name":%q,"input":{}}`, block.ID, block.Name))
		case "thinking":
			// Anthropic always writes the thinking member, empty under
			// display "omitted", and its signature.
			content = append(content, fmt.Sprintf(`{"type":"thinking","thinking":%q,"signature":%q}`, block.Thinking, block.Signature))
		case "redacted_thinking":
			content = append(content, fmt.Sprintf(`{"type":"redacted_thinking","data":%q}`, block.Data))
		case "text":
			content = append(content, fmt.Sprintf(`{"type":"text","text":%q}`, block.Text))
		default:
			panic("unknown block type " + block.Type)
		}
	}
	return `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5.5","content":[` + strings.Join(content, ",") + `],` +
		`"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":5}}`
}

// replayedChatRequest is the Chat request a client sends after a tool call:
// the assistant message exactly as the Chat answer carried it, then a tool
// message for each call.
func replayedChatRequest(t *testing.T, chatAnswer []byte) string {
	t.Helper()
	var answer struct {
		Choices []struct {
			Message json.RawMessage `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(chatAnswer, &answer); err != nil || len(answer.Choices) != 1 {
		t.Fatalf("chat answer = %s", chatAnswer)
	}
	var message struct {
		ToolCalls []struct {
			ID string `json:"id"`
		} `json:"tool_calls"`
	}
	if err := json.Unmarshal(answer.Choices[0].Message, &message); err != nil {
		t.Fatal(err)
	}
	messages := []string{`{"role":"user","content":"run the tests"}`, string(answer.Choices[0].Message)}
	for _, call := range message.ToolCalls {
		messages = append(messages, fmt.Sprintf(`{"role":"tool","tool_call_id":%q,"content":"ok"}`, call.ID))
	}
	if len(message.ToolCalls) == 0 {
		messages = append(messages, `{"role":"user","content":"go on"}`)
	}
	return `{"model":"auto","messages":[` + strings.Join(messages, ",") + `]}`
}

// encodedAssistantBlocks reads the assistant turn's blocks out of an encoded
// Messages request.
func encodedAssistantBlocks(t *testing.T, body []byte) []claudeTurnBlock {
	t.Helper()
	var request struct {
		Messages []struct {
			Role    string            `json:"role"`
			Content []claudeTurnBlock `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatalf("messages request = %s", body)
	}
	for _, message := range request.Messages {
		if message.Role == "assistant" {
			return message.Content
		}
	}
	t.Fatalf("no assistant turn in %s", body)
	return nil
}

// A Chat client on a Messages Claude arm replays Claude's thinking exactly
// as Claude wrote it: every signed thinking block and every redacted_thinking
// block comes back to the client as an anthropic-claude-v1 reasoning_details
// item, and a verbatim replay of that message reaches a Messages target with
// every block restored, so the unsigned row has nothing to drop
// (atlasfutures/semantic-router#278).
func TestClaudeThinkingRoundTripsThroughAChatClient(t *testing.T) {
	engine := NewBuiltinEngine()
	call := claudeTurnBlock{Type: "tool_use", ID: "toolu_1", Name: "bash"}
	for _, tc := range []struct {
		name   string
		blocks []claudeTurnBlock
	}{
		{"one signed block", []claudeTurnBlock{{Type: "thinking", Thinking: "Check the tool.", Signature: "sig-1"}, call}},
		{"two signed blocks", []claudeTurnBlock{
			{Type: "thinking", Thinking: "First.", Signature: "sig-1"},
			{Type: "thinking", Thinking: "Second.", Signature: "sig-2"}, call,
		}},
		{"redacted then signed", []claudeTurnBlock{
			{Type: "redacted_thinking", Data: "opaque-1"},
			{Type: "thinking", Thinking: "Check the tool.", Signature: "sig-1"}, call,
		}},
		{"omitted display: signed with no text", []claudeTurnBlock{{Type: "thinking", Thinking: "", Signature: "sig-1"}, call}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			answer, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(anthropicTurn(tc.blocks)), nil)
			if err != nil {
				t.Fatalf("translate response: %v", err)
			}
			if !hasSignatureDiagnostic(answer.Diagnostics, llmprotocol.DiagnosticApproximated) {
				t.Fatalf("diagnostics = %+v, want the carried signature reported", answer.Diagnostics)
			}
			request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(replayedChatRequest(t, answer.Body)))
			if err != nil {
				t.Fatalf("decode replay: %v", err)
			}
			wantBlocks := 0
			for _, block := range tc.blocks {
				if block.Type != "tool_use" {
					wantBlocks++
				}
			}
			if rebuilt := ClaudeThinkingFromReasoningDetails(&request); rebuilt != wantBlocks {
				t.Fatalf("rebuilt %d blocks, want %d", rebuilt, wantBlocks)
			}
			if carry := CarryReasoningTo(&request, llmprotocol.AnthropicMessagesV1); carry.Dropped() > 0 || carry.SignaturesStripped > 0 {
				t.Fatalf("the Messages target lost Claude's reasoning: %+v", carry)
			}
			request.Generation++
			encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
			if err != nil {
				t.Fatalf("encode for Messages: %v", err)
			}
			got := encodedAssistantBlocks(t, encoded.Body)
			var want []claudeTurnBlock
			for _, block := range tc.blocks {
				if block.Type != "tool_use" {
					want = append(want, block)
				}
			}
			if len(got) < len(want) {
				t.Fatalf("assistant turn = %+v, want it to open with %+v", got, want)
			}
			for index, block := range want {
				if got[index] != block {
					t.Fatalf("block %d = %+v, want %+v\n%s", index, got[index], block, encoded.Body)
				}
			}
			if got[len(want)].Type != "tool_use" {
				t.Fatalf("the tool call did not follow the thinking: %+v", got)
			}
		})
	}
}

// chatStreamReasoningDetails merges the reasoning_details fragments a Chat
// stream carried, by index, as an OpenRouter client does, and collects the
// reasoning_content deltas and the raw fragments in order.
func chatStreamReasoningDetails(t *testing.T, wire string) (json.RawMessage, []string, []string) {
	t.Helper()
	var merged json.RawMessage
	var reasoning, fragments []string
	for _, frame := range strings.Split(wire, "\n\n") {
		data, found := strings.CutPrefix(strings.TrimSpace(frame), "data: ")
		if !found || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Reasoning        *string         `json:"reasoning_content"`
					ReasoningDetails json.RawMessage `json:"reasoning_details"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("chunk %s: %v", data, err)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Reasoning != nil {
				reasoning = append(reasoning, *choice.Delta.Reasoning)
			}
			if len(choice.Delta.ReasoningDetails) > 0 {
				fragments = append(fragments, string(choice.Delta.ReasoningDetails))
				var err error
				if merged, err = mergeReasoningDetailsFragment(merged, choice.Delta.ReasoningDetails); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	return merged, reasoning, fragments
}

// The live stream from a Messages arm carries each thinking block's signature
// after its text and a redacted_thinking block whole. When the Router has said
// the worker is Claude, a Chat client is given OpenRouter's shape: a tagged
// text fragment beside each reasoning_content delta and the signature last,
// so a client that merges fragments by index ends with the array a buffered
// answer carries and no text is sent twice in one chunk. When the encoder
// knows only the wire format, it waits for the signature and mints the item
// whole. Either way the merged array is the same.
func TestClaudeThinkingStreamsToAChatClientAsReasoningDetails(t *testing.T) {
	events := []string{
		`event: message_start` + "\n" + `data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5.5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":1}}}`,
		`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Check "}}`,
		`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"the tool."}}`,
		`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}`,
		`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}`,
		`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque-1"}}`,
		`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":1}`,
		`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
		`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Done."}}`,
		`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":2}`,
		`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`,
		`event: message_stop` + "\n" + `data: {"type":"message_stop"}`,
	}
	want := `[{"type":"reasoning.text","text":"Check the tool.","signature":"sig-1","format":"anthropic-claude-v1","index":0},` +
		`{"type":"reasoning.encrypted","data":"opaque-1","format":"anthropic-claude-v1","index":1}]`
	engine := NewBuiltinEngine()
	for _, tc := range []struct {
		name, family string
		fragments    []string
	}{
		{"the worker is Claude: OpenRouter's fragments", "anthropic", []string{
			`[{"format":"anthropic-claude-v1","index":0,"text":"Check ","type":"reasoning.text"}]`,
			`[{"format":"anthropic-claude-v1","index":0,"text":"the tool.","type":"reasoning.text"}]`,
			`[{"format":"anthropic-claude-v1","index":0,"signature":"sig-1","type":"reasoning.text"}]`,
			`[{"data":"opaque-1","format":"anthropic-claude-v1","index":1,"type":"reasoning.encrypted"}]`,
		}},
		{"the worker is unknown: one item at the signature", "", []string{
			`[{"format":"anthropic-claude-v1","index":0,"signature":"sig-1","text":"Check the tool.","type":"reasoning.text"}]`,
			`[{"data":"opaque-1","format":"anthropic-claude-v1","index":1,"type":"reasoning.encrypted"}]`,
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := engine.NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public", ProviderModel: "claude-opus-5.5", WorkerFamily: tc.family,
			})
			if err != nil {
				t.Fatal(err)
			}
			var wire strings.Builder
			var diagnostics llmprotocol.Diagnostics
			for _, event := range events {
				frames, _, pushDiagnostics, pushErr := stream.Push([]byte(event + "\n\n"))
				if pushErr != nil {
					t.Fatalf("push: %v", pushErr)
				}
				diagnostics = append(diagnostics, pushDiagnostics...)
				for _, frame := range frames {
					wire.Write(frame)
				}
			}
			frames, _, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatalf("finalize: %v", err)
			}
			for _, frame := range frames {
				wire.Write(frame)
			}
			out := wire.String()
			if strings.Contains(out, "signature_delta") {
				t.Fatalf("a Messages signature_delta reached a Chat client: %s", out)
			}
			merged, reasoning, fragments := chatStreamReasoningDetails(t, out)
			assertEqualReasoningJSON(t, "streamed reasoning_details", merged, json.RawMessage(want))
			if fmt.Sprint(fragments) != fmt.Sprint(tc.fragments) {
				t.Fatalf("reasoning_details fragments = %v, want %v", fragments, tc.fragments)
			}
			if strings.Join(reasoning, "") != "Check the tool." {
				t.Fatalf("reasoning_content deltas = %q", reasoning)
			}
			if !hasSignatureDiagnostic(diagnostics, llmprotocol.DiagnosticApproximated) {
				t.Fatalf("diagnostics = %+v, want the carried signature reported", diagnostics)
			}
		})
	}

	// The same answer, buffered and then streamed to the client, carries the
	// same array.
	decoded, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(anthropicTurn([]claudeTurnBlock{
		{Type: "thinking", Thinking: "Check the tool.", Signature: "sig-1"},
		{Type: "redacted_thinking", Data: "opaque-1"},
		{Type: "text", Text: "Done."},
	})), nil)
	if err != nil {
		t.Fatal(err)
	}
	body, _, err := engine.EncodeResponseStream(llmprotocol.OpenAIChatV1, decoded.Response, llmprotocol.StreamContext{Context: context.Background()})
	if err != nil {
		t.Fatalf("encode stream: %v", err)
	}
	buffered, _, _ := chatStreamReasoningDetails(t, string(body))
	assertEqualReasoningJSON(t, "buffered-then-streamed reasoning_details", buffered, json.RawMessage(want))
}

// A replay the cell cannot restore whole is left for the unsigned row to drop
// whole. Anthropic accepts a turn with every thinking block removed and
// rejects one with some removed, so a partial restore is never made.
func TestPartialClaudeReplayIsDroppedWhole(t *testing.T) {
	engine := NewBuiltinEngine()
	marker := ThinkingMarker("anthropic", "Check the tool.")
	signed := `{"type":"reasoning.text","text":"Check the tool.","signature":"sig-1","format":"anthropic-claude-v1","index":0}`
	for _, tc := range []struct {
		name, reasoning, details string
		rebuilt                  int
	}{
		{"complete", `"Check the tool."`, "[" + signed + "]", 1},
		{"the array alone, as OpenRouter documents the replay", `null`, "[" + signed + "]", 1},
		{"the array alone with two blocks", `null`, "[" + signed + `,{"type":"reasoning.text","text":"More.","signature":"sig-2","format":"anthropic-claude-v1","index":1}]`, 2},
		{"an unsigned item", `"Check the tool.More."`, "[" + signed + `,{"type":"reasoning.text","text":"More.","format":"anthropic-claude-v1","index":1}]`, 0},
		{"another format", `"Check the tool."`, "[" + signed + `,{"type":"reasoning.encrypted","data":"g","format":"google-gemini-v1","index":1}]`, 0},
		{"an encrypted item with no data", `"Check the tool."`, "[" + signed + `,{"type":"reasoning.encrypted","format":"anthropic-claude-v1","index":1}]`, 0},
		{"text that is not the items'", `"Check the tool. And more."`, "[" + signed + "]", 0},
		{"a Router marker as signature", `"Check the tool."`, `[{"type":"reasoning.text","text":"Check the tool.","signature":"` + marker + `","format":"anthropic-claude-v1","index":0}]`, 0},
		{"an unknown item type", `"Check the tool."`, "[" + signed + `,{"type":"reasoning.summary","summary":"s","format":"anthropic-claude-v1","index":1}]`, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"auto","messages":[{"role":"user","content":"run the tests"},` +
				`{"role":"assistant","content":null,"reasoning_content":` + tc.reasoning + `,"reasoning_details":` + tc.details + `,` +
				`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{}"}}]},` +
				`{"role":"tool","tool_call_id":"call_1","content":"ok"}]}`
			request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(body))
			if err != nil {
				t.Fatal(err)
			}
			if rebuilt := ClaudeThinkingFromReasoningDetails(&request); rebuilt != tc.rebuilt {
				t.Fatalf("rebuilt %d, want %d", rebuilt, tc.rebuilt)
			}
			carry := CarryReasoningTo(&request, llmprotocol.AnthropicMessagesV1)
			assistant := request.Messages[1]
			var reasoning int
			for _, content := range assistant.Content {
				if content.Kind == llmprotocol.ContentReasoning || redactedThinkingBlock(content) {
					reasoning++
				}
			}
			if tc.rebuilt == 0 && (carry.UnsignedDropped != 1 || reasoning != 0) {
				t.Fatalf("a partial replay was not dropped whole: carry=%+v contents=%+v", carry, assistant.Content)
			}
			if tc.rebuilt > 0 && (carry.Dropped() != 0 || reasoning != tc.rebuilt) {
				t.Fatalf("a complete replay was dropped: carry=%+v contents=%+v", carry, assistant.Content)
			}
			if len(assistant.Content) == 0 || assistant.Content[len(assistant.Content)-1].Kind != llmprotocol.ContentToolCall {
				t.Fatalf("the tool call was lost: %+v", assistant.Content)
			}
		})
	}
}

// A signature in the Router's namespace is a marker, not Anthropic's, and is
// never minted into an item: it is cleared and reported, as every signature
// was before #278.
func TestRouterMarkerIsNeverCarriedToAChatClient(t *testing.T) {
	engine := NewBuiltinEngine()
	decoded, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(anthropicTurn([]claudeTurnBlock{
		{Type: "thinking", Thinking: "Check the tool.", Signature: "sig-1"},
		{Type: "text", Text: "Done."},
	})), nil)
	if err != nil {
		t.Fatal(err)
	}
	response := decoded.Response
	response.Output = append([]llmprotocol.OutputItem(nil), response.Output...)
	response.Output[0].Content = append([]llmprotocol.Content(nil), response.Output[0].Content...)
	response.Output[0].Content[0].Signature = ThinkingMarker("kimi", "Check the tool.")
	response.Generation++
	result, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(result.Body), "vsr.") || strings.Contains(string(result.Body), "reasoning_details") {
		t.Fatalf("a Router marker reached a Chat client: %s", result.Body)
	}
	if !hasSignatureDiagnostic(result.Diagnostics, llmprotocol.DiagnosticDropped) {
		t.Fatalf("diagnostics = %+v, want the dropped marker reported", result.Diagnostics)
	}
}

// A signature no item can carry -- on a content that already holds
// reasoning_details (a signed summary never gets this far: ValidateResponse
// refuses it) -- is cleared and reported rather than left for the capability
// gate to refuse, and the items minted for the blocks that can be carried are
// numbered after the items the choice already holds, so a client merging by
// index keeps every item (Codex P2 x2 on #278).
func TestUncarriableSignaturesAreClearedAndItemsNumberAfterExistingOnes(t *testing.T) {
	engine := NewBuiltinEngine()
	decoded, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(anthropicTurn([]claudeTurnBlock{
		{Type: "thinking", Thinking: "earlier", Signature: "sig-with-details"},
		{Type: "thinking", Thinking: "Check the tool.", Signature: "sig-1"},
		{Type: "text", Text: "Done."},
	})), nil)
	if err != nil {
		t.Fatal(err)
	}
	response := decoded.Response
	response.Output = append([]llmprotocol.OutputItem(nil), response.Output...)
	response.Output[0].Content = append([]llmprotocol.Content(nil), response.Output[0].Content...)
	// The first block already carries a reasoning_details item beside its
	// signature, as a content from a Chat upstream would.
	response.Output[0].Content[0].Extensions = reasoningDetailsFields([]byte(`[{"type":"reasoning.text","text":"earlier","format":"anthropic-claude-v1","index":0}]`))
	response.Generation++
	result, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("a Chat encode was refused for a signature no item can carry: %v", err)
	}
	body := string(result.Body)
	if strings.Contains(body, "sig-with-details") {
		t.Fatalf("an uncarriable signature reached the Chat client: %s", body)
	}
	if !strings.Contains(body, `{"format":"anthropic-claude-v1","index":1,"signature":"sig-1","text":"Check the tool.","type":"reasoning.text"}`) {
		t.Fatalf("the minted item was not numbered after the item the choice already held: %s", body)
	}
	if !hasSignatureDiagnostic(result.Diagnostics, llmprotocol.DiagnosticDropped) || !hasSignatureDiagnostic(result.Diagnostics, llmprotocol.DiagnosticApproximated) {
		t.Fatalf("diagnostics = %+v, want both the cleared and the carried signature reported", result.Diagnostics)
	}
}

// A Claude arm served over Chat already streams OpenRouter's own items, and a
// Chat-sourced block is never signed, so even when the Router says the worker
// is Claude the encoder mints nothing for a Chat-sourced stream: the upstream's
// items pass through and its text-only chunks stay reasoning_content alone,
// never a Claude-tagged item with no signature to come (Codex P2 x2 on #278).
func TestChatSourcedClaudeStreamIsNotTagged(t *testing.T) {
	chunk := func(delta string) string {
		return `data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"anthropic/claude-opus-5","choices":[{"index":0,"delta":` + delta + `,"finish_reason":null}]}` + "\n\n"
	}
	chunks := []string{
		chunk(`{"role":"assistant","content":"","reasoning":"Check "}`),
		chunk(`{"content":"","reasoning":"the tool.","reasoning_details":[{"type":"reasoning.text","text":"the tool.","format":"anthropic-claude-v1","index":0}]}`),
		chunk(`{"content":"Done."}`),
		`data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"anthropic/claude-opus-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n",
	}
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public", ProviderModel: "anthropic/claude-opus-5", WorkerFamily: "anthropic",
	})
	if err != nil {
		t.Fatal(err)
	}
	var wire strings.Builder
	for _, c := range chunks {
		frames, _, _, pushErr := stream.Push([]byte(c))
		if pushErr != nil {
			t.Fatal(pushErr)
		}
		for _, frame := range frames {
			wire.Write(frame)
		}
	}
	frames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, frame := range frames {
		wire.Write(frame)
	}
	_, reasoning, fragments := chatStreamReasoningDetails(t, wire.String())
	if strings.Join(reasoning, "") != "Check the tool." {
		t.Fatalf("reasoning_content deltas = %q", reasoning)
	}
	want := []string{`[{"type":"reasoning.text","text":"the tool.","format":"anthropic-claude-v1","index":0}]`}
	if fmt.Sprint(fragments) != fmt.Sprint(want) {
		t.Fatalf("reasoning_details fragments = %v, want only the upstream's own %v", fragments, want)
	}
}
