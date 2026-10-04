package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The recorded Anthropic stream: a bash tool_use block whose input_json_delta
// partials end mid-JSON, then content_block_stop, message_delta with
// stop_reason max_tokens, and message_stop.
const maxTokensToolCutFixture = "stream/040-anthropic-max-tokens-mid-tool-in.json"

// The cut arguments, as the model wrote them.
const maxTokensToolCutArguments = `{"command": "cat > /tmp/list.txt <<'EOF'\napples\nbananas\noranges\ngrapes\nstrawberries\nblueberries\nraspber`

func loadMaxTokensToolCutChunks(t *testing.T, path string) []string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "golden", path))
	if err != nil {
		t.Fatal(err)
	}
	var input goldenStreamInput
	if err := json.Unmarshal(body, &input); err != nil {
		t.Fatal(err)
	}
	return input.Chunks
}

// translateCut runs Anthropic chunks to a client format and returns the
// client's SSE frames, decoded, and the first error.
func translateCut(t *testing.T, target llmprotocol.WireFormat, chunks []string) ([]goldenStreamFrame, error) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	var wire [][]byte
	var firstErr error
	for _, chunk := range chunks {
		frames, _, _, pushErr := stream.Push([]byte(chunk))
		wire = append(wire, frames...)
		if pushErr != nil && firstErr == nil {
			firstErr = pushErr
		}
	}
	frames, _, _, finalErr := stream.Finalize(firstErr)
	wire = append(wire, frames...)
	if firstErr == nil {
		firstErr = finalErr
	}
	var transcript goldenStreamTranscript
	if err := json.Unmarshal(marshalGoldenStreamTranscript(t, wire), &transcript); err != nil {
		t.Fatal(err)
	}
	return transcript.Frames, firstErr
}

func frameJSON(t *testing.T, frame goldenStreamFrame) string {
	t.Helper()
	body, err := json.Marshal(frame.Data)
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}

func TestAToolCallCutAtMaxTokensStreamsAsALengthStop(t *testing.T) {
	chunks := loadMaxTokensToolCutChunks(t, maxTokensToolCutFixture)
	t.Run("anthropic", func(t *testing.T) {
		frames, err := translateCut(t, llmprotocol.AnthropicMessagesV1, chunks)
		if err != nil {
			t.Fatal(err)
		}
		// Anthropic's own sequence: the partial input already went out, then
		// content_block_stop, message_delta max_tokens with usage, message_stop.
		tail := frames[len(frames)-3:]
		if tail[0].Event != "content_block_stop" || tail[1].Event != "message_delta" || tail[2].Event != "message_stop" {
			t.Fatalf("stream ends %s, %s, %s", tail[0].Event, tail[1].Event, tail[2].Event)
		}
		delta := frameJSON(t, tail[1])
		if !strings.Contains(delta, `"stop_reason":"max_tokens"`) || !strings.Contains(delta, `"output_tokens":60`) {
			t.Fatalf("message_delta = %s", delta)
		}
	})
	t.Run("chat", func(t *testing.T) {
		frames, err := translateCut(t, llmprotocol.OpenAIChatV1, chunks)
		if err != nil {
			t.Fatal(err)
		}
		var arguments strings.Builder
		finish := ""
		for _, frame := range frames {
			var chunk chatChunkWire
			if json.Unmarshal([]byte(frameJSON(t, frame)), &chunk) != nil {
				continue
			}
			for _, choice := range chunk.Choices {
				for _, call := range choice.Delta.ToolCalls {
					arguments.WriteString(call.Function.Arguments)
				}
				if choice.FinishReason != nil {
					finish = *choice.FinishReason
				}
			}
		}
		if finish != "length" || arguments.String() != maxTokensToolCutArguments {
			t.Fatalf("finish_reason = %q, arguments = %q", finish, arguments.String())
		}
	})
	t.Run("responses", func(t *testing.T) {
		frames, err := translateCut(t, llmprotocol.OpenAIResponsesV1, chunks)
		if err != nil {
			t.Fatal(err)
		}
		requireResponsesCutShape(t, frames)
	})
}

func requireResponsesCutShape(t *testing.T, frames []goldenStreamFrame) {
	t.Helper()
	var itemDone, terminal map[string]any
	for _, frame := range frames {
		data, _ := frame.Data.(map[string]any)
		switch frame.Event {
		case "response.output_item.done":
			itemDone, _ = data["item"].(map[string]any)
		case "response.incomplete", "response.completed":
			terminal, _ = data["response"].(map[string]any)
			terminal["event"] = frame.Event
		}
	}
	if itemDone["status"] != "incomplete" || itemDone["arguments"] != maxTokensToolCutArguments {
		t.Fatalf("function_call item = %v", itemDone)
	}
	details, _ := terminal["incomplete_details"].(map[string]any)
	if terminal["event"] != "response.incomplete" || terminal["status"] != "incomplete" || details["reason"] != "max_output_tokens" {
		t.Fatalf("terminal = %v", terminal)
	}
	output, _ := terminal["output"].([]any)
	if len(output) != 1 || output[0].(map[string]any)["status"] != "incomplete" {
		t.Fatalf("terminal output = %v", output)
	}
}

// The non-streaming client gets the same turn in its buffered form.
func TestAToolCallCutAtMaxTokensBuffersAsALengthStop(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(strings.Join(loadMaxTokensToolCutChunks(t, maxTokensToolCutFixture), ""))
	response, _, err := engine.DecodeResponseStream(llmprotocol.AnthropicMessagesV1, body, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	call := response.Output[len(response.Output)-1].Content[0].ToolCall
	if response.StopReason != llmprotocol.StopMaxTokens || call == nil || !call.Incomplete || call.Arguments != maxTokensToolCutArguments {
		t.Fatalf("reconstructed response = %+v, call = %+v", response, call)
	}
	requireBufferedCutShapes(t, engine, response)
}

// A non-streaming Messages response cut mid tool call, as Anthropic returns
// it (recorded live; ids replaced): the tool_use block is kept with input {}.
const anthropicBufferedToolCut = `{"id":"msg_fixture","type":"message","role":"assistant","container":null,"content":[{"type":"tool_use","id":"toolu_fixture","caller":{"type":"direct"},"name":"bash","input":{}}],"model":"anthropic/claude-opus-5","stop_reason":"max_tokens","stop_details":null,"stop_sequence":null,"usage":{"input_tokens":556,"output_tokens":120,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}`

func TestABufferedMessagesToolCutReadsAsIncomplete(t *testing.T) {
	engine := NewBuiltinEngine()
	response, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(anthropicBufferedToolCut))
	if err != nil {
		t.Fatal(err)
	}
	call := response.Output[0].Content[0].ToolCall
	if response.StopReason != llmprotocol.StopMaxTokens || call == nil || !call.Incomplete {
		t.Fatalf("decoded response = %+v, call = %+v", response, call)
	}
	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, response, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(encoded.Body, []byte(`"status":"incomplete"`)) ||
		!bytes.Contains(encoded.Body, []byte(`"reason":"max_output_tokens"`)) {
		t.Fatalf("Responses body = %s", encoded.Body)
	}
	// A Responses function_call that ends incomplete reads back as one.
	decoded, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, encoded.Body)
	if err != nil {
		t.Fatal(err)
	}
	if call := decoded.Output[len(decoded.Output)-1].Content[0].ToolCall; call == nil || !call.Incomplete {
		t.Fatalf("Responses round trip lost the incomplete mark: %+v", decoded.Output)
	}
}

// Only a truncated object is held to the terminal. A whole value that is not
// an object is no prefix of a call and fails at the item's completion.
func TestAWholeNonObjectArgumentFailsAtCompletion(t *testing.T) {
	depth := llmprotocol.DefaultPolicy().Limits.JSONDepth
	for _, arguments := range []string{
		`[]`, `true`, `"x"`, `{"a":1,"a":2}`,
		// No bytes could complete these, or not within the depth limit.
		`{]`, `{"x":]`, `{"a":` + strings.Repeat(`[`, depth+1),
	} {
		state := newTestStreamState()
		startTestStream(t, state)
		if _, err := state.next(llmprotocol.Event{
			Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "tool_item",
			Role: llmprotocol.RoleAssistant, ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "bash"},
		}); err != nil {
			t.Fatal(err)
		}
		_, err := state.next(llmprotocol.Event{
			Type: llmprotocol.EventOutputItemCompleted, ItemIndex: 0,
			ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "bash", Arguments: arguments},
		})
		requireProtocolErrorCode(t, err, "invalid_stream_tool_arguments")
	}
}

// A model cut off at max_tokens generates nothing more: output after a cut
// call proves the call malformed, even when the turn then stops at max_tokens.
func TestOutputAfterACutToolCallFailsItUnderMaxTokens(t *testing.T) {
	all := strings.Join(loadMaxTokensToolCutChunks(t, maxTokensToolCutFixture), "")
	terminalAt := strings.Index(all, "event: message_delta")
	body := all[:terminalAt] +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"more\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" + all[terminalAt:]
	_, err := translateCut(t, llmprotocol.OpenAIResponsesV1, []string{body})
	requireProtocolErrorCode(t, err, "invalid_stream_tool_arguments")
}

// Without OpenRouter's native length reason, the recorded Chat cut reads as
// a tool_calls stop over truncated arguments and fails as such: the error
// that refuses the terminal is the one reported.
func TestAChatToolCutWithoutANativeLengthReasonFails(t *testing.T) {
	chunks := strings.ReplaceAll(
		strings.Join(loadMaxTokensToolCutChunks(t, "stream/041-chat-max-tokens-mid-tool-in.json"), ""),
		`"native_finish_reason":"max_output_tokens"`, `"native_finish_reason":null`,
	)
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
	} {
		stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
			Context: context.Background(), PublicModel: "public-model",
		})
		if err != nil {
			t.Fatal(err)
		}
		_, _, _, pushErr := stream.Push([]byte(chunks))
		requireProtocolErrorCode(t, pushErr, "invalid_stream_tool_arguments")
	}
}

func requireBufferedCutShapes(t *testing.T, engine *Engine, response llmprotocol.Response) {
	t.Helper()
	t.Run("anthropic", func(t *testing.T) {
		encoded, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, response, llmprotocol.Envelope{})
		if err != nil {
			t.Fatal(err)
		}
		var wire struct {
			StopReason string `json:"stop_reason"`
			Content    []struct {
				Type  string          `json:"type"`
				Input json.RawMessage `json:"input"`
			} `json:"content"`
		}
		if err := json.Unmarshal(encoded.Body, &wire); err != nil {
			t.Fatal(err)
		}
		last := wire.Content[len(wire.Content)-1]
		if wire.StopReason != "max_tokens" || last.Type != "tool_use" || string(last.Input) != "{}" {
			t.Fatalf("Messages body = %s", encoded.Body)
		}
	})
	t.Run("chat", func(t *testing.T) {
		encoded, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, llmprotocol.Envelope{})
		if err != nil {
			t.Fatal(err)
		}
		var wire chatResponseWire
		if err := json.Unmarshal(encoded.Body, &wire); err != nil {
			t.Fatal(err)
		}
		choice := wire.Choices[0]
		if choice.FinishReason == nil || *choice.FinishReason != "length" || len(choice.Message.ToolCalls) != 1 ||
			choice.Message.ToolCalls[0].Function.Arguments != maxTokensToolCutArguments {
			t.Fatalf("Chat body = %s", encoded.Body)
		}
	})
	t.Run("responses", func(t *testing.T) {
		encoded, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, response, llmprotocol.Envelope{})
		if err != nil {
			t.Fatal(err)
		}
		var resource map[string]any
		if err := json.Unmarshal(encoded.Body, &resource); err != nil {
			t.Fatal(err)
		}
		requireResponsesCutShape(t, []goldenStreamFrame{
			{Event: "response.output_item.done", Data: map[string]any{"item": lastOutput(resource)}},
			{Event: "response.incomplete", Data: map[string]any{"response": resource}},
		})
	})
}

func lastOutput(resource map[string]any) any {
	output, _ := resource["output"].([]any)
	if len(output) == 0 {
		return nil
	}
	return output[len(output)-1]
}

// The same truncated arguments under any stop but max_tokens, or with
// output after them, are malformed and fail as before.
func TestTruncatedToolArgumentsWithoutAMaxTokensStopStillFail(t *testing.T) {
	chunks := loadMaxTokensToolCutChunks(t, maxTokensToolCutFixture)
	all := strings.Join(chunks, "")
	beforeTerminal := all[:strings.Index(all, "event: message_delta")]
	afterCut := beforeTerminal + "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
	for name, body := range map[string]string{
		"end_turn":     strings.Replace(all, `"stop_reason":"max_tokens"`, `"stop_reason":"end_turn"`, 1),
		"tool_use":     strings.Replace(all, `"stop_reason":"max_tokens"`, `"stop_reason":"tool_use"`, 1),
		"output after": afterCut,
	} {
		for _, target := range []llmprotocol.WireFormat{
			llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
		} {
			t.Run(name+"/"+string(target), func(t *testing.T) {
				_, err := translateCut(t, target, []string{body})
				requireProtocolErrorCode(t, err, "invalid_stream_tool_arguments")
			})
		}
	}
}

// The recorded cut with no terminal after it (the stream simply ends, or
// ends mid-line) never said the reply finished, nor that it stopped at
// max_tokens: it is an incomplete stream, and the client is told so.
func TestATruncatedToolCallWithNoTerminalIsAnIncompleteStream(t *testing.T) {
	all := strings.Join(loadMaxTokensToolCutChunks(t, maxTokensToolCutFixture), "")
	beforeTerminal := all[:strings.Index(all, "event: message_delta")]
	for name, body := range map[string]string{
		"no terminal":   beforeTerminal,
		"stream cut at": beforeTerminal[:len(beforeTerminal)-1],
	} {
		for target, code := range map[llmprotocol.WireFormat]string{
			llmprotocol.AnthropicMessagesV1: "api_error", llmprotocol.OpenAIChatV1: "stream_incomplete", llmprotocol.OpenAIResponsesV1: "stream_incomplete",
		} {
			t.Run(name+"/"+string(target), func(t *testing.T) {
				run := runFailureCut(t, llmprotocol.AnthropicMessagesV1, target, body, nil)
				requireFailureNotCut(t, target, run)
				requireFailureFrame(t, target, run, code)
			})
		}
	}
}

// A complete tool call followed by a text block cut at max_tokens is
// untouched: the call is whole, only the turn is a length stop.
func TestACompleteToolCallBeforeAMaxTokensTextCutIsComplete(t *testing.T) {
	stream := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_fixture\",\"name\":\"bash\",\"input\":{}}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\": \\\"ls\\\"}\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"and then I wi\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	frames, err := translateCut(t, llmprotocol.OpenAIResponsesV1, []string{stream})
	if err != nil {
		t.Fatal(err)
	}
	for _, frame := range frames {
		data, _ := frame.Data.(map[string]any)
		if item, _ := data["item"].(map[string]any); frame.Event == "response.output_item.done" && item["type"] == "function_call" {
			if item["status"] != "completed" || item["arguments"] != `{"command": "ls"}` {
				t.Fatalf("complete call = %v", item)
			}
			return
		}
	}
	t.Fatal("no function_call item completed")
}
