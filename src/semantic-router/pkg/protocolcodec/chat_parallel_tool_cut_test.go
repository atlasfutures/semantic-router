package protocolcodec

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const chatWholeArguments = `{"command":"ls"}`

// chatParallelToolStream is a Chat stream of two parallel tool calls, each
// streaming its arguments in turn, ended by a length finish.
func chatParallelToolStream(first, second string) string {
	chunk := func(delta string, finish string) string {
		finishMember := "null"
		if finish != "" {
			finishMember = strconv.Quote(finish)
		}
		return `data: {"id":"gen-fixture","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,` +
			`"delta":` + delta + `,"finish_reason":` + finishMember + `}]}` + "\n\n"
	}
	call := func(index int, id, arguments string) string {
		return `{"tool_calls":[{"index":` + strconv.Itoa(index) + `,"id":"` + id + `","type":"function",` +
			`"function":{"name":"bash","arguments":` + strconv.Quote(arguments) + `}}]}`
	}
	return chunk(`{"role":"assistant","content":null}`, "") +
		chunk(call(0, "call_0", first), "") +
		chunk(call(1, "call_1", second), "") +
		chunk(`{}`, "length") +
		"data: [DONE]\n\n"
}

func translateChatStream(t *testing.T, target llmprotocol.WireFormat, body string) ([]goldenStreamFrame, error) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	wire, _, _, firstErr := stream.Push([]byte(body))
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

// A cut call is the last item to complete. A Chat stream completes its
// parallel calls in index order at the finish chunk, so a truncated call 0
// followed by a whole call 1 is malformed, not a length stop.
func TestAChatParallelCutThatIsNotLastFails(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
	} {
		_, err := translateChatStream(t, target, chatParallelToolStream(chatBufferedCutArguments, chatWholeArguments))
		requireProtocolErrorCode(t, err, "invalid_stream_tool_arguments")
	}
}

// A whole call 0 followed by a truncated call 1 is a length stop: call 0 is
// complete and only call 1 is cut.
func TestAChatParallelCutThatIsLastIsALengthStop(t *testing.T) {
	frames, err := translateChatStream(t, llmprotocol.OpenAIResponsesV1,
		chatParallelToolStream(chatWholeArguments, chatBufferedCutArguments))
	if err != nil {
		t.Fatal(err)
	}
	statuses := map[string]any{}
	terminal := ""
	for _, frame := range frames {
		data, _ := frame.Data.(map[string]any)
		if item, _ := data["item"].(map[string]any); frame.Event == "response.output_item.done" {
			statuses[item["call_id"].(string)] = item["status"]
		}
		if frame.Event == "response.incomplete" || frame.Event == "response.completed" {
			terminal = frame.Event
		}
	}
	if statuses["call_0"] != "completed" || statuses["call_1"] != "incomplete" || terminal != "response.incomplete" {
		t.Fatalf("item statuses = %v, terminal = %q", statuses, terminal)
	}
}

// The buffered Chat path keeps the same rule (markChatCutToolCall): only the
// last call may be cut.
func TestABufferedChatParallelCutFollowsTheLastCallRule(t *testing.T) {
	body := func(first, second string) []byte {
		return []byte(`{"id":"gen-fixture","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":null,"tool_calls":[` +
			`{"id":"call_0","type":"function","function":{"name":"bash","arguments":` + strconv.Quote(first) + `}},` +
			`{"id":"call_1","type":"function","function":{"name":"bash","arguments":` + strconv.Quote(second) + `}}]},` +
			`"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}
	engine := NewBuiltinEngine()
	_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, body(chatBufferedCutArguments, chatWholeArguments))
	requireProtocolErrorCode(t, err, "invalid_tool_call")

	response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, body(chatWholeArguments, chatBufferedCutArguments))
	if err != nil {
		t.Fatal(err)
	}
	contents := response.Output[0].Content
	first, last := contents[len(contents)-2].ToolCall, contents[len(contents)-1].ToolCall
	if response.StopReason != llmprotocol.StopMaxTokens || first.Incomplete || !last.Incomplete {
		t.Fatalf("stop = %s, call_0 incomplete = %v, call_1 incomplete = %v", response.StopReason, first.Incomplete, last.Incomplete)
	}
}
