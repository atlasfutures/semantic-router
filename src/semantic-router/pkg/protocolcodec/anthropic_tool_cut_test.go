package protocolcodec

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// emptyToolBlockStream is the recorded Messages stream's shape with the
// tool_use block stopped before any input_json_delta, under the given stop.
func emptyToolBlockStream(t *testing.T, stop string) string {
	t.Helper()
	all := strings.Join(loadMaxTokensToolCutChunks(t, maxTokensToolCutFixture), "")
	var kept []string
	for _, frame := range strings.SplitAfter(all, "\n\n") {
		if strings.Contains(frame, "input_json_delta") {
			continue
		}
		kept = append(kept, frame)
	}
	body := strings.Join(kept, "")
	if stop != "max_tokens" {
		body = strings.Replace(body, `"stop_reason":"max_tokens"`, `"stop_reason":"`+stop+`"`, 1)
	}
	return body
}

func finalResponsesCall(t *testing.T, frames []goldenStreamFrame) (map[string]any, string) {
	t.Helper()
	var call map[string]any
	terminal := ""
	for _, frame := range frames {
		data, _ := frame.Data.(map[string]any)
		if item, _ := data["item"].(map[string]any); frame.Event == "response.output_item.done" && item["type"] == "function_call" {
			call = item
		}
		if frame.Event == "response.incomplete" || frame.Event == "response.completed" {
			terminal = frame.Event
		}
	}
	return call, terminal
}

// A tool_use block stopped before any argument delta, then a max_tokens
// stop, is the reply's last block under max_tokens: an incomplete call, as a
// buffered reply of that shape is, with its {} placeholder as arguments.
func TestAnEmptyFinalToolBlockUnderMaxTokensIsCut(t *testing.T) {
	body := emptyToolBlockStream(t, "max_tokens")
	frames, err := translateCut(t, llmprotocol.OpenAIResponsesV1, []string{body})
	if err != nil {
		t.Fatal(err)
	}
	call, terminal := finalResponsesCall(t, frames)
	if call["status"] != "incomplete" || call["arguments"] != "{}" || terminal != "response.incomplete" {
		t.Fatalf("call = %v, terminal = %q", call, terminal)
	}
	if _, err := translateCut(t, llmprotocol.AnthropicMessagesV1, []string{body}); err != nil {
		t.Fatal(err)
	}
	response, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.AnthropicMessagesV1, []byte(body),
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	if got := response.Output[len(response.Output)-1].Content[0].ToolCall; got == nil || !got.Incomplete || got.Arguments != "{}" {
		t.Fatalf("streamed reconstruction = %+v", got)
	}
}

// The control: a zero-argument call is valid, so the same block under any
// other stop is a whole call with {}.
func TestAnEmptyToolBlockUnderAnotherStopIsWhole(t *testing.T) {
	for _, stop := range []string{"tool_use", "end_turn"} {
		frames, err := translateCut(t, llmprotocol.OpenAIResponsesV1, []string{emptyToolBlockStream(t, stop)})
		if err != nil {
			t.Fatal(err)
		}
		call, terminal := finalResponsesCall(t, frames)
		if call["status"] != "completed" || call["arguments"] != "{}" || terminal != "response.completed" {
			t.Fatalf("%s: call = %v, terminal = %q", stop, call, terminal)
		}
	}
}

// A whole tool call that is the last block under max_tokens is incomplete
// too, streamed as buffered: Anthropic's documented test is the stop and the
// last block's type, not the shape of its input.
func TestAWholeFinalToolBlockUnderMaxTokensIsCutStreamedAsBuffered(t *testing.T) {
	stream := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_fixture\",\"name\":\"bash\",\"input\":{}}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\": \\\"ls\\\"}\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":9}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	frames, err := translateCut(t, llmprotocol.OpenAIResponsesV1, []string{stream})
	if err != nil {
		t.Fatal(err)
	}
	if call, _ := finalResponsesCall(t, frames); call["status"] != "incomplete" {
		t.Fatalf("call = %v", call)
	}
	buffered := `{"id":"msg_fixture","type":"message","role":"assistant","model":"m","content":[{"type":"tool_use","id":"toolu_fixture","name":"bash","input":{"command":"ls"}}],"stop_reason":"max_tokens","stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":9}}`
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(buffered))
	if err != nil {
		t.Fatal(err)
	}
	if call := response.Output[0].Content[0].ToolCall; call == nil || !call.Incomplete {
		t.Fatalf("buffered call = %+v", call)
	}
}
