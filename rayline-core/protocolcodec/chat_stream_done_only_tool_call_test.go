package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// responsesToolCallStream is a Responses provider stream of one function call
// to exec_command. With deltas, its arguments arrive as one
// function_call_arguments.delta before the done event; without, only in the
// done event, as some local servers (LM Studio) stream them.
func responsesToolCallStream(deltas bool) string {
	frame := func(event, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	frames := []string{
		frame("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":100,"model":"local-model","status":"in_progress","output":[]}}`),
		frame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"exec_command"}}`),
	}
	sequence := 2
	if deltas {
		frames = append(frames, frame("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"fc_1","output_index":0,"delta":"{\"cmd\":\"cat notes.txt\"}"}`))
		sequence = 3
	}
	number := strconv.Itoa
	frames = append(frames,
		frame("response.function_call_arguments.done", `{"type":"response.function_call_arguments.done","sequence_number":`+number(sequence)+`,"item_id":"fc_1","output_index":0,"name":"exec_command","arguments":"{\"cmd\":\"cat notes.txt\"}"}`),
		frame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":`+number(sequence+1)+`,"output_index":0,"item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\"cmd\":\"cat notes.txt\"}","call_id":"call_1","name":"exec_command"}}`),
		frame("response.completed", `{"type":"response.completed","sequence_number":`+number(sequence+2)+`,"response":{"id":"resp_1","object":"response","created_at":100,"model":"local-model","status":"completed","output":[{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\"cmd\":\"cat notes.txt\"}","call_id":"call_1","name":"exec_command"}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`),
	)
	return strings.Join(frames, "")
}

type chatStreamedCall struct {
	id, name, arguments string
}

// chatStreamToolCalls streams body to a Chat client and returns its tool
// calls as a Chat client assembles them (by index, arguments concatenated)
// and the finish reason.
func chatStreamToolCalls(t *testing.T, body string) (map[int]*chatStreamedCall, string) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "local-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, pushErr := stream.Push([]byte(body))
	final, _, _, finalErr := stream.Finalize(pushErr)
	if pushErr != nil || finalErr != nil {
		t.Fatalf("stream refused: %v, %v", pushErr, finalErr)
	}
	calls := map[int]*chatStreamedCall{}
	finish := ""
	for _, line := range bytes.Split(bytes.Join(append(frames, final...), nil), []byte("\n")) {
		data, ok := bytes.CutPrefix(line, []byte("data: "))
		if !ok || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(data, &chunk); err != nil {
			t.Fatalf("chunk is not JSON: %v\n%s", err, data)
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason != nil {
				finish = *choice.FinishReason
			}
			for _, call := range choice.Delta.ToolCalls {
				assembled := calls[call.Index]
				if assembled == nil {
					assembled = &chatStreamedCall{}
					calls[call.Index] = assembled
				}
				if call.ID != "" {
					assembled.id = call.ID
				}
				if call.Function.Name != "" {
					assembled.name = call.Function.Name
				}
				assembled.arguments += call.Function.Arguments
			}
		}
	}
	return calls, finish
}

// A Chat client is sent tool calls from argument deltas. A call whose
// arguments arrive only in the done event reaches it as one tool_calls entry
// with its id, name and arguments, as a call streamed as deltas does, and
// with the arguments once.
func TestChatClientGetsAToolCallStreamedWithoutArgumentDeltas(t *testing.T) {
	for name, deltas := range map[string]bool{"done event only": false, "argument deltas": true} {
		t.Run(name, func(t *testing.T) {
			calls, finish := chatStreamToolCalls(t, responsesToolCallStream(deltas))
			call := calls[0]
			if len(calls) != 1 || call == nil || call.id != "call_1" || call.name != "exec_command" || call.arguments != `{"cmd":"cat notes.txt"}` {
				t.Fatalf("tool calls = %+v, want one exec_command call_1 with its arguments once", calls)
			}
			if finish != "tool_calls" {
				t.Fatalf("finish_reason = %q, want tool_calls", finish)
			}
		})
	}
}

// A call a Chat client was sent is output the client holds, so a turn cut
// after it is not treated as one that produced nothing, whether its
// arguments came as deltas or only in the done event.
func TestDoneOnlyToolCallCountsAsPublishedOutput(t *testing.T) {
	for name, deltas := range map[string]bool{"done event only": false, "argument deltas": true} {
		t.Run(name, func(t *testing.T) {
			body, _, found := strings.Cut(responsesToolCallStream(deltas), "event: response.completed")
			if !found {
				t.Fatal("the stream has no response.completed event")
			}
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public-model", ProviderModel: "local-model",
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, _, _, err := stream.Push([]byte(body)); err != nil {
				t.Fatal(err)
			}
			if !stream.OutputPublished() {
				t.Fatal("the call the Chat client was sent did not count as published output")
			}
		})
	}
}
