package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// doneOnlyOrStreamedCall is a Responses stream of one get_time call, opened
// with empty arguments, whose arguments arrive as one delta or, as some
// local servers (LM Studio) stream them, only in the done event.
func doneOnlyOrStreamedCall(deltas bool) string {
	const arguments = `{"city":"Paris"}`
	if deltas {
		return responsesCallStream("", []string{arguments}, arguments)
	}
	return responsesCallStream("", nil, arguments)
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
			calls, finish := chatStreamToolCalls(t, doneOnlyOrStreamedCall(deltas))
			call := calls[0]
			if len(calls) != 1 || call == nil || call.id != "call_1" || call.name != "get_time" || call.arguments != `{"city":"Paris"}` {
				t.Fatalf("tool calls = %+v, want one get_time call_1 with its arguments once", calls)
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
			body, _, found := strings.Cut(doneOnlyOrStreamedCall(deltas), "event: response.completed")
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
