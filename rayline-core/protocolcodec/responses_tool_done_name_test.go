package protocolcodec

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// streamResponsesTo streams a provider's Responses body to a client of
// format and returns the accumulated response and the first stream error.
func streamResponsesTo(t *testing.T, client llmprotocol.WireFormat, body string) (llmprotocol.Response, error) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, client, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "local-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	accumulator := NewResponseAccumulator()
	_, events, _, pushErr := stream.Push([]byte(body))
	if applyErr := accumulator.Apply(events); applyErr != nil {
		t.Fatal(applyErr)
	}
	_, events, _, finalErr := stream.Finalize(pushErr)
	if applyErr := accumulator.Apply(events); applyErr != nil {
		t.Fatal(applyErr)
	}
	response, err := accumulator.Response()
	if err != nil {
		t.Fatal(err)
	}
	if pushErr != nil {
		return response, pushErr
	}
	return response, finalErr
}

func lmStudioToolCallStream(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile("testdata/provider/lmstudio/responses-tool-call-stream.txt")
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// LM Studio's function_call_arguments.done event carries no name; the
// published Responses event type has none either, and the item's added event
// names the call. Requiring it cut every tool-calling turn from such a server
// after the model had finished, so Codex reported "stream closed before
// response.completed" and retried. The fixture is an LM Studio stream as
// captured (one reasoning item and one shell call), texts shortened and ids
// replaced.
func TestResponsesToolDoneWithoutANameCompletes(t *testing.T) {
	for _, client := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(client), func(t *testing.T) {
			response, err := streamResponsesTo(t, client, lmStudioToolCallStream(t))
			if err != nil || response.Error != nil {
				t.Fatalf("stream refused: %v, response error %v", err, response.Error)
			}
			var calls []*llmprotocol.ToolCall
			for _, item := range response.Output {
				for _, content := range item.Content {
					if content.Kind == llmprotocol.ContentToolCall {
						calls = append(calls, content.ToolCall)
					}
				}
			}
			if len(calls) != 1 || calls[0].Name != "exec_command" || calls[0].Arguments != `{"cmd":"cat notes.txt"}` {
				t.Fatalf("tool calls = %+v, want one exec_command call with its arguments", calls)
			}
		})
	}
}

// A done event that does send a name must name the call's tool.
func TestResponsesToolDoneWithAnotherNameIsRefused(t *testing.T) {
	for name, want := range map[string]string{
		`"other_tool"`: "stream_tool_identity_mismatch",
		`""`:           "stream_required_field",
		`null`:         "stream_required_field",
	} {
		t.Run(name, func(t *testing.T) {
			raw := lmStudioToolCallStream(t)
			renamed := strings.Replace(raw, `"type":"response.function_call_arguments.done",`,
				`"type":"response.function_call_arguments.done","name":`+name+`,`, 1)
			if renamed == raw {
				t.Fatal("the fixture has no function_call_arguments.done event")
			}
			if _, err := streamResponsesTo(t, llmprotocol.OpenAIResponsesV1, renamed); err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("stream error = %v, want %s", err, want)
			}
		})
	}
}
