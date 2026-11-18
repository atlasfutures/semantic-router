package protocolcodec

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// streamResponsesTo streams a provider's Responses body to a client of
// format and returns the client's frames, the accumulated response and the
// first stream error.
func streamResponsesTo(t *testing.T, client llmprotocol.WireFormat, body string) ([]byte, llmprotocol.Response, error) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, client, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "local-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	accumulator := NewResponseAccumulator()
	frames, events, _, pushErr := stream.Push([]byte(body))
	if applyErr := accumulator.Apply(events); applyErr != nil {
		t.Fatal(applyErr)
	}
	final, events, _, finalErr := stream.Finalize(pushErr)
	if applyErr := accumulator.Apply(events); applyErr != nil {
		t.Fatal(applyErr)
	}
	response, err := accumulator.Response()
	if err != nil {
		t.Fatal(err)
	}
	written := bytes.Join(append(frames, final...), nil)
	if pushErr != nil {
		return written, response, pushErr
	}
	return written, response, finalErr
}

func lmStudioToolCallStream(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile("testdata/provider/lmstudio/responses-tool-call-stream.txt")
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// withDoneName gives the stream's function_call_arguments.done event a name
// member holding raw; an empty raw leaves the event as captured, with none.
func withDoneName(t *testing.T, raw string) string {
	t.Helper()
	stream := lmStudioToolCallStream(t)
	if raw == "" {
		return stream
	}
	named := strings.Replace(stream, `"type":"response.function_call_arguments.done",`,
		`"type":"response.function_call_arguments.done","name":`+raw+`,`, 1)
	if named == stream {
		t.Fatal("the fixture has no function_call_arguments.done event")
	}
	return named
}

// LM Studio's function_call_arguments.done event carries no name; the
// published Responses event type has none either, and the item's added event
// names the call. Requiring it cut every tool-calling turn from such a server
// after the model had finished, so Codex reported "stream closed before
// response.completed" and retried. A null or empty name says no more than an
// absent one. Whichever the event sends, every client gets the call under the
// name its added event gave it. The fixture is an LM Studio stream as
// captured (one reasoning item and one shell call), texts shortened and ids
// replaced.
func TestResponsesToolDoneWithoutANameCompletes(t *testing.T) {
	for label, name := range map[string]string{"absent": "", "null": `null`, "empty": `""`} {
		for _, client := range []llmprotocol.WireFormat{
			llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		} {
			t.Run(label+"/"+string(client), func(t *testing.T) {
				frames, response, err := streamResponsesTo(t, client, withDoneName(t, name))
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
				if bytes.Contains(frames, []byte(`"name":""`)) || bytes.Contains(frames, []byte(`"name":null`)) {
					t.Fatalf("the client was sent an empty tool name:\n%s", frames)
				}
				if !bytes.Contains(frames, []byte(`"name":"exec_command"`)) {
					t.Fatalf("the client was not sent the call's own name:\n%s", frames)
				}
			})
		}
	}
}

// A done event that sends another tool's name is refused, and so is a name
// that is not a string.
func TestResponsesToolDoneWithAnotherNameIsRefused(t *testing.T) {
	for raw, want := range map[string]string{
		`"other_tool"`: "stream_tool_identity_mismatch",
		`7`:            "",
	} {
		t.Run(raw, func(t *testing.T) {
			if _, _, err := streamResponsesTo(t, llmprotocol.OpenAIResponsesV1, withDoneName(t, raw)); err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("stream error = %v, want %q", err, want)
			}
		})
	}
}
