package protocolcodec

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// streamLMStudioResponses streams a provider's Responses body to a Responses
// client and returns the frames, the push error and the accumulated response.
func streamLMStudioResponses(t *testing.T, body string) ([]byte, error, llmprotocol.Response) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "local-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	accumulator := NewResponseAccumulator()
	frames, events, _, pushErr := stream.Push([]byte(body))
	if err := accumulator.Apply(events); err != nil {
		t.Fatal(err)
	}
	final, events, _, _ := stream.Finalize(pushErr)
	if err := accumulator.Apply(events); err != nil {
		t.Fatal(err)
	}
	response, err := accumulator.Response()
	if err != nil {
		t.Fatal(err)
	}
	return bytes.Join(append(frames, final...), nil), pushErr, response
}

// LM Studio's function_call_arguments.done event carries no name; the current
// published Responses event type has none either, and the item's added event
// names the call. Requiring it cut every tool-calling turn from such a server
// after the model had finished, so Codex reported "stream closed before
// response.completed" and retried. The fixture is an LM Studio stream of one
// reasoning item and one shell call, with texts shortened and ids replaced.
func TestResponsesToolDoneWithoutANameCompletes(t *testing.T) {
	raw, err := os.ReadFile("testdata/provider/lmstudio/responses-tool-call-stream.txt")
	if err != nil {
		t.Fatal(err)
	}
	frames, pushErr, response := streamLMStudioResponses(t, string(raw))
	if pushErr != nil || response.Error != nil {
		t.Fatalf("stream refused: push %v, response %v", pushErr, response.Error)
	}
	for _, want := range []string{`"type":"response.completed"`, `"name":"exec_command"`, `{\"cmd\":\"cat notes.txt\"}`} {
		if !bytes.Contains(frames, []byte(want)) {
			t.Fatalf("client stream lacks %s:\n%s", want, frames)
		}
	}
}

// A done event that does send a name must name the same tool.
func TestResponsesToolDoneWithAnotherNameIsRefused(t *testing.T) {
	raw, err := os.ReadFile("testdata/provider/lmstudio/responses-tool-call-stream.txt")
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{
		`"other_tool"`: "stream_tool_identity_mismatch",
		`""`:           "stream_required_field",
		`null`:         "stream_required_field",
	} {
		t.Run(name, func(t *testing.T) {
			renamed := strings.Replace(string(raw), `"type":"response.function_call_arguments.done",`,
				`"type":"response.function_call_arguments.done","name":`+name+`,`, 1)
			if renamed == string(raw) {
				t.Fatal("the fixture has no function_call_arguments.done event")
			}
			_, pushErr, _ := streamLMStudioResponses(t, renamed)
			if pushErr == nil || !strings.Contains(pushErr.Error(), want) {
				t.Fatalf("push error = %v, want %s", pushErr, want)
			}
		})
	}
}
