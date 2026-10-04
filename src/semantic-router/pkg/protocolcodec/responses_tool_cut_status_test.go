package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// responsesCutStream is the Responses stream our encoder writes for the
// recorded cut, as SSE, with the function_call's status set to status in the
// item done event and in the terminal resource.
func responsesCutStream(t *testing.T, status string) (string, []byte) {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "golden", "stream", "040-anthropic-max-tokens-mid-tool-responses-out.json"))
	if err != nil {
		t.Fatal(err)
	}
	var transcript goldenStreamTranscript
	if err := json.Unmarshal(body, &transcript); err != nil {
		t.Fatal(err)
	}
	var stream bytes.Buffer
	var resource []byte
	for _, frame := range transcript.Frames {
		data, _ := frame.Data.(map[string]any)
		if item, _ := data["item"].(map[string]any); item != nil && frame.Event == "response.output_item.done" {
			item["status"] = status
		}
		if response, _ := data["response"].(map[string]any); response != nil && frame.Event == "response.incomplete" {
			for _, output := range response["output"].([]any) {
				output.(map[string]any)["status"] = status
			}
			resource, _ = json.Marshal(response)
		}
		encoded, err := json.Marshal(data)
		if err != nil {
			t.Fatal(err)
		}
		stream.WriteString("event: " + frame.Event + "\ndata: " + string(encoded) + "\n\n")
	}
	return stream.String(), resource
}

// On a Responses stream a partial function call is a cut only when the
// provider ended its item incomplete. One it calls completed is malformed,
// as the same item in a buffered response is.
func TestAResponsesCallIsCutOnlyWhenTheProviderSaysIncomplete(t *testing.T) {
	engine := NewBuiltinEngine()
	decodeStream := func(body string) error {
		stream, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
			Context: context.Background(), PublicModel: "public-model",
		})
		if err != nil {
			t.Fatal(err)
		}
		_, _, _, pushErr := stream.Push([]byte(body))
		_, _, _, finalErr := stream.Finalize(pushErr)
		if pushErr != nil {
			return pushErr
		}
		return finalErr
	}

	incomplete, incompleteResource := responsesCutStream(t, "incomplete")
	if err := decodeStream(incomplete); err != nil {
		t.Fatalf("an incomplete item was refused: %v", err)
	}
	response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, incompleteResource)
	if err != nil {
		t.Fatalf("buffered incomplete item was refused: %v", err)
	}
	if call := response.Output[len(response.Output)-1].Content[0].ToolCall; call == nil || !call.Incomplete {
		t.Fatalf("buffered call = %+v", call)
	}

	completed, completedResource := responsesCutStream(t, "completed")
	requireProtocolErrorCode(t, decodeStream(completed), "invalid_stream_tool_arguments")
	_, _, _, err = engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, completedResource)
	requireProtocolErrorCode(t, err, "invalid_tool_call")
}
