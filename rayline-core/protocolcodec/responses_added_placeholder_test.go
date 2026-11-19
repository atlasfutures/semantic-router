package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// A Responses client that accumulates a call as it streams, as the OpenAI
// SDK's stream helper does, takes the added item's arguments and appends
// every argument delta. A Claude tool_use opens with input {} and then
// streams its input as input_json_delta, so the added item must not carry
// that "{}" or the client assembles it in front of the arguments.
func TestResponsesClientAssemblesClaudeToolArgumentsOnce(t *testing.T) {
	frame := func(event, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	stream := func(deltas ...string) string {
		frames := []string{
			frame("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`),
			frame("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`),
		}
		for _, delta := range deltas {
			encoded, _ := json.Marshal(delta)
			frames = append(frames, frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":`+string(encoded)+`}}`))
		}
		frames = append(frames,
			frame("content_block_stop", `{"type":"content_block_stop","index":0}`),
			frame("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}`),
			frame("message_stop", `{"type":"message_stop"}`),
		)
		return strings.Join(frames, "")
	}
	cases := map[string]struct {
		body string
		want string
	}{
		"zero arguments":     {stream("{}"), "{}"},
		"streamed arguments": {stream(`{"city":`, `"Paris"}`), `{"city":"Paris"}`},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			translated, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "public-model", ProviderModel: "claude-model",
			})
			if err != nil {
				t.Fatal(err)
			}
			frames, _, _, pushErr := translated.Push([]byte(test.body))
			final, _, _, finalErr := translated.Finalize(pushErr)
			if pushErr != nil || finalErr != nil {
				t.Fatalf("stream refused: %v, %v", pushErr, finalErr)
			}
			var assembled []string
			for _, line := range bytes.Split(bytes.Join(append(frames, final...), nil), []byte("\n")) {
				data, ok := bytes.CutPrefix(line, []byte("data: "))
				if !ok {
					continue
				}
				var event struct {
					Type  string `json:"type"`
					Delta string `json:"delta"`
					Item  struct {
						Type      string `json:"type"`
						Arguments string `json:"arguments"`
					} `json:"item"`
				}
				if err := json.Unmarshal(data, &event); err != nil {
					continue // non-string deltas belong to other events
				}
				switch event.Type {
				case "response.output_item.added":
					if event.Item.Type == "function_call" {
						assembled = append(assembled, event.Item.Arguments)
					}
				case "response.function_call_arguments.delta":
					assembled[len(assembled)-1] += event.Delta
				}
			}
			if len(assembled) != 1 || assembled[0] != test.want {
				t.Fatalf("client assembled %q, want one call with %s", assembled, test.want)
			}
		})
	}
}
