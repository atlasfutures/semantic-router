package protocolcodec

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A client that keeps no server state (store: false, as Codex runs) resends the
// previous turn's output items verbatim: each carries the id and status the
// Responses API gave it. The API accepts them as input, so routing must too.
// (The body leaves store out: an explicit store is a separate capability that
// not every target format carries.)
const responsesResentOutputBody = `{
  "model": "gpt-5",
  "input": [
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run the tests"}]},
    {"type": "reasoning", "id": "rs_1", "status": "completed",
     "summary": [{"type": "summary_text", "text": "check the suite first"}]},
    {"type": "function_call", "id": "fc_1", "status": "completed",
     "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"go test\"}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
    {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
     "content": [{"type": "output_text", "text": "the suite passes", "annotations": [], "logprobs": []}]},
    {"type": "message", "id": "msg_2", "status": "incomplete", "role": "assistant",
     "content": [{"type": "output_text", "text": "and then", "annotations": []}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "now lint"}]}
  ]
}`

func TestResponsesResentOutputItemsRoute(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			routed := routeResponsesRequest(t, responsesResentOutputBody, target)
			for _, mark := range []string{"run the tests", "call_1", "the suite passes", "and then", "now lint"} {
				if !bytes.Contains(routed, []byte(mark)) {
					t.Fatalf("%q was lost routing resent output items: %s", mark, routed)
				}
			}
			if bytes.Contains(routed, []byte(`"incomplete"`)) || bytes.Contains(routed, []byte(`"status"`)) {
				t.Fatalf("an input item's status travelled to the provider: %s", routed)
			}
		})
	}
}
