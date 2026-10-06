package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Messages client may split a turn: Anthropic joins consecutive turns of one
// role itself, so assistant(tool_use A), assistant(tool_use B),
// user(tool_result A, tool_result B) is a valid history. Re-encoded for a
// Messages worker it goes out as one assistant turn holding both calls, so an
// adapter that pairs each result with the assistant message right before it
// finds both.
func TestAMessagesClientsSplitTurnIsJoined(t *testing.T) {
	body := []byte(`{"model":"m","max_tokens":64,"messages":[
		{"role":"user","content":"check both"},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_A","name":"read","input":{"path":"a"}}]},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_B","name":"read","input":{"path":"b"}}]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"toolu_A","content":"a"},
			{"type":"tool_result","tool_use_id":"toolu_B","content":"b"}]}],
		"tools":[{"name":"read","input_schema":{"type":"object"}}]}`)
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	request.Generation++ // routed: re-encode rather than replay the client's bytes
	result, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
				ID   string `json:"id"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Messages) != 3 || wire.Messages[1].Role != "assistant" || len(wire.Messages[1].Content) != 2 {
		t.Fatalf("want user, assistant(tool_use A, tool_use B), user(results): %s", result.Body)
	}
}
