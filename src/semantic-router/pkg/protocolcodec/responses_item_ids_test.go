package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A request translated into Responses carries no item id its client did not
// send: input ids are optional, and invented ones were duplicated across
// id-less messages.
func TestResponsesInputInventsNoItemIDs(t *testing.T) {
	engine := NewBuiltinEngine()
	chat := `{"model":"m","messages":[{"role":"user","content":"one"},{"role":"assistant","content":"two"},` +
		`{"role":"user","content":"three"}]}`
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(chat))
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "target-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Input []map[string]json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(encoded.Body, &wire); err != nil || len(wire.Input) != 3 {
		t.Fatalf("input = %s (%v)", encoded.Body, err)
	}
	for _, item := range wire.Input {
		if _, present := item["id"]; present {
			t.Fatalf("an input item carries an invented id: %s", encoded.Body)
		}
	}
}
