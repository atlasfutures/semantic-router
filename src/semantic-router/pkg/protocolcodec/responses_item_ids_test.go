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

// A Chat message's id names the assistant message. Reasoning now comes
// first in the Responses output, and the message id stays on the message
// item rather than moving to the reasoning item ahead of it.
func TestResponsesOutputKeepsTheChatMessageIDOnTheMessage(t *testing.T) {
	chat := `{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"id":"msg_1","role":"assistant",` +
		`"content":"done","reasoning":"inspect"}}]}`
	response, envelope, _, err := (OpenAIChatCodec{}).DecodeResponse([]byte(chat), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	body, _, err := (OpenAIResponsesCodec{}).EncodeResponse(response, envelope, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Output []struct {
			Type string `json:"type"`
			ID   string `json:"id"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &wire); err != nil || len(wire.Output) != 2 {
		t.Fatalf("output = %s (%v)", body, err)
	}
	reasoning, message := wire.Output[0], wire.Output[1]
	if reasoning.Type != "reasoning" || message.Type != "message" {
		t.Fatalf("output order = %s, %s; want reasoning, message", reasoning.Type, message.Type)
	}
	if message.ID != "msg_1" {
		t.Fatalf("message id = %q, want msg_1", message.ID)
	}
	if reasoning.ID == "" || reasoning.ID == "msg_1" {
		t.Fatalf("reasoning id = %q, want a generated id", reasoning.ID)
	}
}
