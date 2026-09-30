package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// What a Responses provider returns to a client that asked for
// include: ["reasoning.encrypted_content"]: a reasoning item with no summary
// that carries only the blob, then a function call.
const encryptedReasoningResponse = `{"id":"resp_1","object":"response","created_at":100,"model":"gpt-5-codex",` +
	`"status":"completed","output":[` +
	`{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAAB-issued"},` +
	`{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"exec_command","arguments":"{}"}],` +
	`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`

func translateResponsesResponse(t *testing.T, body string, client llmprotocol.WireFormat) []byte {
	t.Helper()
	engine := NewBuiltinEngine()
	response, envelope, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	response.Model = "public-model"
	response.Generation++
	encoded, err := engine.EncodeResponse(client, response, envelope)
	if err != nil {
		t.Fatalf("EncodeResponse(%s) error = %v", client, err)
	}
	return encoded.Body
}

// The blob reaches a Responses client, which resends it next turn. A Chat
// client sees the call and nothing of the blob, not even an empty reasoning
// field; before this, the reasoning item failed the whole response.
func TestEncryptedReasoningReachesAResponsesClient(t *testing.T) {
	responses := translateResponsesResponse(t, encryptedReasoningResponse, llmprotocol.OpenAIResponsesV1)
	if !bytes.Contains(responses, []byte(`"encrypted_content":"gAAAAB-issued"`)) || !bytes.Contains(responses, []byte(`"summary":[]`)) {
		t.Fatalf("the reasoning item lost its blob: %s", responses)
	}
	chat := translateResponsesResponse(t, encryptedReasoningResponse, llmprotocol.OpenAIChatV1)
	if bytes.Contains(chat, []byte("gAAAAB")) || bytes.Contains(chat, []byte(`"reasoning_content"`)) || bytes.Contains(chat, []byte(`"reasoning":`)) {
		t.Fatalf("the blob or an empty reasoning field reached a Chat client: %s", chat)
	}
	if !bytes.Contains(chat, []byte("exec_command")) {
		t.Fatalf("the tool call was lost: %s", chat)
	}
}

func encryptedReasoningStream() string {
	frame := func(event string, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	return frame("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":100,"model":"gpt-5-codex","status":"in_progress","output":[]}}`) +
		frame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`) +
		frame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAAB-issued"}}`) +
		frame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":3,"output_index":1,"item":{"type":"function_call","id":"fc_1","status":"in_progress","call_id":"call_1","name":"exec_command","arguments":""}}`) +
		frame("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","sequence_number":4,"output_index":1,"item_id":"fc_1","delta":"{}"}`) +
		frame("response.function_call_arguments.done", `{"type":"response.function_call_arguments.done","sequence_number":5,"output_index":1,"item_id":"fc_1","name":"exec_command","arguments":"{}"}`) +
		frame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":6,"output_index":1,"item":{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"exec_command","arguments":"{}"}}`) +
		frame("response.completed", `{"type":"response.completed","sequence_number":7,"response":{"id":"resp_1","object":"response","created_at":100,"model":"gpt-5-codex","status":"completed","output":[`+
			`{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAAB-issued"},`+
			`{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"exec_command","arguments":"{}"}],`+
			`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`)
}

// Codex streams, so the blob has to survive the stream too: on the reasoning
// item's done event and in the completed response's output.
func TestEncryptedReasoningSurvivesAResponsesStream(t *testing.T) {
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "gpt-5-codex",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(encryptedReasoningStream()))
	if err != nil {
		t.Fatalf("push: %v", err)
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	var itemDone, completed bool
	for _, frame := range append(frames, final...) {
		text := string(frame)
		data := text[strings.Index(text, "data: ")+len("data: "):]
		var event struct {
			Type     string          `json:"type"`
			Item     json.RawMessage `json:"item"`
			Response json.RawMessage `json:"response"`
		}
		if json.Unmarshal([]byte(strings.TrimSpace(data)), &event) != nil {
			continue
		}
		switch {
		case event.Type == "response.output_item.done" && bytes.Contains(event.Item, []byte(`"reasoning"`)):
			itemDone = bytes.Contains(event.Item, []byte(`"encrypted_content":"gAAAAB-issued"`))
		case event.Type == "response.completed":
			completed = bytes.Contains(event.Response, []byte(`"encrypted_content":"gAAAAB-issued"`))
		}
	}
	if !itemDone || !completed {
		t.Fatalf("blob on reasoning done=%v, on completed output=%v:\n%s", itemDone, completed, bytes.Join(append(frames, final...), nil))
	}
}

// A provider may name a call's namespace only when the item completes; the
// client still gets it.
func TestStreamNamespaceLearnedAtCompletion(t *testing.T) {
	frame := func(event string, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	body := frame("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":100,"model":"m","status":"in_progress","output":[]}}`) +
		frame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"function_call","id":"fc_1","status":"in_progress","call_id":"call_1","name":"spawn_agent","arguments":""}}`) +
		frame("response.function_call_arguments.done", `{"type":"response.function_call_arguments.done","sequence_number":2,"output_index":0,"item_id":"fc_1","name":"spawn_agent","arguments":"{}"}`) +
		frame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","namespace":"multi_agent_v1","name":"spawn_agent","arguments":"{}"}}`) +
		frame("response.completed", `{"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","object":"response","created_at":100,"model":"m","status":"completed","output":[`+
			`{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","namespace":"multi_agent_v1","name":"spawn_agent","arguments":"{}"}],`+
			`"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`)
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "m",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(body))
	if err != nil {
		t.Fatalf("push: %v", err)
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	wire := bytes.Join(append(frames, final...), nil)
	done := wire[bytes.Index(wire, []byte("response.output_item.done")):]
	if !bytes.Contains(done, []byte(`"namespace":"multi_agent_v1"`)) {
		t.Fatalf("the namespace learned at completion did not reach the client:\n%s", wire)
	}
}
