/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package thinkinglever

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// A steer placed after a Messages client's tool results travels as a trailing
// text block of the user message that carries them, with the client's
// grouping kept, as pathfinder's multi_tool_result golden expects; on the next
// turn it is replayed inside that same message.
func TestSteerAfterMessagesToolResultsJoinsTheirMessage(t *testing.T) {
	const history = `{"role":"user","content":"Inspect the repo."},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"Bash","input":{"command":"ls"}},` +
		`{"type":"tool_use","id":"c2","name":"Bash","input":{"command":"pwd"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"README.md"},` +
		`{"type":"tool_result","tool_use_id":"c2","content":"/work"}]}`
	const steered = `{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"text","text":"README.md"}]},` +
		`{"type":"tool_result","tool_use_id":"c2","content":[{"type":"text","text":"/work"}]},` +
		`{"type":"text","text":"` + steerDown + `"}]}`
	e := &episode{t: t, binding: suffixBinding(EmitOnChange, "none")}

	plan, provider := e.serve(decodeMessages(t, `[`+history+`]`), "down")
	if plan.Placement != PlaceUserAfterToolRun {
		t.Fatalf("placement = %q, want the steer after the tool run", plan.Placement)
	}
	sent := encodeMessages(t, provider)
	if len(sent) != 3 {
		t.Fatalf("sent %d messages, want the client's 3: %s", len(sent), sent)
	}
	assertMessageJSON(t, sent[2], steered)
	e.commit(plan)

	next := `[` + history + `,{"role":"assistant","content":[{"type":"text","text":"Done."}]},{"role":"user","content":"Thanks."}]`
	_, provider = e.serve(decodeMessages(t, next), "down")
	sent = encodeMessages(t, provider)
	if len(sent) != 5 {
		t.Fatalf("sent %d messages on the replay turn, want the client's 5: %s", len(sent), sent)
	}
	assertMessageJSON(t, sent[2], steered)
}

func decodeMessages(t *testing.T, messages string) []llmprotocol.Message {
	t.Helper()
	body := `{"model":"m","max_tokens":32,"messages":` + messages + `}`
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err != nil {
		t.Fatalf("decode messages: %v", err)
	}
	return request.Messages
}

func encodeMessages(t *testing.T, messages []llmprotocol.Message) []json.RawMessage {
	t.Helper()
	maxTokens := int64(32)
	request := llmprotocol.Request{Generation: 1, Model: "m", Messages: messages}
	request.Sampling.MaxOutputTokens = &maxTokens
	body, _, err := (protocolcodec.AnthropicMessagesCodec{}).EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("encode messages: %v", err)
	}
	var wire struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode encoded messages: %v", err)
	}
	return wire.Messages
}

func assertMessageJSON(t *testing.T, actual json.RawMessage, expected string) {
	t.Helper()
	var actualValue, expectedValue any
	if err := json.Unmarshal(actual, &actualValue); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(expected), &expectedValue); err != nil {
		t.Fatal(err)
	}
	actualText, _ := json.Marshal(actualValue)
	expectedText, _ := json.Marshal(expectedValue)
	if string(actualText) != string(expectedText) {
		t.Fatalf("message = %s\nwant      %s", actualText, expectedText)
	}
}
