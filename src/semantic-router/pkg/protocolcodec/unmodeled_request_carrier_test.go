package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The Router selects a destination; it does not edit the payload. A member the
// client and the provider both understand must therefore reach the provider
// even when the neutral request has no name for it. These tests pin that
// contract for the members live Anthropic Messages traffic actually carries.

const anthropicUnmodeledTopLevelBody = `{
  "model": "claude-sonnet-4-5",
  "max_tokens": 64,
  "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
  "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
  "diagnostics": {"enabled": true},
  "fallbacks": ["claude-haiku-4-5"],
  "response_format": {"type": "text"}
}`

var anthropicUnmodeledTopLevelNames = []string{
	"context_management", "diagnostics", "fallbacks", "response_format",
}

// routeAnthropicRequest reproduces what Router ingress does: decode for
// mutation, change the model the way selection does, and re-encode for a
// destination.
func routeAnthropicRequest(
	t *testing.T,
	body string,
	target llmprotocol.WireFormat,
) []byte {
	t.Helper()
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err != nil {
		t.Fatalf("DecodeRequestForMutation() error = %v", err)
	}
	request.Model = "selected-arm"
	request.Generation++
	result, err := engine.EncodeRequest(target, request, envelope)
	if err != nil {
		t.Fatalf("EncodeRequest(%s) error = %v", target, err)
	}
	return result.Body
}

func decodeJSONObject(t *testing.T, body []byte) map[string]json.RawMessage {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		t.Fatalf("encoded body is not a JSON object: %v", err)
	}
	return object
}

func TestAnthropicUnmodeledTopLevelFieldsReachTheSameFormat(t *testing.T) {
	source := decodeJSONObject(t, []byte(anthropicUnmodeledTopLevelBody))
	routed := decodeJSONObject(t, routeAnthropicRequest(t, anthropicUnmodeledTopLevelBody, llmprotocol.AnthropicMessagesV1))
	for _, name := range anthropicUnmodeledTopLevelNames {
		value, present := routed[name]
		if !present {
			t.Fatalf("%q did not survive routing to the same wire format", name)
		}
		if !jsonEqual(t, value, source[name]) {
			t.Fatalf("%q changed: got %s, want %s", name, value, source[name])
		}
	}
	if string(routed["model"]) != `"selected-arm"` {
		t.Fatalf("model = %s, want the selected arm", routed["model"])
	}
}

func TestAnthropicUnmodeledTopLevelFieldsDoNotReachAnotherFormat(t *testing.T) {
	routed := decodeJSONObject(t, routeAnthropicRequest(t, anthropicUnmodeledTopLevelBody, llmprotocol.OpenAIChatV1))
	for _, name := range anthropicUnmodeledTopLevelNames {
		if _, present := routed[name]; present {
			t.Fatalf("%q reached a wire format that does not name it", name)
		}
	}
	if string(routed["model"]) != `"selected-arm"` {
		t.Fatalf("model = %s, want the selected arm", routed["model"])
	}
}

// A member the codec does name is still decoded strictly. The carrier widens
// the request contract; it does not relax the fields inside it.
func TestAnthropicNamedFieldsStayStrictBesideTheCarrier(t *testing.T) {
	engine := NewBuiltinEngine()
	body := `{
	  "model": "claude-sonnet-4-5",
	  "max_tokens": 64,
	  "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "unknown_member": 1}]}],
	  "context_management": {"edits": []}
	}`
	if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body)); err == nil {
		t.Fatal("an unknown member inside a named block was accepted")
	}
}

func jsonEqual(t *testing.T, left, right json.RawMessage) bool {
	t.Helper()
	var leftValue, rightValue any
	if err := json.Unmarshal(left, &leftValue); err != nil {
		t.Fatalf("left value is not JSON: %v", err)
	}
	if err := json.Unmarshal(right, &rightValue); err != nil {
		t.Fatalf("right value is not JSON: %v", err)
	}
	leftBody, _ := json.Marshal(leftValue)
	rightBody, _ := json.Marshal(rightValue)
	return string(leftBody) == string(rightBody)
}
