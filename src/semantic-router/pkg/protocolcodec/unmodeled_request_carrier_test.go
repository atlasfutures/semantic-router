package protocolcodec

import (
	"bytes"
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

// Shape 7 in the fixture, and the largest single ingress failure in the
// measured Claude Code corpus: a tool_reference block nested inside a
// tool_result. The carrier has to reach nested content, not only the top-level
// block list.
const anthropicToolReferenceBody = `{
  "model": "claude-sonnet-4-5",
  "max_tokens": 64,
  "messages": [
    {"role": "assistant", "content": [
      {"type": "tool_use", "id": "call_grep", "name": "grep", "input": {"q": "x"}}
    ]},
    {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "call_grep", "content": [
        {"type": "text", "text": "3 hits"},
        {"type": "tool_reference", "tool_name": "grep"}
      ]}
    ]}
  ]
}`

func anthropicToolResultBlocks(t *testing.T, body []byte) []map[string]json.RawMessage {
	t.Helper()
	var wire struct {
		Messages []struct {
			Content []map[string]json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("routed body is not an Anthropic request: %v", err)
	}
	for _, message := range wire.Messages {
		for _, block := range message.Content {
			if string(block["type"]) != `"tool_result"` {
				continue
			}
			var nested []map[string]json.RawMessage
			if err := json.Unmarshal(block["content"], &nested); err != nil {
				t.Fatalf("tool_result content is not a block list: %v", err)
			}
			return nested
		}
	}
	t.Fatal("the routed body carries no tool_result block")
	return nil
}

func TestAnthropicUnmodeledNestedBlockReachesTheSameFormat(t *testing.T) {
	blocks := anthropicToolResultBlocks(t, routeAnthropicRequest(t, anthropicToolReferenceBody, llmprotocol.AnthropicMessagesV1))
	if len(blocks) != 2 {
		t.Fatalf("tool_result holds %d blocks, want the text block and the carried one", len(blocks))
	}
	if string(blocks[1]["type"]) != `"tool_reference"` {
		t.Fatalf("second block is %s, want the carried tool_reference", blocks[1]["type"])
	}
	if string(blocks[1]["tool_name"]) != `"grep"` {
		t.Fatalf("carried block lost its members: %v", blocks[1])
	}
}

func TestAnthropicUnmodeledNestedBlockDoesNotReachAnotherFormat(t *testing.T) {
	routed := routeAnthropicRequest(t, anthropicToolReferenceBody, llmprotocol.OpenAIChatV1)
	if bytes.Contains(routed, []byte("tool_reference")) {
		t.Fatalf("a block the target cannot name reached it: %s", routed)
	}
	if !bytes.Contains(routed, []byte("3 hits")) {
		t.Fatalf("dropping the carried block also dropped the text beside it: %s", routed)
	}
}

// Shape 1 in the fixture, and 9.4 percent of the measured Workshop corpus: an
// inline base64 image in user content. The Router never reads the payload
// bytes; the provider does. Parsing them at ingress only turns a routable body
// into a 400.
func anthropicInlineImageBody(data string) string {
	return `{
	  "model": "claude-sonnet-4-5",
	  "max_tokens": 64,
	  "messages": [{"role": "user", "content": [
	    {"type": "text", "text": "what is this"},
	    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "` + data + `"}}
	  ]}]
	}`
}

func TestAnthropicInlineImageReachesAChatProvider(t *testing.T) {
	routed := routeAnthropicRequest(t, anthropicInlineImageBody("aW1hZ2U="), llmprotocol.OpenAIChatV1)
	if !bytes.Contains(routed, []byte(`"url":"data:image/png;base64,aW1hZ2U="`)) {
		t.Fatalf("the inline image did not reach the provider as a data URI: %s", routed)
	}
}

// The same image has to come back. A provider answer is decoded against the
// destination format, so the round trip is what keeps an image-bearing turn
// routable in both directions.
func TestChatInlineImageReturnsToAnthropic(t *testing.T) {
	engine := NewBuiltinEngine()
	chatBody := routeAnthropicRequest(t, anthropicInlineImageBody("aW1hZ2U="), llmprotocol.OpenAIChatV1)
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, chatBody)
	if err != nil {
		t.Fatalf("the routed Chat body did not decode: %v", err)
	}
	request.Generation++
	result, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatalf("EncodeRequest(anthropic) error = %v", err)
	}
	if !bytes.Contains(result.Body, []byte(`"media_type":"image/png"`)) ||
		!bytes.Contains(result.Body, []byte(`"data":"aW1hZ2U="`)) {
		t.Fatalf("the image did not return to its Anthropic shape: %s", result.Body)
	}
}

// The payload the Router never reads must not decide whether the request
// routes. A payload the provider will reject is the provider's answer to give.
func TestAnthropicInlineImagePayloadIsNotParsedAtIngress(t *testing.T) {
	engine := NewBuiltinEngine()
	_, _, _, err := engine.DecodeRequestForMutation(
		llmprotocol.AnthropicMessagesV1, []byte(anthropicInlineImageBody("not-base64!!")),
	)
	if err != nil {
		t.Fatalf("a body was refused for payload bytes the Router never reads: %v", err)
	}
}
