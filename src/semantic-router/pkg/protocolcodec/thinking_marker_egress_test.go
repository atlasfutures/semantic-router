package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The decoders strip a Router signature, so these requests are built in the
// neutral form a path around the decoders would hand an encoder: each
// encoder must still refuse to send the signature to a provider.
func markerSignedRequest(signature string) llmprotocol.Request {
	text := func(role llmprotocol.Role, value string) llmprotocol.Message {
		return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: value}}}
	}
	return llmprotocol.Request{
		Model: "routed-model", Generation: 1,
		Messages: []llmprotocol.Message{
			text(llmprotocol.RoleUser, "weather in Paris?"),
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentReasoning, Reasoning: llmprotocol.ReasoningScopeText, Text: "kimi reasoning", Signature: signature},
				{Kind: llmprotocol.ContentText, Text: "It is sunny."},
			}},
			text(llmprotocol.RoleUser, "and tomorrow?"),
			// A message whose only block is marker-signed thinking.
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentReasoning, Reasoning: llmprotocol.ReasoningScopeText, Text: "more kimi reasoning", Signature: signature},
			}},
			text(llmprotocol.RoleUser, "thanks"),
		},
	}
}

func TestRequestEncodersNeverSendARouterSignature(t *testing.T) {
	signatures := map[string]string{
		"marker":    "vsr.thinking.v1.moonshotai." + testMarkerDigest,
		"malformed": "vsr.thinking.v1.MoonShot.x",
		"namespace": "vsr.something-else",
	}
	for name, signature := range signatures {
		for _, target := range goldenFormats {
			t.Run(name+"/"+target.name, func(t *testing.T) {
				result, err := NewBuiltinEngine().EncodeRequest(target.format, markerSignedRequest(signature), llmprotocol.Envelope{})
				if target.format == llmprotocol.OpenAIChatV1 {
					// Chat has no thinking signature: a signed content is
					// refused outright, and only reasoning_details can carry
					// one (TestChatEncoderRefusesARouterSignatureInReasoningDetails).
					if err == nil {
						t.Fatalf("a signed reasoning content was encoded for Chat: %s", result.Body)
					}
					return
				}
				if err != nil {
					t.Fatalf("encode: %v", err)
				}
				body := string(result.Body)
				if strings.Contains(body, "vsr.") {
					t.Fatalf("a Router signature reached the %s request: %s", target.name, body)
				}
				switch target.format {
				case llmprotocol.AnthropicMessagesV1:
					// Anthropic refuses thinking it did not sign: the block
					// goes whole, and so does a message left empty. The
					// request's two trailing user messages are one Messages
					// turn (a non-Messages history's consecutive turns of one
					// role are joined).
					if strings.Contains(body, "kimi reasoning") {
						t.Fatalf("unsigned thinking reached the Messages request: %s", body)
					}
					assertAnthropicMessageCount(t, result.Body, 3)
				default:
					// Responses carries the text as unsigned reasoning.
					if !strings.Contains(body, "kimi reasoning") {
						t.Fatalf("reasoning text was lost on the %s request: %s", target.name, body)
					}
				}
			})
		}
	}
}

func assertAnthropicMessageCount(t *testing.T, body []byte, want int) {
	t.Helper()
	var wire struct {
		Messages []struct {
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Messages) != want {
		t.Fatalf("messages = %d, want %d: %s", len(wire.Messages), want, body)
	}
	for index, message := range wire.Messages {
		if len(message.Content) == 0 {
			t.Fatalf("message %d has no content: %s", index, body)
		}
	}
}

func TestRequestEncodersKeepProviderSignatures(t *testing.T) {
	const signature = "EqQBCkgIARABGAIiQL2xRnd3"
	for _, target := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1} {
		result, err := NewBuiltinEngine().EncodeRequest(target, markerSignedRequest(signature), llmprotocol.Envelope{})
		if err != nil {
			t.Fatalf("encode %s: %v", target, err)
		}
		if got := strings.Count(string(result.Body), signature); got != 2 {
			t.Fatalf("%s request carries the provider signature %d times, want 2: %s", target, got, result.Body)
		}
	}
}

// A Claude worker reached over Chat is sent signed thinking as a
// reasoning_details item (SignedThinkingAsReasoningDetails). A marker there
// must not become that item's signature on the wire.
func TestChatEncoderRefusesARouterSignatureInReasoningDetails(t *testing.T) {
	request := markerSignedRequest("vsr.thinking.v1.moonshotai." + testMarkerDigest)
	if rewritten := SignedThinkingAsReasoningDetails(&request); rewritten != 2 {
		t.Fatalf("rewritten = %d, want 2", rewritten)
	}
	result, err := NewBuiltinEngine().EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if strings.Contains(string(result.Body), "vsr.") {
		t.Fatalf("a Router signature reached the Chat request: %s", result.Body)
	}
	if strings.Contains(string(result.Body), "reasoning_details") {
		t.Fatalf("an emptied reasoning_details array was sent: %s", result.Body)
	}
}

func TestWithoutRouterSignedDetailsKeepsOtherItemsByteForByte(t *testing.T) {
	clean := json.RawMessage(`[{"type":"reasoning.text","text":"a","signature":"EqQB","index":0}]`)
	got, err := withoutRouterSignedDetails(clean, func(string) {})
	if err != nil || string(got) != string(clean) {
		t.Fatalf("clean array = %s, %v; want it unchanged", got, err)
	}
	mixed := json.RawMessage(`[{"type":"reasoning.text","signature":"vsr.x","index":0},{"type":"reasoning.encrypted","data":"gAAA","index":1}]`)
	got, err = withoutRouterSignedDetails(mixed, func(string) {})
	if err != nil || string(got) != `[{"type":"reasoning.encrypted","data":"gAAA","index":1}]` {
		t.Fatalf("mixed array = %s, %v", got, err)
	}
	duplicate := json.RawMessage(`[{"signature":"vsr.x","signature":"EqQB"},{"data":"gAAA"}]`)
	got, err = withoutRouterSignedDetails(duplicate, func(string) {})
	if err != nil || string(got) != `[{"data":"gAAA"}]` {
		t.Fatalf("duplicate-key item = %s, %v; want it refused", got, err)
	}
	for _, aliased := range []string{
		`[{"ſignature":"EqQB","signature":"vsr.x"},{"data":"gAAA"}]`,
		`[{"signature":"vsr.x","ſignature":"EqQB"},{"data":"gAAA"}]`,
	} {
		got, err = withoutRouterSignedDetails(json.RawMessage(aliased), func(string) {})
		if err != nil || string(got) != `[{"data":"gAAA"}]` {
			t.Fatalf("folded-alias item %s = %s, %v; want it refused", aliased, got, err)
		}
	}
	if _, err := withoutRouterSignedDetails(json.RawMessage(`{"not":"an array"}`), func(string) {}); err == nil {
		t.Fatal("a non-array reasoning_details was accepted")
	}
}
