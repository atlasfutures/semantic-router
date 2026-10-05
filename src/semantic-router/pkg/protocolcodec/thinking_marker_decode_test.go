package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const markerHistoryTemplate = `{
  "model": "client-model",
  "max_tokens": 64,
  "messages": [
    {"role": "user", "content": "weather in Paris?"},
    {"role": "assistant", "content": [
      {"type": "thinking", "thinking": "kimi reasoning", "signature": "SIGNATURE"},
      {"type": "text", "text": "It is sunny."}
    ]},
    {"role": "user", "content": "and tomorrow?"}
  ]
}`

func decodeMarkerHistory(t *testing.T, signature string) llmprotocol.Request {
	t.Helper()
	body := strings.Replace(markerHistoryTemplate, "SIGNATURE", signature, 1)
	request, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	return request
}

func reasoningContents(request llmprotocol.Request) []llmprotocol.Content {
	var found []llmprotocol.Content
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentReasoning {
				found = append(found, content)
			}
		}
	}
	return found
}

// Every value in the Router's namespace -- an exact marker, a forged one
// (any family and digest: v1 markers are keyless) and a malformed one --
// decodes to unsigned thinking, so it never counts as Anthropic's proof.
func TestAnthropicDecoderStripsRouterSignatures(t *testing.T) {
	for _, signature := range []string{
		"vsr.thinking.v1.moonshotai." + testMarkerDigest,
		"vsr.thinking.v1.anthropic." + strings.Repeat("A", 32),
		"vsr.thinking.v1.MoonShot.x",
		"vsr.",
	} {
		reasoning := reasoningContents(decodeMarkerHistory(t, signature))
		if len(reasoning) != 1 || reasoning[0].Signature != "" || reasoning[0].Text != "kimi reasoning" {
			t.Fatalf("%q decoded to %+v, want one unsigned thinking block", signature, reasoning)
		}
	}
}

func TestAnthropicDecoderKeepsProviderSignatures(t *testing.T) {
	reasoning := reasoningContents(decodeMarkerHistory(t, "EqQBCkgIARABGAIiQL2xRnd3"))
	if len(reasoning) != 1 || reasoning[0].Signature != "EqQBCkgIARABGAIiQL2xRnd3" {
		t.Fatalf("decoded to %+v, want the provider signature kept", reasoning)
	}
}

// Once decoded, marker-signed thinking follows the rules for unsigned
// reasoning (#184): dropped for a Claude worker, carried as reasoning to a
// Chat worker, dropped for any other Messages worker.
func TestMarkerSignedThinkingFollowsTheUnsignedReasoningRules(t *testing.T) {
	marker := "vsr.thinking.v1.moonshotai." + testMarkerDigest

	claude := decodeMarkerHistory(t, marker)
	if dropped := DropReasoningNotFromAnthropic(&claude, llmprotocol.AnthropicMessagesV1); dropped != 1 {
		t.Fatalf("Claude worker: foreign_dropped = %d, want 1", dropped)
	}
	if reasoning := reasoningContents(claude); len(reasoning) != 0 {
		t.Fatalf("Claude worker kept %+v", reasoning)
	}

	chat := decodeMarkerHistory(t, marker)
	if carry := CarryReasoningTo(&chat, llmprotocol.OpenAIChatV1); carry.Changed() {
		t.Fatalf("Chat worker: carry changed the history: %+v", carry)
	}
	result, err := NewBuiltinEngine().EncodeRequest(llmprotocol.OpenAIChatV1, chat, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("encode Chat: %v", err)
	}
	if body := string(result.Body); !strings.Contains(body, `"reasoning_content":"kimi reasoning"`) || strings.Contains(body, "vsr.") {
		t.Fatalf("Chat worker request = %s", body)
	}

	messages := decodeMarkerHistory(t, marker)
	if carry := CarryReasoningTo(&messages, llmprotocol.AnthropicMessagesV1); carry.UnsignedDropped != 1 {
		t.Fatalf("Messages worker: carry = %+v, want one unsigned_dropped", carry)
	}
}

func TestResponsesDecoderStripsRouterSignatures(t *testing.T) {
	body := `{"model":"client-model","input":[
	  {"role":"user","content":[{"type":"input_text","text":"weather in Paris?"}]},
	  {"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"kimi reasoning"}],
	   "signature":"vsr.thinking.v1.moonshotai.` + testMarkerDigest + `","format":"anthropic-claude-v1"},
	  {"role":"user","content":[{"type":"input_text","text":"and tomorrow?"}]}]}`
	request, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	reasoning := reasoningContents(request)
	if len(reasoning) != 1 || reasoning[0].Signature != "" {
		t.Fatalf("decoded to %+v, want one unsigned reasoning content", reasoning)
	}
	if dropped := DropReasoningNotFromAnthropic(&request, llmprotocol.OpenAIResponsesV1); dropped != 1 {
		t.Fatalf("Claude worker: foreign_dropped = %d, want 1", dropped)
	}
}

// A Chat client's reasoning_details item names its own format, so an
// anthropic-claude-v1 item under a Router marker is the client's claim, not
// Claude's provenance. The decoder strips it, and what remains is unsigned
// reasoning: foreign for a Claude worker, carried for an open-weight one.
func TestChatDecoderStripsRouterSignedReasoningDetails(t *testing.T) {
	marker := "vsr.thinking.v1.moonshotai." + testMarkerDigest
	body := `{"model":"m","messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":"ok","reasoning_content":"kimi reasoning","reasoning_details":[` +
		`{"type":"reasoning.text","text":"kimi reasoning","signature":"` + marker + `","format":"anthropic-claude-v1","index":0}]},` +
		`{"role":"user","content":"go on"}]}`
	decode := func() llmprotocol.Request {
		request, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, []byte(body))
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		return request
	}
	reasoning := reasoningContents(decode())
	if len(reasoning) != 1 || reasoning[0].Extensions != nil || reasoning[0].Text != "kimi reasoning" {
		t.Fatalf("decoded to %+v, want unsigned reasoning with no reasoning_details", reasoning)
	}
	claude := decode()
	if dropped := DropReasoningNotFromAnthropic(&claude, llmprotocol.OpenAIChatV1); dropped != 1 {
		t.Fatalf("Claude worker: foreign_dropped = %d, want 1", dropped)
	}
	if left := reasoningContents(claude); len(left) != 0 {
		t.Fatalf("Claude worker kept %+v", left)
	}
}

// Claude's own item beside a marker item survives, with only the text it
// holds; the marker item and its reasoning go.
func TestChatDecoderKeepsClaudeItemsBesideAMarkerItem(t *testing.T) {
	marker := "vsr.thinking.v1.moonshotai." + testMarkerDigest
	body := `{"model":"m","messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":"ok","reasoning_content":"kimi reasoningclaude reasoning","reasoning_details":[` +
		`{"type":"reasoning.text","text":"kimi reasoning","signature":"` + marker + `","format":"anthropic-claude-v1","index":0},` +
		`{"type":"reasoning.text","text":"claude reasoning","signature":"EqQB","format":"anthropic-claude-v1","index":1}]},` +
		`{"role":"user","content":"go on"}]}`
	request, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	// The kimi text beside Claude's item is dropped and counted as foreign,
	// as when the provenance filter removes an item itself.
	if dropped := DropReasoningNotFromAnthropic(&request, llmprotocol.OpenAIChatV1); dropped != 1 {
		t.Fatalf("foreign_dropped = %d, want 1", dropped)
	}
	reasoning := reasoningContents(request)
	if len(reasoning) != 1 || reasoning[0].Text != "claude reasoning" {
		t.Fatalf("Claude worker reasoning = %+v, want only Claude's text", reasoning)
	}
	details, _ := reasoningDetailsOf(reasoning[0])
	if strings.Contains(string(details), "vsr.") || !strings.Contains(string(details), "EqQB") {
		t.Fatalf("reasoning_details = %s", details)
	}
}

// A minted Responses blob is bytes the client holds and resends, so its items
// are stripped the same way.
func TestMintedReasoningDetailsStripRouterSignedItems(t *testing.T) {
	marker := "vsr.thinking.v1.moonshotai." + testMarkerDigest
	only := mintReasoningDetails(json.RawMessage(`[{"type":"reasoning.text","text":"kimi","signature":"` + marker + `","format":"anthropic-claude-v1","index":0}]`))
	if details, ok := mintedReasoningDetails(only); !ok || details != nil {
		t.Fatalf("a blob of only marker items decoded to %s, %v; want minted with no items", details, ok)
	}
	mixed := mintReasoningDetails(json.RawMessage(`[{"type":"reasoning.text","text":"kimi","signature":"` + marker + `","index":0},{"type":"reasoning.encrypted","data":"gAAA","index":1}]`))
	details, ok := mintedReasoningDetails(mixed)
	if !ok || strings.Contains(string(details), "vsr.") || !strings.Contains(string(details), "gAAA") {
		t.Fatalf("mixed blob decoded to %s, %v", details, ok)
	}
}

// A Responses target is never sent a minted blob, an emptied one included:
// the item decodes to its reasoning text and the request is encoded afresh.
func TestEmptiedMintedBlobNeverReachesAResponsesTarget(t *testing.T) {
	marker := "vsr.thinking.v1.moonshotai." + testMarkerDigest
	blob := mintReasoningDetails(json.RawMessage(`[{"type":"reasoning.text","text":"kimi reasoning","signature":"` + marker + `","format":"anthropic-claude-v1","index":0}]`))
	body := `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]},` +
		`{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"kimi reasoning"}],"encrypted_content":` + string(blob) + `},` +
		`{"role":"user","content":[{"type":"input_text","text":"go on"}]}]}`
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if reasoning := reasoningContents(request); len(reasoning) != 1 || reasoning[0].Text != "kimi reasoning" || reasoning[0].Extensions != nil {
		t.Fatalf("decoded to %+v, want unsigned reasoning text", reasoning)
	}
	result, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if sent := string(result.Body); strings.Contains(sent, "vsr.") || strings.Contains(sent, "encrypted_content") ||
		!strings.Contains(sent, "kimi reasoning") {
		t.Fatalf("Responses target request = %s", sent)
	}
}
