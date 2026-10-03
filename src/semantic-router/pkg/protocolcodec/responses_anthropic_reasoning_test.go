package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Claude signature as the provider sends it: opaque base64, long enough
// that a truncation or a join would show.
const claudeThinkingSignature = "EqQBCkYIBxgCKkDzqLx3m1y0b7zKq9Zt0x5JrD2sQ+fU8vH0pXcR3o1T5bN7wY2aM4eK6gS9dV1hL0jP3uC8iB5nA7xW2zE4EgwK1e3mA9rQ5tY7uIoaDF3pL8sV2cN6bH0jZiIwR4tY8uE1oP3aS5dF7gH9jK2lZ4xC6vB8nM0qW1eR3tY5uI7oP9aS2dF4gH6jK8lZ0="

const claudeRedactedThinkingData = "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpPkNRj2YfWXGmKDxH4mPnZ5sQ7vB6xa3bnKTiDnM5d8qvP2TaB"

// claudeTurnItems is a Claude turn as OpenRouter's /api/v1/responses writes
// it (the shape measured in atlasfutures/semantic-router#164): signed
// thinking, redacted thinking, then the answer.
func claudeTurnItems() []string {
	return []string{
		`{"type":"reasoning","id":"rs_01","summary":[],"content":[{"type":"reasoning_text","text":"The test fails because the fixture path is relative."}],` +
			`"signature":"` + claudeThinkingSignature + `","format":"anthropic-claude-v1"}`,
		`{"type":"reasoning","id":"rs_02","summary":[],"encrypted_content":"` + claudeRedactedThinkingData + `","format":"anthropic-claude-v1"}`,
		`{"type":"message","id":"msg_01","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Make the fixture path absolute.","annotations":[]}]}`,
	}
}

// replayedTurn is the next request of a Responses client (store:false) that
// got a turn: it resends the turn's output items verbatim, then asks again.
func replayedTurn(items []string) string {
	input := append([]string{`{"role":"user","content":"why does the test fail?"}`}, items...)
	input = append(input, `{"role":"user","content":"do it"}`)
	return `{"model":"auto","store":false,"input":[` + strings.Join(input, ",") + `]}`
}

// dispatchResponsesRequest prepares a Responses client's request for a worker
// of target as the router does: decode for mutation, make history reasoning
// carriable by the target, encode.
func dispatchResponsesRequest(t *testing.T, body string, target llmprotocol.WireFormat) (llmprotocol.Request, []byte) {
	t.Helper()
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	CarryReasoningTo(&request, target)
	request.Model = "provider-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(target, request, envelope)
	if err != nil {
		t.Fatalf("encode for %s: %v", target, err)
	}
	return request, encoded.Body
}

type anthropicBlockWire struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
	Data      string `json:"data"`
	Text      string `json:"text"`
}

func anthropicAssistantBlocks(t *testing.T, body []byte) []anthropicBlockWire {
	t.Helper()
	var wire struct {
		Messages []struct {
			Role    string               `json:"role"`
			Content []anthropicBlockWire `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("Messages body: %v\n%s", err, body)
	}
	var blocks []anthropicBlockWire
	for _, message := range wire.Messages {
		if message.Role == "assistant" {
			blocks = append(blocks, message.Content...)
		}
	}
	return blocks
}

// The replayed turn reaches a Messages worker as Claude sent it: the thinking
// with its identical signature, and the redacted block with its identical
// data. Without them the provider silently drops the thinking.
func TestReplayedClaudeReasoningReachesAMessagesWorkerSigned(t *testing.T) {
	_, body := dispatchResponsesRequest(t, replayedTurn(claudeTurnItems()), llmprotocol.AnthropicMessagesV1)
	var thinking, redacted, text int
	for _, block := range anthropicAssistantBlocks(t, body) {
		switch block.Type {
		case "thinking":
			thinking++
			if block.Signature != claudeThinkingSignature || block.Thinking != "The test fails because the fixture path is relative." {
				t.Fatalf("thinking block = %+v, want the original text and signature", block)
			}
		case "redacted_thinking":
			redacted++
			if block.Data != claudeRedactedThinkingData {
				t.Fatalf("redacted_thinking data = %q, want the original", block.Data)
			}
		case "text":
			text++
		}
	}
	if thinking != 1 || redacted != 1 || text != 1 {
		t.Fatalf("assistant blocks: %d thinking, %d redacted, %d text; want 1 each:\n%s", thinking, redacted, text, body)
	}
}

// Control: a reasoning item without the Anthropic format tag gains no
// signature, even when it holds a member of that name, so a Messages worker
// is sent no thinking it cannot verify.
func TestUntaggedReasoningItemInventsNoSignature(t *testing.T) {
	for name, item := range map[string]string{
		"no signature": `{"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"thought"}]}`,
		"untagged signature": `{"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"thought"}],` +
			`"signature":"` + claudeThinkingSignature + `"}`,
	} {
		body := `{"model":"auto","input":[{"role":"user","content":"q"},` + item +
			`,{"role":"assistant","content":[{"type":"output_text","text":"a"}]},{"role":"user","content":"q2"}]}`
		request, wire := dispatchResponsesRequest(t, body, llmprotocol.AnthropicMessagesV1)
		for _, message := range request.Messages {
			for _, content := range message.Content {
				if content.Signature != "" {
					t.Fatalf("%s: decoded a signature: %+v", name, content)
				}
			}
		}
		if bytes.Contains(wire, []byte(`"thinking"`)) || bytes.Contains(wire, []byte(claudeThinkingSignature)) {
			t.Fatalf("%s: a Messages worker was sent unverifiable thinking:\n%s", name, wire)
		}
	}
}

// A worker that is not Claude is never sent the signature or the redacted
// data, and the turn is not failed for them: a Chat worker keeps the thinking
// text, as it does for a Messages client's Claude thinking.
func TestReplayedClaudeReasoningToAnotherWorkerDropsTheSignature(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		request, body := dispatchResponsesRequest(t, replayedTurn(claudeTurnItems()), target)
		// The reasoning items' ids are the surface's that wrote them; a
		// Responses worker under store:false would look them up and fail.
		for _, mark := range []string{claudeThinkingSignature, claudeRedactedThinkingData, "anthropic-claude-v1", `"signature"`, `"rs_01"`, `"rs_02"`} {
			if bytes.Contains(body, []byte(mark)) {
				t.Fatalf("%s worker was sent %q:\n%s", target, mark, body)
			}
		}
		if !bytes.Contains(body, []byte("Make the fixture path absolute.")) {
			t.Fatalf("%s worker lost the visible answer:\n%s", target, body)
		}
		for _, message := range request.Messages {
			for _, content := range message.Content {
				if content.Signature != "" {
					t.Fatalf("%s: a signature survived the carry: %+v", target, content)
				}
			}
		}
	}
	_, chat := dispatchResponsesRequest(t, replayedTurn(claudeTurnItems()), llmprotocol.OpenAIChatV1)
	if !bytes.Contains(chat, []byte("The test fails because the fixture path is relative.")) {
		t.Fatalf("a Chat worker lost the thinking text:\n%s", chat)
	}
}

// OpenRouter's /responses gives Claude's turn in the same shape; an Anthropic
// client gets it back as signed thinking and redacted_thinking.
func TestOpenRouterResponsesClaudeReasoningReachesAMessagesClient(t *testing.T) {
	provider := `{"id":"gen-1","object":"response","created_at":100,"model":"anthropic/claude-haiku-4.5","status":"completed","output":[` +
		`{"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"check the path"}],` +
		`"signature":"` + claudeThinkingSignature + `","format":"anthropic-claude-v1"},` +
		`{"type":"reasoning","id":"rs_2","summary":[],"encrypted_content":"` + claudeRedactedThinkingData + `","format":"anthropic-claude-v1"},` +
		`{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],` +
		`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`
	body := translateResponsesResponse(t, provider, llmprotocol.AnthropicMessagesV1)
	var response struct {
		Content []anthropicBlockWire `json:"content"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		t.Fatalf("response: %v\n%s", err, body)
	}
	if len(response.Content) != 3 ||
		response.Content[0].Type != "thinking" || response.Content[0].Signature != claudeThinkingSignature || response.Content[0].Thinking != "check the path" ||
		response.Content[1].Type != "redacted_thinking" || response.Content[1].Data != claudeRedactedThinkingData ||
		response.Content[2].Text != "done" {
		t.Fatalf("Anthropic client got %s", body)
	}
}
