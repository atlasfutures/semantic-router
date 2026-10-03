package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An image inside a tool result reaches a Chat arm as a user message after the
// run of tool messages. The golden 019-anthropic-tool-result-image covers
// OpenClaw's view_image shape; these cover the edges of the run.

const toolResultMediaTestPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"

func chatMessagesFor(t *testing.T, anthropicBody string) []map[string]json.RawMessage {
	t.Helper()
	result, err := NewBuiltinEngine().TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(anthropicBody), nil,
	)
	if err != nil {
		t.Fatalf("a tool result carrying an image was refused for a Chat arm: %v", err)
	}
	var wire struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	return wire.Messages
}

func messageRole(t *testing.T, message map[string]json.RawMessage) string {
	t.Helper()
	var role string
	if err := json.Unmarshal(message["role"], &role); err != nil {
		t.Fatal(err)
	}
	return role
}

func messageParts(t *testing.T, message map[string]json.RawMessage) []chatContentWire {
	t.Helper()
	var parts []chatContentWire
	if err := json.Unmarshal(message["content"], &parts); err == nil {
		return parts
	}
	var text string
	if err := json.Unmarshal(message["content"], &text); err != nil {
		t.Fatalf("message content is neither parts nor a string: %s", message["content"])
	}
	return []chatContentWire{{Type: "text", Text: text}}
}

// The media message must not sit between two tool messages: Chat requires the
// answers to an assistant's tool calls to follow it unbroken. An image-only
// result keeps a non-empty tool message, and the text the caller sent after
// the results stays after the media.
func TestChatToolResultMediaFollowsTheWholeToolRun(t *testing.T) {
	body := `{"model":"m","max_tokens":16,"messages":[` +
		`{"role":"user","content":"look"},` +
		`{"role":"assistant","content":[` +
		`{"type":"tool_use","id":"call_img","name":"view_image","input":{}},` +
		`{"type":"tool_use","id":"call_txt","name":"exec","input":{}}]},` +
		`{"role":"user","content":[` +
		`{"type":"tool_result","tool_use_id":"call_img","content":[` +
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + toolResultMediaTestPNG + `"}}]},` +
		`{"type":"tool_result","tool_use_id":"call_txt","content":"ok"},` +
		`{"type":"text","text":"what do you see?"}]}]}`
	messages := chatMessagesFor(t, body)
	roles := make([]string, 0, len(messages))
	for _, message := range messages {
		roles = append(roles, messageRole(t, message))
	}
	want := []string{"user", "assistant", "tool", "tool", "user", "user"}
	if len(roles) != len(want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	for index := range want {
		if roles[index] != want[index] {
			t.Fatalf("roles = %v, want %v", roles, want)
		}
	}
	if parts := messageParts(t, messages[2]); len(parts) != 1 || parts[0].Text != toolResultMediaPointer {
		t.Fatalf("image-only tool message = %s, want the pointer text", messages[2]["content"])
	}
	media := messageParts(t, messages[4])
	if len(media) != 2 || media[0].Text != toolResultMediaLabel("call_img") ||
		media[1].Type != "image_url" || media[1].ImageURL == nil ||
		media[1].ImageURL.URL != "data:image/png;base64,"+toolResultMediaTestPNG {
		t.Fatalf("media message = %s", messages[4]["content"])
	}
	if parts := messageParts(t, messages[5]); len(parts) != 1 || parts[0].Text != "what do you see?" {
		t.Fatalf("trailing user text = %s", messages[5]["content"])
	}
}

// A text-only tool turn is unchanged: no pointer, no extra message.
func TestChatToolResultWithoutMediaAddsNoMessage(t *testing.T) {
	body := `{"model":"m","max_tokens":16,"messages":[` +
		`{"role":"user","content":"run"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"exec","input":{}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":"ok"}]}]}`
	messages := chatMessagesFor(t, body)
	if len(messages) != 3 || messageRole(t, messages[2]) != "tool" {
		t.Fatalf("messages = %v, want user, assistant, tool", messages)
	}
}

// A message encoded on its own, as the response path does, has nowhere to put
// the media, so it keeps the text-only refusal rather than dropping it.
func TestSingleChatMessageStillRefusesToolResultMedia(t *testing.T) {
	_, err := encodeChatMessage(llmprotocol.Message{
		Role: llmprotocol.RoleTool,
		Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult,
			ToolResult: &llmprotocol.ToolResult{CallID: "c1", Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentImage, MediaType: "image/png", Data: toolResultMediaTestPNG,
			}}},
		}},
	})
	if err == nil {
		t.Fatal("a lone tool message with an image encoded without its media")
	}
}
