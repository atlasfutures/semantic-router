package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A tool result that returned an image (Claude Code reading a PNG, an MCP
// screenshot) reaches a Chat arm whole: the tool message keeps its text and
// the image follows in a user message after the run of tool messages, since
// Chat tool messages carry text only and must directly follow their
// assistant turn.
func TestChatEncodeMovesToolResultMediaAfterTheToolMessages(t *testing.T) {
	const png = "iVBORw0KGgo="
	request := []byte(`{
		"model":"m","max_tokens":16,
		"messages":[
			{"role":"user","content":"look at these"},
			{"role":"assistant","content":[
				{"type":"tool_use","id":"call_a","name":"read","input":{"path":"a.png"}},
				{"type":"tool_use","id":"call_b","name":"read","input":{"path":"b.txt"}},
				{"type":"tool_use","id":"call_c","name":"read","input":{"path":"c.png"}}
			]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"call_a","content":[
					{"type":"text","text":"a.png, 1x1"},
					{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + png + `"}}
				]},
				{"type":"tool_result","tool_use_id":"call_b","content":"plain text"},
				{"type":"tool_result","tool_use_id":"call_c","content":[
					{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + png + `"}}
				]},
				{"type":"text","text":"what do you see?"}
			]}
		]
	}`)
	engine := NewBuiltinEngine()
	decoded, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, request)
	if err != nil {
		t.Fatal(err)
	}
	if got := llmprotocol.RequiredRoutingCapabilities(decoded); len(got) != 1 || got[0] != llmprotocol.RoutingCapabilityToolResultImages {
		t.Fatalf("required capabilities = %v", got)
	}
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, decoded, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var wire struct {
		Messages []struct {
			Role       string          `json:"role"`
			ToolCallID string          `json:"tool_call_id"`
			Content    json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	var roles []string
	for _, message := range wire.Messages {
		roles = append(roles, message.Role+":"+message.ToolCallID)
	}
	want := []string{"user:", "assistant:", "tool:call_a", "tool:call_b", "tool:call_c", "user:", "user:"}
	if strings.Join(roles, ",") != strings.Join(want, ",") {
		t.Fatalf("messages = %v, want %v", roles, want)
	}
	tools := map[string]string{}
	for _, message := range wire.Messages[2:5] {
		tools[message.ToolCallID] = string(message.Content)
	}
	if !strings.Contains(tools["call_a"], "a.png, 1x1") || strings.Contains(tools["call_a"], "image_url") ||
		!strings.Contains(tools["call_c"], "follows in the next user message") || !strings.Contains(tools["call_b"], "plain text") {
		t.Fatalf("tool messages = %v", tools)
	}
	media := string(wire.Messages[5].Content)
	if strings.Count(media, `"type":"image_url"`) != 2 || !strings.Contains(media, "tool call call_a") ||
		!strings.Contains(media, "tool call call_c") || strings.Contains(media, "call_b") ||
		!strings.Contains(media, "data:image/png;base64,"+png) {
		t.Fatalf("media message = %s", media)
	}
	if !strings.Contains(string(wire.Messages[6].Content), "what do you see?") {
		t.Fatalf("the user's own text moved: %s", wire.Messages[6].Content)
	}
}

// Control: a text-only tool result encodes exactly as before, with no
// message added.
func TestChatEncodeLeavesTextToolResultsAlone(t *testing.T) {
	request := []byte(`{
		"model":"m","max_tokens":16,
		"messages":[
			{"role":"user","content":"go"},
			{"role":"assistant","content":[{"type":"tool_use","id":"call_a","name":"read","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a","content":"done"}]}
		]
	}`)
	engine := NewBuiltinEngine()
	decoded, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, request)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, decoded, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Messages) != 3 || strings.Contains(string(encoded.Body), "next user message") {
		t.Fatalf("text-only tool result changed shape: %s", encoded.Body)
	}
}

// A turn that ends on tool results, the usual shape of an agent loop, still
// carries their media: it follows as the request's last message.
func TestChatEncodeCarriesMediaFromTheLastToolResult(t *testing.T) {
	request := []byte(`{
		"model":"m","max_tokens":16,
		"messages":[
			{"role":"user","content":"screenshot please"},
			{"role":"assistant","content":[{"type":"tool_use","id":"call_a","name":"shot","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a","content":[
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}
			]}]}
		]
	}`)
	engine := NewBuiltinEngine()
	decoded, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, request)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, decoded, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	last := wire.Messages[len(wire.Messages)-1]
	if len(wire.Messages) != 4 || last.Role != "user" || !strings.Contains(string(last.Content), `"type":"image_url"`) {
		t.Fatalf("messages: %s", encoded.Body)
	}
}
