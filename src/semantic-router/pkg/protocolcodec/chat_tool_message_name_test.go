package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Responses history whose call came from a Claude worker (a toolu_ id),
// sent on to a Chat worker. Kimi on OpenRouter refuses a tool message it
// cannot match to its call, so with NamesToolResults the tool message names
// the tool; without it the body has no name, as OpenAI's Chat schema expects.
const crossModelResponsesHistory = `{
	"model":"m",
	"input":[
		{"type":"message","role":"user","content":"list the files"},
		{"type":"function_call","call_id":"toolu_01AbCdEf","name":"shell","arguments":"{\"cmd\":\"ls\"}"},
		{"type":"function_call_output","call_id":"toolu_01AbCdEf","output":"a.txt"}
	]
}`

type chatToolMessageWire struct {
	Role       string          `json:"role"`
	ToolCallID string          `json:"tool_call_id"`
	Name       json.RawMessage `json:"name"`
	ToolCalls  []struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	} `json:"tool_calls"`
}

func encodeChatToolMessages(t *testing.T, request llmprotocol.Request) ([]chatToolMessageWire, llmprotocol.Diagnostics) {
	t.Helper()
	encoded, err := NewBuiltinEngine().EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var wire struct {
		Messages []chatToolMessageWire `json:"messages"`
	}
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	return wire.Messages, encoded.Diagnostics
}

func toolMessageOf(t *testing.T, messages []chatToolMessageWire) chatToolMessageWire {
	t.Helper()
	for _, message := range messages {
		if message.Role == "tool" {
			return message
		}
	}
	t.Fatalf("no tool message in %+v", messages)
	return chatToolMessageWire{}
}

func TestChatToolMessageCarriesTheCallsToolNameWhenAsked(t *testing.T) {
	decoded, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(crossModelResponsesHistory))
	if err != nil {
		t.Fatal(err)
	}
	decoded.NamesToolResults = true
	messages, _ := encodeChatToolMessages(t, decoded)
	tool := toolMessageOf(t, messages)
	if tool.ToolCallID != "toolu_01AbCdEf" || string(tool.Name) != `"shell"` {
		t.Fatalf("tool message = %+v (name %s), want name \"shell\"", tool, tool.Name)
	}
}

func TestChatToolMessageCarriesNoNameByDefault(t *testing.T) {
	decoded, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(crossModelResponsesHistory))
	if err != nil {
		t.Fatal(err)
	}
	messages, _ := encodeChatToolMessages(t, decoded)
	if tool := toolMessageOf(t, messages); tool.Name != nil {
		t.Fatalf("tool message name = %s, want none", tool.Name)
	}
}

// A namespaced call's tool message names it the way the call's function.name
// is written, so the two match.
func TestChatToolMessageNameIsQualifiedLikeTheCall(t *testing.T) {
	request := llmprotocol.Request{
		Model:            "m",
		Generation:       1,
		NamesToolResults: true,
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "spawn"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
				Kind:     llmprotocol.ContentToolCall,
				ToolCall: &llmprotocol.ToolCall{ID: "call_ns", Namespace: "multi_agent", Name: "spawn", Arguments: "{}"},
			}}},
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{
					CallID: "call_ns", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "ok"}},
				},
			}}},
		},
	}
	messages, _ := encodeChatToolMessages(t, request)
	callName := messages[1].ToolCalls[0].Function.Name
	want, _ := json.Marshal(llmprotocol.QualifiedToolName("multi_agent", "spawn"))
	if tool := toolMessageOf(t, messages); string(tool.Name) != string(want) || callName != llmprotocol.QualifiedToolName("multi_agent", "spawn") {
		t.Fatalf("tool message name = %s, call function.name = %q, want both %s", tool.Name, callName, want)
	}
}

// A result whose call is not in the request (a truncated history) goes
// without a name, and the omission is counted rather than refused.
func TestChatToolMessageWithoutItsCallIsSentUnnamedAndCounted(t *testing.T) {
	request := llmprotocol.Request{
		Model:            "m",
		Generation:       1,
		NamesToolResults: true,
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
				Kind:     llmprotocol.ContentToolCall,
				ToolCall: &llmprotocol.ToolCall{ID: "call_known", Name: "shell", Arguments: "{}"},
			}}},
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{
					CallID: "call_lost", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "ok"}},
				},
			}}},
		},
	}
	engine := NewBuiltinEngine()
	body, diagnostics, err := OpenAIChatCodec{}.EncodeRequest(request, llmprotocol.Envelope{}, engine.requestEncodePolicy())
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var wire struct {
		Messages []chatToolMessageWire `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if tool := toolMessageOf(t, wire.Messages); tool.Name != nil {
		t.Fatalf("unmatched tool message name = %s, want none", tool.Name)
	}
	counted := false
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "messages.tool.name" && diagnostic.Action == llmprotocol.DiagnosticDropped {
			counted = true
		}
	}
	if !counted {
		t.Fatalf("diagnostics = %+v, want a messages.tool.name drop", diagnostics)
	}
}
