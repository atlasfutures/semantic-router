package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// What Anthropic returns when the model searched: the server_tool_use block,
// the results, and an answer citing one of them.
const anthropicWebSearchResponse = `{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[` +
	`{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"go release history"}},` +
	`{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://go.dev/doc/devel/release","title":"Release History","encrypted_content":"abc","page_age":null}]},` +
	`{"type":"text","text":"Go releases twice a year.","citations":[{"type":"web_search_result_location","url":"https://go.dev/doc/devel/release","title":"Release History","encrypted_index":"xyz","cited_text":"twice a year"}]}],` +
	`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1,"server_tool_use":{"web_search_requests":1}}}`

func translateAnthropicResponse(t *testing.T, body string, client llmprotocol.WireFormat) []byte {
	t.Helper()
	engine := NewBuiltinEngine()
	response, envelope, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(body))
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

// An Anthropic client (Claude Code) gets its search blocks and citations back
// as Anthropic sent them; before this, the response failed to decode.
func TestAnthropicWebSearchRoundTripsForAnAnthropicClient(t *testing.T) {
	routed := translateAnthropicResponse(t, anthropicWebSearchResponse, llmprotocol.AnthropicMessagesV1)
	for _, mark := range []string{`"type":"server_tool_use"`, `"query":"go release history"`, `"type":"web_search_tool_result"`,
		`"encrypted_content":"abc"`, `"encrypted_index":"xyz"`, "Go releases twice a year."} {
		if !bytes.Contains(routed, []byte(mark)) {
			t.Fatalf("%s did not reach the Anthropic client: %s", mark, routed)
		}
	}
}

// A Responses client (Codex) gets one web_search_call with the query and the
// result's source, and the answer with a url_citation.
func TestAnthropicWebSearchBecomesAResponsesWebSearchCall(t *testing.T) {
	routed := translateAnthropicResponse(t, anthropicWebSearchResponse, llmprotocol.OpenAIResponsesV1)
	for _, mark := range []string{`"type":"web_search_call"`, `"id":"srvtoolu_1"`, `"query":"go release history"`,
		`"url":"https://go.dev/doc/devel/release"`, `"type":"url_citation"`, "Go releases twice a year."} {
		if !bytes.Contains(routed, []byte(mark)) {
			t.Fatalf("%s did not reach the Responses client: %s", mark, routed)
		}
	}
	if bytes.Contains(routed, []byte("server_tool_use")) || bytes.Contains(routed, []byte("encrypted_content")) {
		t.Fatalf("Anthropic blocks reached the Responses client: %s", routed)
	}
}

// A Chat client gets the answer, without the search blocks.
func TestAnthropicWebSearchChatClientGetsTheAnswer(t *testing.T) {
	routed := translateAnthropicResponse(t, anthropicWebSearchResponse, llmprotocol.OpenAIChatV1)
	if !bytes.Contains(routed, []byte("Go releases twice a year.")) || bytes.Contains(routed, []byte("server_tool_use")) {
		t.Fatalf("chat client got %s", routed)
	}
}

func anthropicWebSearchStream() string {
	frame := func(event, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	return frame("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}`) +
		frame("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`) +
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\": \"go release"}}`) +
		frame("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" history\"}"}}`) +
		frame("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		frame("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://go.dev/doc/devel/release","title":"Release History","encrypted_content":"abc","page_age":null}]}}`) +
		frame("content_block_stop", `{"type":"content_block_stop","index":1}`) +
		frame("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`) +
		frame("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Go releases twice a year."}}`) +
		frame("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","url":"https://go.dev/doc/devel/release","title":"Release History","encrypted_index":"xyz","cited_text":"twice a year"}}}`) +
		frame("content_block_stop", `{"type":"content_block_stop","index":2}`) +
		frame("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":20,"server_tool_use":{"web_search_requests":1}}}`) +
		frame("message_stop", `{"type":"message_stop"}`)
}

func runAnthropicWebSearchStream(t *testing.T, client llmprotocol.WireFormat) []byte {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, client, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "claude",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(anthropicWebSearchStream()))
	if err != nil {
		t.Fatalf("push to %s: %v", client, err)
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize to %s: %v", client, err)
	}
	return bytes.Join(append(frames, final...), nil)
}

// A streamed Anthropic search reaches Claude Code as the provider streamed it,
// Codex as a web_search_call with its query and sources plus the cited answer,
// and a Chat client as the answer.
func TestAnthropicWebSearchStreams(t *testing.T) {
	anthropic := runAnthropicWebSearchStream(t, llmprotocol.AnthropicMessagesV1)
	for _, mark := range []string{`"type":"server_tool_use"`, `"partial_json":"{\"query\":\"go release history\"}"`,
		`"type":"web_search_tool_result"`, `"encrypted_content":"abc"`, `"type":"citations_delta"`, `"encrypted_index":"xyz"`, "Go releases twice a year."} {
		if !bytes.Contains(anthropic, []byte(mark)) {
			t.Fatalf("%s did not reach the Anthropic client:\n%s", mark, anthropic)
		}
	}
	responses := runAnthropicWebSearchStream(t, llmprotocol.OpenAIResponsesV1)
	for _, mark := range []string{`"type":"web_search_call"`, `"query":"go release history"`, `"url":"https://go.dev/doc/devel/release"`,
		`"url_citation"`, "Go releases twice a year."} {
		if !bytes.Contains(responses, []byte(mark)) {
			t.Fatalf("%s did not reach the Responses client:\n%s", mark, responses)
		}
	}
	chat := runAnthropicWebSearchStream(t, llmprotocol.OpenAIChatV1)
	if bytes.Contains(chat, []byte("server_tool_use")) || !bytes.Contains(chat, []byte("Go releases twice a year.")) {
		t.Fatalf("chat client got:\n%s", chat)
	}
}

// Codex's web_search declaration reaches an admitted Anthropic arm as
// Anthropic's web search tool; an arm not admitted gets nothing.
func TestWebSearchDeclarationMapsToAnthropicsTool(t *testing.T) {
	body := `{"model":"m","input":"what shipped in go 1.30?","tools":[{"type":"web_search"}]}`
	route := func(hosted []string) ([]byte, string) {
		engine := NewBuiltinEngine()
		request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		request.HostedTools = hosted
		request.Generation++
		result, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
		if err != nil {
			t.Fatal(err)
		}
		return result.Body, strings.Join(droppedFields(result.Diagnostics), ",")
	}
	admitted, dropped := route([]string{"web_search"})
	if !bytes.Contains(admitted, []byte(`"tools":[{"type":"web_search_20250305","name":"web_search"}]`)) || strings.Contains(dropped, "tools.web_search") {
		t.Fatalf("an admitted Anthropic arm got %s (dropped: %s)", admitted, dropped)
	}
	plain, dropped := route(nil)
	if bytes.Contains(plain, []byte("web_search")) || !strings.Contains(dropped, "tools.web_search") {
		t.Fatalf("an arm not admitted got %s (dropped: %s)", plain, dropped)
	}
}
