package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// What a Responses provider returns when the model searched: a
// web_search_call item, then the answer with a url_citation.
const webSearchResponse = `{"id":"resp_1","object":"response","created_at":100,"model":"gpt-5-codex","status":"completed","output":[` +
	`{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"go 1.30 release date","sources":[{"type":"url","url":"https://go.dev/doc/devel/release"}]}},` +
	`{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Go 1.30 shipped in August.",` +
	`"annotations":[{"type":"url_citation","url":"https://go.dev/doc/devel/release","title":"Release History","start_index":0,"end_index":6}]}]}],` +
	`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`

// A Responses client gets the web_search_call item as the provider sent it; a
// Chat client gets the answer, and no failure. (An Anthropic client refuses the
// url_citation under the default lossy policy, as it does for any Responses
// answer with citations; that path is not changed here.)
func TestWebSearchCallReachesAResponsesClient(t *testing.T) {
	responses := translateResponsesResponse(t, webSearchResponse, llmprotocol.OpenAIResponsesV1)
	for _, mark := range []string{`"type":"web_search_call"`, `"id":"ws_1"`, `"query":"go 1.30 release date"`, `"sources":[{"type":"url"`, `"url_citation"`} {
		if !bytes.Contains(responses, []byte(mark)) {
			t.Fatalf("%s did not reach the Responses client: %s", mark, responses)
		}
	}
	for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1} {
		routed := translateResponsesResponse(t, webSearchResponse, client)
		if bytes.Contains(routed, []byte("web_search_call")) || bytes.Contains(routed, []byte("go 1.30 release date")) {
			t.Fatalf("the search item reached a %s client: %s", client, routed)
		}
		if !bytes.Contains(routed, []byte("Go 1.30 shipped in August.")) {
			t.Fatalf("the answer was lost for a %s client: %s", client, routed)
		}
	}
}

func webSearchStream() string {
	frame := func(event string, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	item := `{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"go release history","sources":[{"type":"url","url":"https://go.dev/doc/devel/release"}]}}`
	return frame("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":100,"model":"m","status":"in_progress","output":[]}}`) +
		frame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"in_progress"}}`) +
		frame("response.web_search_call.in_progress", `{"type":"response.web_search_call.in_progress","sequence_number":2,"output_index":0,"item_id":"ws_1"}`) +
		frame("response.web_search_call.searching", `{"type":"response.web_search_call.searching","sequence_number":3,"output_index":0,"item_id":"ws_1"}`) +
		frame("response.web_search_call.completed", `{"type":"response.web_search_call.completed","sequence_number":4,"output_index":0,"item_id":"ws_1"}`) +
		frame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":5,"output_index":0,"item":`+item+`}`) +
		frame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":6,"output_index":1,"item":{"type":"message","id":"msg_1","status":"in_progress","role":"assistant","content":[]}}`) +
		frame("response.content_part.added", `{"type":"response.content_part.added","sequence_number":7,"output_index":1,"item_id":"msg_1","content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}`) +
		frame("response.output_text.delta", `{"type":"response.output_text.delta","sequence_number":8,"output_index":1,"item_id":"msg_1","content_index":0,"delta":"Go releases twice a year."}`) +
		frame("response.output_text.done", `{"type":"response.output_text.done","sequence_number":9,"output_index":1,"item_id":"msg_1","content_index":0,"text":"Go releases twice a year."}`) +
		frame("response.content_part.done", `{"type":"response.content_part.done","sequence_number":10,"output_index":1,"item_id":"msg_1","content_index":0,"part":{"type":"output_text","text":"Go releases twice a year.","annotations":[]}}`) +
		frame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":11,"output_index":1,"item":{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Go releases twice a year.","annotations":[]}]}}`) +
		frame("response.completed", `{"type":"response.completed","sequence_number":12,"response":{"id":"resp_1","object":"response","created_at":100,"model":"m","status":"completed","output":[`+item+`,`+
			`{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Go releases twice a year.","annotations":[]}]}],`+
			`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`)
}

func runWebSearchStream(t *testing.T, client llmprotocol.WireFormat) []byte {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, client, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "m",
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(webSearchStream()))
	if err != nil {
		t.Fatalf("push to %s: %v", client, err)
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize to %s: %v", client, err)
	}
	return bytes.Join(append(frames, final...), nil)
}

// A streamed search reaches a Responses client as the web_search_call item's
// added and done events, with the query and sources on done and in the
// completed response; a Chat or Messages client gets the answer.
func TestWebSearchCallStreams(t *testing.T) {
	responses := runWebSearchStream(t, llmprotocol.OpenAIResponsesV1)
	for _, mark := range []string{`"type":"web_search_call"`, `"query":"go release history"`, "Go releases twice a year."} {
		if !bytes.Contains(responses, []byte(mark)) {
			t.Fatalf("%s did not reach the Responses client:\n%s", mark, responses)
		}
	}
	completed := responses[bytes.Index(responses, []byte("event: response.completed")):]
	if !bytes.Contains(completed, []byte(`"query":"go release history"`)) {
		t.Fatalf("the completed response lost the search item:\n%s", completed)
	}
	for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		routed := runWebSearchStream(t, client)
		if bytes.Contains(routed, []byte("web_search_call")) || bytes.Contains(routed, []byte("go release history")) {
			t.Fatalf("the search reached a %s client:\n%s", client, routed)
		}
		if !bytes.Contains(routed, []byte("Go releases twice a year.")) {
			t.Fatalf("the answer was lost for a %s client:\n%s", client, routed)
		}
		if client == llmprotocol.AnthropicMessagesV1 && bytes.Count(routed, []byte("event: content_block_start")) != 1 {
			t.Fatalf("the search item left a block behind for a Messages client:\n%s", routed)
		}
	}
}

// web_search reaches a Responses arm only when the arm is admitted with it;
// no admission, or another format, drops and counts it.
func TestWebSearchDeclarationNeedsAdmission(t *testing.T) {
	body := `{"model":"m","input":"what shipped in go 1.30?","tools":[{"type":"web_search","external_web_access":false}]}`
	route := func(target llmprotocol.WireFormat, hosted []string) ([]byte, llmprotocol.Diagnostics) {
		engine := NewBuiltinEngine()
		request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
		if err != nil {
			t.Fatal(err)
		}
		request.HostedTools = hosted
		request.Generation++
		result, err := engine.EncodeRequest(target, request, envelope)
		if err != nil {
			t.Fatalf("EncodeRequest(%s) error = %v", target, err)
		}
		return result.Body, result.Diagnostics
	}
	admitted, _ := route(llmprotocol.OpenAIResponsesV1, []string{"web_search"})
	if !bytes.Contains(admitted, []byte(`"tools":[{"type":"web_search","external_web_access":false}]`)) {
		t.Fatalf("an admitted arm did not get the declaration as sent: %s", admitted)
	}
	for name, tc := range map[string]struct {
		target llmprotocol.WireFormat
		hosted []string
	}{
		"Responses arm not admitted": {llmprotocol.OpenAIResponsesV1, nil},
		"Chat arm admitted":          {llmprotocol.OpenAIChatV1, []string{"web_search"}},
	} {
		routed, diagnostics := route(tc.target, tc.hosted)
		if bytes.Contains(routed, []byte("web_search")) {
			t.Fatalf("%s: the declaration was sent: %s", name, routed)
		}
		if !strings.Contains(strings.Join(droppedFields(diagnostics), ","), "tools.web_search") {
			t.Fatalf("%s: the drop was not counted", name)
		}
	}
}

// A buffered searched answer, as the response cache holds it, replays as a
// Responses stream with its web_search_call item.
func TestCachedSearchedAnswerReplaysAsAStream(t *testing.T) {
	engine := NewBuiltinEngine()
	response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, []byte(webSearchResponse))
	if err != nil {
		t.Fatal(err)
	}
	wire, _, err := engine.EncodeResponseStream(llmprotocol.OpenAIResponsesV1, response, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if !bytes.Contains(wire, []byte(`"type":"web_search_call"`)) || !bytes.Contains(wire, []byte("Go 1.30 shipped in August.")) {
		t.Fatalf("the replayed stream lost the search item or the answer:\n%s", wire)
	}
}

// A store:false client resends the search item on its next turn: a Responses
// arm gets it back whole, and any other arm drops it with the conversation
// intact.
func TestResentWebSearchCallRoutes(t *testing.T) {
	body := `{"model":"m","input":[{"type":"message","role":"user","content":"q"},` +
		`{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"go release history"}},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Go releases twice a year.","annotations":[]}]},` +
		`{"type":"message","role":"user","content":"and the next one?"}]}`
	responses := routeResponsesRequest(t, body, llmprotocol.OpenAIResponsesV1)
	if !bytes.Contains(responses, []byte(`"query":"go release history"`)) {
		t.Fatalf("the resent search item did not reach a Responses arm: %s", responses)
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		routed := routeResponsesRequest(t, body, target)
		if bytes.Contains(routed, []byte("web_search_call")) || !bytes.Contains(routed, []byte("and the next one?")) {
			t.Fatalf("%s: %s", target, routed)
		}
	}
}

// A web search that failed, or a snapshot taken mid-search, is carried like a
// completed one: its status is the web search's own.
func TestWebSearchCallStatusesAreCarried(t *testing.T) {
	for _, status := range []string{"failed", "searching", "in_progress"} {
		body := strings.Replace(webSearchResponse, `"id":"ws_1","status":"completed"`, `"id":"ws_1","status":"`+status+`"`, 1)
		responses := translateResponsesResponse(t, body, llmprotocol.OpenAIResponsesV1)
		if !bytes.Contains(responses, []byte(`"status":"`+status+`"`)) {
			t.Fatalf("status %s was not carried: %s", status, responses)
		}
	}
}

// A failed web search may end the item it started, and only that item: an
// item started as a message may not complete as a failed search.
func TestFailedWebSearchStillMatchesItsItem(t *testing.T) {
	frame := func(event string, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
	body := frame("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":100,"model":"m","status":"in_progress","output":[]}}`) +
		frame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"ws_1","status":"in_progress","role":"assistant","content":[]}}`) +
		frame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"failed"}}`)
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "m",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := stream.Push([]byte(body)); err == nil || !strings.Contains(err.Error(), "stream_item_kind_mismatch") {
		t.Fatalf("a message completed as a failed search returned %v", err)
	}
}
