package execution

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// codexNamespacedTurn is a Codex-shaped Responses request: a plain function
// beside a namespace of sub-agent functions, with an earlier namespaced call
// in the history.
const codexNamespacedTurn = `{"model":"arc-1.0","max_output_tokens":64,"store":false,"stream":%STREAM%,"input":[` +
	`{"role":"user","content":"spawn a helper"},` +
	`{"type":"function_call","call_id":"call_old","namespace":"multi_agent_v1","name":"spawn_agent","arguments":"{}"},` +
	`{"type":"function_call_output","call_id":"call_old","output":"agent-7"},` +
	`{"role":"user","content":"now close it"}],` +
	`"tools":[{"type":"function","name":"exec_command","description":"Run a command.","parameters":{"type":"object"}},` +
	`{"type":"namespace","name":"multi_agent_v1","description":"Sub-agents.","tools":[` +
	`{"type":"function","name":"spawn_agent","description":"Spawn one.","parameters":{"type":"object"}},` +
	`{"type":"function","name":"close_agent","parameters":{"type":"object"}}]}]}`

func namespacedTurn(stream bool) string {
	return strings.Replace(codexNamespacedTurn, "%STREAM%", map[bool]string{true: "true", false: "false"}[stream], 1)
}

// functionOnlyServer answers like a user's Responses server that serves
// function tools only: a namespace tool is refused, and the model calls the
// function named by call.
func functionOnlyServer(t *testing.T, call string, sent *[]string, mu *sync.Mutex) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := string(raw)
		mu.Lock()
		*sent = append(*sent, body)
		mu.Unlock()
		if strings.Contains(body, `"namespace"`) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"type":"invalid_request_error","code":null,"message":"tools of type 'namespace' are not supported: this server runs function tools only","param":null}}`)
			return
		}
		item := `{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"` + call + `","arguments":"{}"}`
		if !strings.Contains(body, `"stream":true`) {
			_, _ = io.WriteString(w, `{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"qwen-fixture","output":[`+item+`],"usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21}}`)
			return
		}
		frame := func(event, data string) string { return "event: " + event + "\ndata: " + data + "\n\n" }
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w,
			frame("response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":1,"model":"qwen-fixture","status":"in_progress","output":[]}}`)+
				frame("response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"function_call","id":"fc_1","status":"in_progress","call_id":"call_1","name":"`+call+`","arguments":""}}`)+
				frame("response.function_call_arguments.done", `{"type":"response.function_call_arguments.done","sequence_number":2,"output_index":0,"item_id":"fc_1","name":"`+call+`","arguments":"{}"}`)+
				frame("response.output_item.done", `{"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":`+item+`}`)+
				frame("response.completed", `{"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","object":"response","created_at":1,"model":"qwen-fixture","status":"completed","output":[`+item+`],"usage":{"input_tokens":20,"output_tokens":1,"total_tokens":21}}}`))
	}))
}

// A Responses request for a local deployment declares namespaced functions
// as plain ones under their qualified names, and the call the model makes
// under such a name reaches the client with the namespace Codex declared.
// A call under a name the request never declared passes through unchanged.
func TestLocalResponsesFlattensToolNamespaces(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, tc := range []struct{ returned, client string }{
			{"multi_agent_v1__close_agent", `"name":"close_agent","namespace":"multi_agent_v1"`},
			{"exec_command", `"name":"exec_command"`},
			{"other__tool", `"name":"other__tool"`},
		} {
			t.Run(map[bool]string{true: "stream", false: "buffered"}[stream]+"/"+tc.returned, func(t *testing.T) {
				var mu sync.Mutex
				var sent []string
				upstream := functionOnlyServer(t, tc.returned, &sent, &mu)
				defer upstream.Close()
				_, server := executorWithHTTPDecisions(t, onProtocol(t, testConfig(t, upstream.URL, false), llmprotocol.OpenAIResponsesV1), &decisionCapture{})
				status, result := executeHTTP(t, server.URL, "/v1/responses", namespacedTurn(stream), "namespaces-"+tc.returned)
				if status != 200 {
					t.Fatalf("flattened turn returned %d: %s", status, result)
				}
				mu.Lock()
				body := sent[0]
				mu.Unlock()
				for _, want := range []string{
					`"name":"multi_agent_v1__spawn_agent"`, `"name":"multi_agent_v1__close_agent"`, `"name":"exec_command"`,
					`"Sub-agents.\n\nSpawn one."`,
				} {
					if !strings.Contains(body, want) {
						t.Errorf("upstream request lacks %s: %s", want, body)
					}
				}
				if strings.Count(body, `"multi_agent_v1__spawn_agent"`) != 2 {
					t.Errorf("the namespaced call in history was not flattened: %s", body)
				}
				if !strings.Contains(string(result), tc.client) {
					t.Errorf("client got the call without %s: %s", tc.client, result)
				}
				if tc.returned != "multi_agent_v1__close_agent" && strings.Contains(string(result), `"namespace"`) {
					t.Errorf("an undeclared or plain call gained a namespace: %s", result)
				}
			})
		}
	}
}

// A plain tool already named like a flattened one cannot be told apart from
// it: decoding refuses the request for every target, so nothing is sent.
func TestFlattenedToolNameCollisionIsRefused(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("a request with colliding tool names reached the deployment")
	}))
	defer upstream.Close()
	_, server := executorWithHTTPDecisions(t, onProtocol(t, testConfig(t, upstream.URL, false), llmprotocol.OpenAIResponsesV1), &decisionCapture{})
	status, result := executeHTTP(t, server.URL, "/v1/responses",
		`{"model":"arc-1.0","input":"hi","tools":[{"type":"function","name":"multi_agent_v1__spawn_agent","parameters":{"type":"object"}},`+
			`{"type":"namespace","name":"multi_agent_v1","tools":[{"type":"function","name":"spawn_agent","parameters":{"type":"object"}}]}]}`, "namespace-collision")
	if status != http.StatusBadRequest || !strings.Contains(string(result), "multi_agent_v1__spawn_agent") {
		t.Fatalf("a colliding qualified name returned %d: %s", status, result)
	}
}

// Rayline Platform serves namespace tools: a cloud deployment and every other
// protocol keep the request as the client sent it.
func TestOnlyLocalResponsesDeploymentsFlatten(t *testing.T) {
	for _, deployment := range []Deployment{
		{Kind: "cloud", Protocol: llmprotocol.OpenAIResponsesV1},
		{Kind: "local", Protocol: llmprotocol.OpenAIChatV1},
		{Kind: "local", Protocol: llmprotocol.AnthropicMessagesV1},
	} {
		if flattensToolNamespaces(deployment) {
			t.Fatalf("%+v flattens namespaces", deployment)
		}
	}
	if !flattensToolNamespaces(Deployment{Kind: "local", Protocol: llmprotocol.OpenAIResponsesV1}) {
		t.Fatal("a local Responses deployment keeps namespace tools")
	}
}
