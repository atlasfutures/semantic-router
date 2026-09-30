package extproc

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func namespacedToolRequest() *llmprotocol.Request {
	return &llmprotocol.Request{Tools: []llmprotocol.Tool{
		{Name: "exec_command", InputSchema: []byte(`{"type":"object"}`)},
		{Name: "spawn_agent", Namespace: "multi_agent_v1", InputSchema: []byte(`{"type":"object"}`)},
	}}
}

// A Chat or Messages provider calls a Codex sub-agent tool by its qualified
// name; the Responses client gets the call back with its namespace.
func TestNamespacedCallComesBackWithItsNamespace(t *testing.T) {
	providers := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,` +
			`"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"multi_agent_v1__spawn_agent","arguments":"{}"}}]},` +
			`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		llmprotocol.AnthropicMessagesV1: `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[` +
			`{"type":"tool_use","id":"call_1","name":"multi_agent_v1__spawn_agent","input":{}}],` +
			`"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`,
	}
	for backend, body := range providers {
		t.Run(string(backend), func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := &RequestContext{
				SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: backend,
				SemanticRequest: namespacedToolRequest(),
			}
			response, err := router.decodeClientResponse([]byte(body), ctx)
			if err != nil {
				t.Fatal(err)
			}
			client, err := router.encodeClientResponse(*response, ctx)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(client, []byte(`"name":"spawn_agent","namespace":"multi_agent_v1"`)) {
				t.Fatalf("the call did not come back with its namespace: %s", client)
			}
		})
	}
}

func TestNamespacedStreamedCallComesBackWithItsNamespace(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "public-model", TraceContext: t.Context(),
		SemanticRequest: namespacedToolRequest(),
	}
	if err := router.ensureSemanticResponseStream(ctx); err != nil {
		t.Fatal(err)
	}
	chunk := func(data string) string { return "data: " + data + "\n\n" }
	stream := chunk(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant",`+
		`"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"multi_agent_v1__spawn_agent","arguments":""}}]},"finish_reason":null}]}`) +
		chunk(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{`+
			`"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":null}]}`) +
		chunk(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`) +
		chunk(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`) +
		"data: [DONE]\n\n"
	wire := pushExtProcStreamFixture(t, ctx, []byte(stream))
	client := wire.Bytes()
	if !bytes.Contains(client, []byte(`"name":"spawn_agent","namespace":"multi_agent_v1"`)) {
		t.Fatalf("the streamed call did not come back with its namespace: %s", client)
	}
	if bytes.Contains(client, []byte("multi_agent_v1__spawn_agent")) {
		t.Fatalf("the qualified name reached the client: %s", client)
	}
}
