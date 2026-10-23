package execution

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
)

func TestExecutionRestoresNamespacedToolCallsBufferedAndStreamed(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			body := `{"id":"chat1","object":"chat.completion","created":1,"model":"qwen-fixture","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"multi_agent_v1__spawn_agent","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}`
			engine := protocolcodec.NewBuiltinEngine()
			wire := []byte(body)
			if stream {
				response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, wire)
				if err != nil {
					t.Fatal(err)
				}
				wire, _, err = engine.EncodeResponseStream(llmprotocol.OpenAIChatV1, response, llmprotocol.StreamContext{})
				if err != nil {
					t.Fatal(err)
				}
			}
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write(wire) }))
			defer upstream.Close()
			_, executor := executorWithHTTPDecisions(t, testConfig(t, upstream.URL, false), &decisionCapture{})
			request := fmt.Sprintf(`{"model":"arc-1.0","stream":%v,"input":[{"role":"user","content":"spawn helper"}],"tools":[{"type":"namespace","name":"multi_agent_v1","tools":[{"type":"function","name":"spawn_agent","parameters":{"type":"object"}}]}]}`, stream)
			status, output := executeHTTP(t, executor.URL, "/v1/responses", request, "namespaces")
			if status != 200 {
				t.Fatalf("status %d: %s", status, output)
			}
			if !bytes.Contains(output, []byte(`"namespace":"multi_agent_v1"`)) || !bytes.Contains(output, []byte(`"name":"spawn_agent"`)) || bytes.Contains(output, []byte("multi_agent_v1__spawn_agent")) {
				t.Fatalf("namespace lost on returned call: %s", output)
			}
			if !bytes.Contains(output, []byte(`"model":"arc-1.0"`)) || bytes.Contains(output, []byte(`"model":"qwen-fixture"`)) {
				t.Fatalf("public model alias changed: %s", output)
			}
		})
	}
}

func TestExecutionCollectsBackendUsageWhileHonoringClientStreamPreference(t *testing.T) {
	for _, include := range []bool{false, true} {
		t.Run(fmt.Sprintf("include=%v", include), func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				if !bytes.Contains(body, []byte(`"include_usage":true`)) {
					t.Errorf("backend usage was not requested: %s", body)
				}
				_, _ = io.WriteString(w, "data: {\"id\":\"chat1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen-fixture\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ready\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chat1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen-fixture\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":1,\"total_tokens\":21}}\n\ndata: [DONE]\n\n")
			}))
			defer upstream.Close()
			server, executor := executorWithHTTPDecisions(t, testConfig(t, upstream.URL, false), &decisionCapture{})
			request := fmt.Sprintf(`{"model":"arc-1.0","stream":true,"stream_options":{"include_usage":%v},"messages":[{"role":"user","content":"hello"}]}`, include)
			status, output := executeHTTP(t, executor.URL, "/v1/chat/completions", request, "usage")
			if status != 200 || bytes.Contains(output, []byte(`"usage":`)) != include {
				t.Fatalf("client usage preference changed: status %d, %s", status, output)
			}
			state, err := server.states.acquire(t.Context(), "usage", "main")
			if err != nil {
				t.Fatal(err)
			}
			defer state.release()
			if state.workers["fixture-worker"].Call.PromptTokens == nil {
				t.Fatal("hidden backend usage did not reach cache evidence")
			}
		})
	}
}

func TestRefusedExecutionIsDeliveredButDoesNotCommit(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			body := []byte(`{"id":"chat1","object":"chat.completion","created":1,"model":"qwen-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"","refusal":"Cannot answer."},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}`)
			if stream {
				engine := protocolcodec.NewBuiltinEngine()
				response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, body)
				if err != nil {
					t.Fatal(err)
				}
				body, _, err = engine.EncodeResponseStream(llmprotocol.OpenAIChatV1, response, llmprotocol.StreamContext{})
				if err != nil {
					t.Fatal(err)
				}
			}
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write(body) }))
			defer upstream.Close()
			capture := &decisionCapture{}
			server, executor := executorWithHTTPDecisions(t, testConfig(t, upstream.URL, false), capture)
			request := fmt.Sprintf(`{"model":"arc-1.0","stream":%v,"messages":[{"role":"user","content":"hello"}]}`, stream)
			for range 2 {
				status, output := executeHTTP(t, executor.URL, "/v1/chat/completions", request, "refusal")
				if status != 200 || !bytes.Contains(output, []byte("Cannot answer.")) {
					t.Fatalf("refusal was not delivered: %d %s", status, output)
				}
			}
			state, err := server.states.acquire(t.Context(), "refusal", "main")
			if err != nil {
				t.Fatal(err)
			}
			defer state.release()
			if state.completed != 0 || len(state.workers) != 0 || len(state.placements) != 0 || state.held != "" || len(state.ledger) != 0 || state.pending != nil {
				t.Fatal("refusal committed execution success")
			}
			capture.mu.Lock()
			defer capture.mu.Unlock()
			if len(capture.contexts) != 2 {
				t.Fatal("explicit retry retained the refused decision")
			}
		})
	}
}

func TestClaudeWorkerStreamsReasoningBeforeItsSignature(t *testing.T) {
	body := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"claude\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"consider\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-claude\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	server := &Server{engine: protocolcodec.NewBuiltinEngine()}
	deployment := Deployment{RegistryModel: "claude-opus-4-6", Protocol: llmprotocol.AnthropicMessagesV1}
	metadata := responseMetadataFor(llmprotocol.Request{Model: "public"}, deployment)
	writer := httptest.NewRecorder()
	upstream := &http.Response{Body: io.NopCloser(strings.NewReader(body))}
	_, err := server.forwardStream(writer, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil), upstream, llmprotocol.OpenAIChatV1, admittedAction{deployment: deployment}, metadata, &pendingTurn{})
	if err != nil {
		t.Fatal(err)
	}
	var first string
	for _, line := range strings.Split(writer.Body.String(), "\n") {
		if strings.Contains(line, "reasoning_details") {
			first = line
			break
		}
	}
	if !strings.Contains(first, "consider") || strings.Contains(first, "sig-claude") {
		t.Fatalf("Claude reasoning waited for its signature: %s", writer.Body.String())
	}
	if !strings.Contains(writer.Body.String(), "sig-claude") {
		t.Fatal("final signature was lost")
	}
}
