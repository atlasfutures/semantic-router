package extproc

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// semantic-router #222: a provider's error names the Router's provider
// account (OpenRouter's 402 links the key's settings page). No path may hand
// the client the provider's words; each hands it the public form.
const openRouter402Message = "This request requires more credits, or fewer max_tokens. You requested up to 131072 tokens, " +
	"but can only afford 54095. To increase, visit https://openrouter.ai/settings/keys/0123abcd and create a key with a higher limit"

func assertNoProviderAccountText(t *testing.T, body []byte) {
	t.Helper()
	for _, leaked := range []string{"openrouter.ai", "54095", "0123abcd"} {
		if bytes.Contains(body, []byte(leaked)) {
			t.Fatalf("client body leaks %q: %s", leaked, body)
		}
	}
}

func TestUpstreamTransportErrorHidesProviderAccountRefusal(t *testing.T) {
	router := &OpenAIRouter{}
	upstream := []byte(`{"error":{"message":"` + openRouter402Message + `","code":402,"metadata":{"provider_name":null}}}`)
	for _, target := range extProcMatrixFormats {
		t.Run(string(target), func(t *testing.T) {
			ctx := &RequestContext{
				SourceFormat: target, TargetFormat: llmprotocol.OpenAIChatV1,
				UpstreamStatusCode: 402, RequestID: "request_402", RequestModel: "public-model",
			}
			response := router.handleUpstreamTransportError(upstream, ctx)
			body := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
			assertNoProviderAccountText(t, body)
			if !bytes.Contains(body, []byte("model service unavailable")) {
				t.Fatalf("client body = %s", body)
			}
		})
	}
}

// The client keeps an actionable message, without links; the failure is
// still classed by what the provider said.
func TestUpstreamTransportErrorRedactsLinksAndClassesByProviderText(t *testing.T) {
	router := &OpenAIRouter{}
	upstream := []byte(`{"error":{"type":"invalid_request_error","code":"context_length_exceeded",` +
		`"message":"This endpoint's maximum context length is 200000 tokens. See https://openrouter.ai/docs/limits"}}`)
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		UpstreamStatusCode: 400, RequestID: "request_400", RequestModel: "public-model",
	}
	response := router.handleUpstreamTransportError(upstream, ctx)
	body := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	if bytes.Contains(body, []byte("openrouter.ai")) ||
		!bytes.Contains(body, []byte("maximum context length is 200000 tokens")) ||
		!bytes.Contains(body, []byte("context_length_exceeded")) {
		t.Fatalf("client body = %s", body)
	}
	if ctx.ResponseFailureClass != turnFailureContextOverflow {
		t.Fatalf("class = %q, want %q", ctx.ResponseFailureClass, turnFailureContextOverflow)
	}
}

func streamErrorContext(source, target llmprotocol.WireFormat) *RequestContext {
	return &RequestContext{
		RequestID: "request_stream", SourceFormat: source, TargetFormat: target,
		RequestModel: "public-model", TraceContext: context.Background(),
		SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public-model", Stream: true},
	}
}

func TestStreamedProviderErrorReachesTheClientInPublicForm(t *testing.T) {
	chatError := []byte("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\"," +
		"\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[]," +
		"\"error\":{\"type\":\"server_error\",\"code\":\"402\",\"message\":\"" + openRouter402Message + "\"}}\n\n")
	anthropicError := []byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\"," +
		"\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
		"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"billing_error\",\"message\":\"" + openRouter402Message + "\"}}\n\n")
	tests := []struct {
		name           string
		source, target llmprotocol.WireFormat
		upstream       []byte
	}{
		{"chat from chat (filtered passthrough)", llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, chatError},
		{"messages from messages (passthrough)", llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, anthropicError},
		{
			"messages from messages, error typed by event name", llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
			bytes.Replace(anthropicError, []byte(`{"type":"error","error"`), []byte(`{"error"`), 1),
		},
		{"messages from chat (re-encoded)", llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, chatError},
		{"chat from messages (re-encoded)", llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, anthropicError},
		{"responses from chat (re-encoded)", llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, chatError},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// The frames arrive before the end of the stream, as Envoy
			// delivers them: the chunk that carries the error is not the one
			// that finalizes the stream.
			ctx := streamErrorContext(test.source, test.target)
			router := &OpenAIRouter{}
			var client []byte
			for _, chunk := range []struct {
				body []byte
				end  bool
			}{{test.upstream, false}, {nil, true}} {
				response := router.handleSemanticStreamingResponseBody(chunk.body, chunk.end, ctx)
				mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
				if mutation == nil {
					if !chunk.end {
						t.Fatal("the provider's stream reached the client unchanged")
					}
					continue
				}
				client = append(client, mutation.GetBody()...)
			}
			assertNoProviderAccountText(t, client)
			if !bytes.Contains(client, []byte("model service unavailable")) {
				t.Fatalf("client stream = %s", client)
			}
		})
	}
}

// A same-format Messages stream without an error still travels byte for byte.
func TestSameFormatMessagesStreamIsPassedThrough(t *testing.T) {
	upstream := extProcStreamFixture(llmprotocol.AnthropicMessagesV1)
	ctx := streamErrorContext(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1)
	response := (&OpenAIRouter{}).handleSemanticStreamingResponseBody(upstream, true, ctx)
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil &&
		!bytes.Equal(mutation.GetBody(), upstream) {
		t.Fatalf("a same-format stream was rewritten:\n got %q\nwant %q", mutation.GetBody(), upstream)
	}
}

func TestBufferedProviderErrorBodyReachesTheClientInPublicForm(t *testing.T) {
	_, router, decision := statusCacheRouter()
	ctx := withSelectedDecision(&RequestContext{
		RequestID: "req-200-error", RequestModel: "test", RequestQuery: "hello",
		SemanticRequest: testNeutralRequest("test", "hello"),
		SourceFormat:    llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		TraceContext: context.Background(), UpstreamStatusCode: 200,
	}, decision)
	body := `{"id":"gen-1","object":"chat.completion","created":1,"model":"test",` +
		`"error":{"type":"server_error","code":"402","message":"` + openRouter402Message + `"},"choices":[]}`
	response := router.handleNonStreamingResponseBody([]byte(body), ctx, time.Second)
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatalf("the provider's error body reached the client unchanged: %v", response)
	}
	assertNoProviderAccountText(t, mutation.GetBody())
}
