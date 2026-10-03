package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Round-2 harness testing (2026-10-03) asked whether reasoning_tokens reaches
// the cell's usage line and the client on a thinking Chat arm. These tests run
// the recorded OpenRouter mimo captures (reasoning_tokens 13 in
// usage.completion_tokens_details) through the cell's own buffered and
// streamed paths, for every client protocol, and read the count back from the
// llm_usage line and from the body the client receives.

const openRouterMimoReasoningTokens = 13

func openRouterCapture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("..", "protocolcodec", "testdata", "provider", "openrouter", name))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// clientReasoningCount is the reasoning member each client protocol carries in
// its usage: Messages output_tokens_details.thinking_tokens, Responses
// output_tokens_details.reasoning_tokens, Chat
// completion_tokens_details.reasoning_tokens.
func clientReasoningCount(format llmprotocol.WireFormat) string {
	if format == llmprotocol.AnthropicMessagesV1 {
		return `"thinking_tokens":13`
	}
	return `"reasoning_tokens":13`
}

func assertUsageLineReasoning(t *testing.T, fields map[string]interface{}) {
	t.Helper()
	if got, _ := fields["reasoning_tokens"].(int64); got != openRouterMimoReasoningTokens {
		t.Fatalf("llm_usage reasoning_tokens = %v, want %d from usage.completion_tokens_details",
			fields["reasoning_tokens"], openRouterMimoReasoningTokens)
	}
}

func TestOpenRouterReasoningTokensReachTheBufferedUsageLineAndClient(t *testing.T) {
	for _, client := range extProcMatrixFormats {
		t.Run(string(client), func(t *testing.T) {
			logs := captureLogs(t)
			router := &OpenAIRouter{}
			ctx := &RequestContext{
				RequestID: "rt_reasoning-buffered", RequestModel: "xiaomi/mimo-v2.5-pro@thinking-on",
				SourceFormat: client, TargetFormat: llmprotocol.OpenAIChatV1,
			}
			semantic, err := router.decodeClientResponse(openRouterCapture(t, "chat-response-reasoning.json"), ctx)
			if err != nil {
				t.Fatalf("decodeClientResponse(): %v", err)
			}
			ctx.SemanticResponse = semantic
			body, err := router.encodeClientResponse(*semantic, ctx)
			if err != nil {
				t.Fatalf("encodeClientResponse(): %v", err)
			}
			if !bytes.Contains(compactJSON(t, body), []byte(clientReasoningCount(client))) {
				t.Fatalf("%s client usage lost the reasoning count:\n%s", client, body)
			}
			router.reportNonStreamingUsage(ctx, time.Second, router.takeNeutralResponseUsage(ctx))
			assertUsageLineReasoning(t, findLogEvent(t, logs, "llm_usage"))
		})
	}
}

func TestOpenRouterReasoningTokensReachTheStreamedUsageLineAndClient(t *testing.T) {
	for _, client := range extProcMatrixFormats {
		t.Run(string(client), func(t *testing.T) {
			logs := captureLogs(t)
			router := &OpenAIRouter{}
			includeUsage := true
			ctx := &RequestContext{
				RequestID: "rt_reasoning-stream", RequestModel: "xiaomi/mimo-v2.5-pro@thinking-on",
				SourceFormat: client, TargetFormat: llmprotocol.OpenAIChatV1,
				IsStreamingResponse: true, TraceContext: context.Background(),
				SemanticRequest: &llmprotocol.Request{
					Generation: 1, Model: "xiaomi/mimo-v2.5-pro@thinking-on", Stream: true,
					StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
				},
			}
			response := router.handleSemanticStreamingResponseBody(
				openRouterCapture(t, "chat-stream-reasoning.sse"), true, ctx,
			)
			assertUsageLineReasoning(t, findLogEvent(t, logs, "llm_usage"))
			if client == llmprotocol.OpenAIChatV1 {
				// A same-format stream travels as the upstream sent it.
				return
			}
			body := responseBodyMutationBytes(response)
			if !bytes.Contains(body, []byte(clientReasoningCount(client))) {
				t.Fatalf("%s client stream usage lost the reasoning count:\n%s", client, body)
			}
		})
	}
}

// The same buffered capture reaches a Messages client with its thinking block
// ahead of the answer.
func TestOpenRouterReasoningPrecedesTextForAMessagesClient(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1}
	semantic, err := router.decodeClientResponse(openRouterCapture(t, "chat-response-reasoning.json"), ctx)
	if err != nil {
		t.Fatalf("decodeClientResponse(): %v", err)
	}
	body, err := router.encodeClientResponse(*semantic, ctx)
	if err != nil {
		t.Fatalf("encodeClientResponse(): %v", err)
	}
	var message struct {
		Content []struct {
			Type string `json:"type"`
		} `json:"content"`
	}
	if err := json.Unmarshal(body, &message); err != nil {
		t.Fatal(err)
	}
	if len(message.Content) != 2 || message.Content[0].Type != "thinking" || message.Content[1].Type != "text" {
		t.Fatalf("Messages content order = %+v, want thinking then text\n%s", message.Content, body)
	}
}

func compactJSON(t *testing.T, body []byte) []byte {
	t.Helper()
	var compacted bytes.Buffer
	if err := json.Compact(&compacted, body); err != nil {
		t.Fatalf("client body is not JSON: %v\n%s", err, body)
	}
	return compacted.Bytes()
}
