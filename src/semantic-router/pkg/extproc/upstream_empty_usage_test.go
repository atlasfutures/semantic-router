package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// emptyChatCompletionBody is the turn decision 30 named: an assistant message
// with no content at all, no tool call and no reasoning, stopped normally and
// billed for one token. Measured against deepseek-v4-pro thinking-off through
// OpenRouter on 2026-09-04, StreamLake returned it on 7 of 11 attempts of one
// degenerate tool turn.
//
// The Router cannot use it -- ValidateResponse refuses an output item that
// names no content -- but the upstream still charged for it and still said who
// served it and what it stopped for.
const emptyChatCompletionBody = `{
  "id": "gen-1757030000-Rz4",
  "provider": "StreamLake",
  "model": "deepseek/deepseek-v4-pro",
  "object": "chat.completion",
  "created": 1757030000,
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {"role": "assistant", "content": null}
    }
  ],
  "usage": {"prompt_tokens": 31402, "completion_tokens": 1, "total_tokens": 31403}
}`

func emptyCompletionContext() *RequestContext {
	return &RequestContext{
		RequestID:    "rt_5c1d77e0-000",
		RequestModel: "deepseek-v4-pro@thinking-off",
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		TraceContext: context.Background(),
	}
}

// An empty completion is refused, and the refusal is where its attribution
// used to end. The usage line now names the provider, the stop and the tokens
// the upstream billed, so the turn can be joined to the arm that produced it.
func TestUnusableResponseStillWritesTheUsageLine(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := emptyCompletionContext()

	response := router.handleNonStreamingResponseBody([]byte(emptyChatCompletionBody), ctx, time.Second)

	if response.GetImmediateResponse() == nil {
		t.Fatal("an unusable response must still be refused")
	}
	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["upstream_provider"].(string); got != "StreamLake" {
		t.Fatalf("upstream_provider = %v, want the provider that returned nothing", fields["upstream_provider"])
	}
	if got, _ := fields["native_stop_reason"].(string); got != "stop" {
		t.Fatalf("native_stop_reason = %v, want the reason the upstream sent", fields["native_stop_reason"])
	}
	if got, _ := fields["completion_tokens"].(int64); got != 1 {
		t.Fatalf("completion_tokens = %v, want the one token that was billed", fields["completion_tokens"])
	}
	if got, _ := fields["prompt_tokens"].(int64); got != 31402 {
		t.Fatalf("prompt_tokens = %v, want the prompt the upstream charged for", fields["prompt_tokens"])
	}
	if got, _ := fields["failure_class"].(string); got != "empty_output_item" {
		t.Fatalf("failure_class = %v, want the code the refusal was raised with", fields["failure_class"])
	}
}

// A body that never became a response has nothing to attribute. Writing a
// usage line for it would assert counts no upstream ever stated.
func TestUndecodableResponseWritesNoUsageLine(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := emptyCompletionContext()

	router.handleNonStreamingResponseBody([]byte(`{"choices":`), ctx, time.Second)

	for _, entry := range logs.All() {
		if name, _ := entry.ContextMap()["event"].(string); name == "llm_usage" {
			t.Fatal("a body that never decoded must not be billed")
		}
	}
}
