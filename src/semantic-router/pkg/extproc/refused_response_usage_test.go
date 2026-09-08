package extproc

import (
	"testing"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// refusedChatCompletionBody is an upstream answer the Router refuses. Its
// finish_reason is not one of the five the Chat contract recognises
// (provider_response_validation.go:130-137), so normalizedChatChoices returns
// invalid_chat_finish_reason and decodeClientResponse discards the decode.
// Everything else is well formed, including the usage object stating the
// tokens the upstream has already charged for.
const refusedChatCompletionBody = `{"id":"response_1","object":"chat.completion","model":"source-model",` +
	`"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"unsupported_stop"}],` +
	`"usage":{"prompt_tokens":31402,"completion_tokens":57,"total_tokens":31459}}`

// undecodableChatCompletionBody never becomes a response at all.
const undecodableChatCompletionBody = `{"choices":`

// refusedResponseUsageFields returns the fields of the llm_usage line written
// for body, or nil when no such line was written.
func refusedResponseUsageFields(t *testing.T, body string) map[string]interface{} {
	t.Helper()
	core, observedLogs := observer.New(zapcore.DebugLevel)
	defer zap.ReplaceGlobals(zap.New(core))()

	router := &OpenAIRouter{}
	ctx := &RequestContext{
		RequestID:    "request_1",
		RequestModel: "public-model",
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		TraceContext: t.Context(),
	}
	response := router.handleNonStreamingResponseBody([]byte(body), ctx, time.Second)
	if response.GetImmediateResponse() == nil {
		t.Fatalf("the response contract must refuse this body: %+v", response)
	}
	entries := observedLogs.FilterMessage("llm_usage").All()
	if len(entries) == 0 {
		return nil
	}
	return entries[0].ContextMap()
}

// TestRefusedUpstreamResponseStillRecordsItsUsage is the ask.
//
// decodeClientResponse returns only the error when the response contract
// refuses a body (processor_protocol_contract.go:236-238), so ctx.SemanticResponse
// is never set and handleNonStreamingResponseBody returns before
// reportNonStreamingUsage. A response the upstream billed for leaves no
// llm_usage line, and the charge has no settlement record to reconcile.
//
// A body that never decoded is the bound: it states no tokens, so nothing can
// be attributed to it and it must stay unbilled.
func TestRefusedUpstreamResponseStillRecordsItsUsage(t *testing.T) {
	fields := refusedResponseUsageFields(t, refusedChatCompletionBody)
	if fields == nil {
		t.Fatal("a refused response wrote no llm_usage line: the tokens the upstream charged for are recorded nowhere")
	}
	if got, _ := fields["prompt_tokens"].(int64); got != 31402 {
		t.Errorf("prompt_tokens = %v, want the 31402 the upstream charged for", fields["prompt_tokens"])
	}
	if got, _ := fields["completion_tokens"].(int64); got != 57 {
		t.Errorf("completion_tokens = %v, want the 57 the upstream charged for", fields["completion_tokens"])
	}
	if undecoded := refusedResponseUsageFields(t, undecodableChatCompletionBody); undecoded != nil {
		t.Errorf("a body that never decoded was billed: %v", undecoded)
	}
}

// TestRefusedUpstreamResponseLeavesNoUsageToday pins the present behaviour so
// a change to it shows up in the diff rather than only in the test above.
func TestRefusedUpstreamResponseLeavesNoUsageToday(t *testing.T) {
	if fields := refusedResponseUsageFields(t, refusedChatCompletionBody); fields != nil {
		t.Errorf("llm_usage = %v: the recorded behaviour has changed", fields)
	}
}
