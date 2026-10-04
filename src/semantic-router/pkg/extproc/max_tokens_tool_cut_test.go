package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// maxTokensToolCutChunks loads a recorded Anthropic stream from the protocol
// codec's goldens: a tool_use block cut off at max_tokens, or its end_turn
// control.
func maxTokensToolCutChunks(t *testing.T, path string) []string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("..", "protocolcodec", "testdata", "golden", path))
	if err != nil {
		t.Fatal(err)
	}
	var input struct {
		Chunks []string `json:"chunks"`
	}
	if err := json.Unmarshal(body, &input); err != nil {
		t.Fatal(err)
	}
	return input.Chunks
}

func driveMaxTokensToolCut(
	t *testing.T,
	target llmprotocol.WireFormat,
	chunks []string,
) (*RequestContext, *semanticStreamBuffers, *observer.ObservedLogs) {
	t.Helper()
	logs := captureLogs(t)
	stream, err := protocolcodec.NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{
		RequestID: "req-cut", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200, IsStreamingResponse: true,
		SourceFormat: target, TargetFormat: llmprotocol.AnthropicMessagesV1,
		ProtocolResponseStream: stream,
		SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
	}
	buffers := &semanticStreamBuffers{}
	for _, chunk := range chunks {
		buffers.push([]byte(chunk), ctx)
	}
	buffers.finalize(ctx)
	(&OpenAIRouter{}).finalizeSemanticStreamingResponse(ctx, buffers.streamErr)
	return ctx, buffers, logs
}

// A tool call the model was cut off writing at max_tokens is a served length
// stop: no turn failure, usage as the provider stated it, and the client's
// body ends in its format's max-tokens terminal.
func TestAToolCallCutAtMaxTokensIsAServedLengthStop(t *testing.T) {
	chunks := maxTokensToolCutChunks(t, "stream/040-anthropic-max-tokens-mid-tool-in.json")
	for target, ending := range map[llmprotocol.WireFormat][]string{
		llmprotocol.AnthropicMessagesV1: {`"stop_reason":"max_tokens"`, "event: message_stop"},
		llmprotocol.OpenAIChatV1:        {`"finish_reason":"length"`, "data: [DONE]"},
		llmprotocol.OpenAIResponsesV1:   {`"status":"incomplete"`, "event: response.incomplete"},
	} {
		t.Run(string(target), func(t *testing.T) {
			ctx, buffers, logs := driveMaxTokensToolCut(t, target, chunks)
			if ctx.SemanticStreamErr != nil {
				t.Fatalf("stream failed: %v", ctx.SemanticStreamErr)
			}
			for _, entry := range logs.All() {
				if entry.ContextMap()["event"] == "turn_failed" {
					t.Fatalf("a length stop was classed a turn failure: %v", entry.ContextMap())
				}
			}
			usage := findLogEvent(t, logs, "llm_usage")
			if class := usage["failure_class"]; class != nil && class != "" {
				t.Fatalf("llm_usage failure_class = %v, want none", class)
			}
			if fmt.Sprint(usage["completion_tokens"]) != "60" {
				t.Fatalf("llm_usage completion_tokens = %#v, want 60", usage["completion_tokens"])
			}
			body := strings.TrimSpace(string(buffers.translated))
			for _, want := range ending {
				if !strings.Contains(body, want) {
					t.Fatalf("client body lacks %s:\n%s", want, body)
				}
			}
			if lastFrame := body[strings.LastIndex(body, "\n\n")+1:]; !strings.Contains(lastFrame, ending[len(ending)-1]) {
				t.Fatalf("client body does not end in %s:\n%s", ending[len(ending)-1], body)
			}
			response := ctx.SemanticResponse
			if response == nil || response.StopReason != llmprotocol.StopMaxTokens {
				t.Fatalf("settled response = %#v, want a max_tokens stop", response)
			}
			if !responseCutMidToolCall(response) {
				t.Fatal("the settled response does not mark its cut tool call incomplete")
			}
		})
	}
}

// The control: the same truncated arguments under end_turn are a malformed
// provider stream, classed as before.
func TestTruncatedToolArgumentsUnderEndTurnStillFail(t *testing.T) {
	chunks := maxTokensToolCutChunks(t, "rejection/240-anthropic-stream-truncated-tool-end-turn-in.json")
	ctx, _, logs := driveMaxTokensToolCut(t, llmprotocol.OpenAIResponsesV1, chunks)
	if ctx.SemanticResponse != nil {
		t.Fatal("a malformed stream settled a response")
	}
	failed := findLogEvent(t, logs, "turn_failed")
	if failed["failure_class"] != turnFailureUpstreamError || failed["failure_detail"] != "invalid_stream_tool_arguments" {
		t.Fatalf("turn_failed = %#v, want upstream_error/invalid_stream_tool_arguments", failed)
	}
}

// A cut turn is not cached: the client's retry with a higher limit would be
// answered with the same cut call. A max_tokens stop without one caches.
func TestACutToolCallTurnIsNotCached(t *testing.T) {
	cut := &llmprotocol.Response{StopReason: llmprotocol.StopMaxTokens, Output: []llmprotocol.OutputItem{{
		ID: "item", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
			Kind:     llmprotocol.ContentToolCall,
			ToolCall: &llmprotocol.ToolCall{ID: "toolu_1", Name: "bash", Arguments: `{"command": "ca`, Incomplete: true},
		}},
	}}}
	text := &llmprotocol.Response{StopReason: llmprotocol.StopMaxTokens, Output: []llmprotocol.OutputItem{{
		ID: "item", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "half"}},
	}}}
	for name, test := range map[string]struct {
		response *llmprotocol.Response
		cached   bool
	}{
		"cut tool call":    {response: cut, cached: false},
		"max_tokens text":  {response: text, cached: true},
		"no response kept": {response: nil, cached: true},
	} {
		t.Run(name, func(t *testing.T) {
			mockCache, router, decision := statusCacheRouter()
			ctx := withSelectedDecision(&RequestContext{
				RequestID: "req-cut-cache", UpstreamStatusCode: 200, RequestModel: "test", RequestQuery: "hello",
				SemanticRequest: testNeutralRequest("test", "hello"), SemanticResponse: test.response,
			}, decision)
			router.updateResponseCache(ctx, []byte(`{"choices":[]}`))
			if mockCache.addEntryCalled != test.cached {
				t.Fatalf("cached = %v, want %v", mockCache.addEntryCalled, test.cached)
			}
		})
	}
}
