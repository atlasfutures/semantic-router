package extproc

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const sameFormatCutArguments = `{"command":"cat > /tmp/probe3.js <<'EOF'\n$(for i in $(seq 1 300); do`

// chatToolReply is a non-streaming Chat reply with one tool call, as an
// OpenRouter Chat backend returns it.
func chatToolReply(arguments, finish, native string) string {
	return `{"id":"gen-fixture","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,` +
		`"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_fixture","type":"function",` +
		`"function":{"name":"bash","arguments":` + strconv.Quote(arguments) + `}}]},` +
		`"finish_reason":` + strconv.Quote(finish) + `,"native_finish_reason":` + strconv.Quote(native) + `}],` +
		`"usage":{"prompt_tokens":92,"completion_tokens":40,"total_tokens":132}}`
}

// sameFormatChatClientBody runs a buffered Chat reply through the
// same-format response pipeline and returns what the client receives: the
// rewritten body, or the upstream's when nothing rewrote it.
func sameFormatChatClientBody(t *testing.T, upstream string) string {
	t.Helper()
	_, router, decision := statusCacheRouter()
	ctx := withSelectedDecision(&RequestContext{
		RequestID: "req-same-format-cut", RequestModel: "test", RequestQuery: "hello",
		SemanticRequest: testNeutralRequest("test", "hello"),
		SourceFormat:    llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		TraceContext: context.Background(), UpstreamStatusCode: 200,
	}, decision)
	response := router.handleNonStreamingResponseBody([]byte(upstream), ctx, time.Second)
	require.NotNil(t, response)
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		return string(mutation.GetBody())
	}
	return upstream
}

func chatFinishReason(t *testing.T, body string) string {
	t.Helper()
	var wire struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal([]byte(body), &wire), body)
	require.Len(t, wire.Choices, 1)
	return wire.Choices[0].FinishReason
}

// An OpenRouter Chat backend labels a call cut at the output limit
// tool_calls, with the provider's max_output_tokens only in
// native_finish_reason. A same-format Chat client must not receive that body
// as it is: it is told finish_reason length.
func TestASameFormatChatCutIsSentAsALengthStop(t *testing.T) {
	body := sameFormatChatClientBody(t, chatToolReply(sameFormatCutArguments, "tool_calls", "max_output_tokens"))
	require.Equal(t, "length", chatFinishReason(t, body))
}

// The control: a whole tool call under a plain tool_calls stop is handed
// back as the upstream sent it.
func TestASameFormatChatToolCallPassesThrough(t *testing.T) {
	upstream := chatToolReply(`{"command":"ls"}`, "tool_calls", "tool_calls")
	body := sameFormatChatClientBody(t, upstream)
	require.Equal(t, "tool_calls", chatFinishReason(t, body))
	require.JSONEq(t, upstream, body)
}

// chatToolStream is an OpenRouter-shaped Chat stream of one tool call, ended
// by the finish chunk and the usage chunk that repeats it.
func chatToolStream(arguments, finish, native string) []string {
	chunk := func(choice string, extra string) string {
		return `data: {"id":"gen-fixture","object":"chat.completion.chunk","created":1,"model":"m","choices":[` + choice + `]` + extra + "}\n\n"
	}
	return []string{
		chunk(`{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_fixture","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":null,"native_finish_reason":null}`, ""),
		chunk(`{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":`+strconv.Quote(arguments)+`}}]},"finish_reason":null,"native_finish_reason":null}`, ""),
		chunk(`{"index":0,"delta":{"content":"","role":"assistant"},"finish_reason":"`+finish+`","native_finish_reason":"`+native+`"}`, ""),
		chunk(`{"index":0,"delta":{"content":"","role":"assistant"},"finish_reason":"`+finish+`","native_finish_reason":"`+native+`"}`,
			`,"usage":{"prompt_tokens":92,"completion_tokens":40,"total_tokens":132}`) +
			"data: [DONE]\n\n",
	}
}

// sameFormatChatStreamBody drives a Chat stream through the same-format
// streaming handler and returns the bytes the client receives.
func sameFormatChatStreamBody(t *testing.T, chunks []string, includeUsage bool) string {
	t.Helper()
	request := testNeutralRequest("test", "hello")
	request.StreamOptions.IncludeUsage = &includeUsage
	ctx := &RequestContext{
		RequestID: "req-same-format-stream", RequestModel: "test", StartTime: time.Now(),
		UpstreamStatusCode: 200, IsStreamingResponse: true, SemanticRequest: request,
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
	}
	router := &OpenAIRouter{}
	var client strings.Builder
	for index, chunk := range chunks {
		response := router.handleSemanticStreamingResponseBody([]byte(chunk), index == len(chunks)-1, ctx)
		require.NotNil(t, response)
		if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			client.Write(mutation.GetBody())
		} else {
			client.WriteString(chunk)
		}
	}
	require.NoError(t, ctx.SemanticStreamErr)
	return client.String()
}

func streamFinishReasons(body string) []string {
	var reasons []string
	for _, frame := range strings.Split(body, "\n\n") {
		data, found := strings.CutPrefix(strings.TrimSpace(frame), "data: ")
		if !found || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason != nil {
				reasons = append(reasons, *choice.FinishReason)
			}
		}
	}
	return reasons
}

// The streamed form: whether or not the client asked for usage, a
// same-format Chat stream ends a cut call with finish_reason length.
func TestASameFormatChatStreamCutEndsAsALengthStop(t *testing.T) {
	for _, includeUsage := range []bool{false, true} {
		body := sameFormatChatStreamBody(t, chatToolStream(sameFormatCutArguments, "tool_calls", "max_output_tokens"), includeUsage)
		reasons := streamFinishReasons(body)
		require.NotEmpty(t, reasons, body)
		for _, reason := range reasons {
			require.Equal(t, "length", reason, "include_usage=%v:\n%s", includeUsage, body)
		}
		require.Equal(t, includeUsage, strings.Contains(body, `"usage"`), body)
	}
}

// The control: a whole call under a plain tool_calls stop streams as it was.
func TestASameFormatChatStreamToolCallPassesThrough(t *testing.T) {
	chunks := chatToolStream(`{"command":"ls"}`, "tool_calls", "tool_calls")
	body := sameFormatChatStreamBody(t, chunks, true)
	require.Equal(t, strings.Join(chunks, ""), body)
}
