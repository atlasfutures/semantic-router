package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
)

// Some OpenAI-compatible servers end a Chat reply holding a whole tool call
// with finish_reason "stop". An agent client reads the stop reason to decide
// whether to run the call, so each client format must be told the turn ended
// on a tool call: through the re-encoded body or the same-format filter.

func TestAChatToolCallUnderAStopFinishReachesEveryClientAsAToolCall(t *testing.T) {
	stream := strings.Join(maxTokensToolCutChunks(t, "stream/048-chat-tool-call-stop-finish-in.json"), "")
	buffered, err := os.ReadFile(filepath.Join("../../../../rayline-core/protocolcodec", "testdata", "golden",
		"response/014-chat-tool-call-stop-finish-in.json"))
	require.NoError(t, err)
	for client, want := range map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:        `"finish_reason":"tool_calls"`,
		llmprotocol.AnthropicMessagesV1: `"stop_reason":"tool_use"`,
	} {
		t.Run(string(client)+"/stream", func(t *testing.T) {
			includeUsage := true
			ctx := chatToolCallStopContext(client, &llmprotocol.Request{
				Generation: 1, Model: "public-model", Stream: true,
				StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
			})
			response := (&OpenAIRouter{}).handleSemanticStreamingResponseBody([]byte(stream), true, ctx)
			body := string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
			require.Contains(t, body, want)
			require.NotContains(t, body, `"finish_reason":"stop"`)
			require.NotContains(t, body, `"stop_reason":"end_turn"`)
			require.NotNil(t, ctx.SemanticResponse)
			require.Equal(t, llmprotocol.StopToolCall, ctx.SemanticResponse.StopReason)
		})
		// A buffered body with no finish_reason at all is answered the same.
		var reply map[string]any
		require.NoError(t, json.Unmarshal(buffered, &reply))
		delete(reply["choices"].([]any)[0].(map[string]any), "finish_reason")
		noFinish, err := json.Marshal(reply)
		require.NoError(t, err)
		for name, upstream := range map[string][]byte{"buffered": buffered, "buffered, no finish_reason": noFinish} {
			t.Run(string(client)+"/"+name, func(t *testing.T) {
				ctx := chatToolCallStopContext(client, testNeutralRequest("public-model", "weather?"))
				response, err := (&OpenAIRouter{}).handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
					ResponseBody: &ext_proc.HttpBody{Body: upstream, EndOfStream: true},
				}, ctx)
				require.NoError(t, err)
				require.Nil(t, response.GetImmediateResponse())
				mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
				require.NotNil(t, mutation, "the upstream body reached the client unchanged")
				var compact bytes.Buffer
				require.NoError(t, json.Compact(&compact, mutation.GetBody()))
				body := compact.String()
				require.Contains(t, body, want)
				require.NotContains(t, body, `"finish_reason":"stop"`)
				require.NotContains(t, body, `"stop_reason":"end_turn"`)
			})
		}
	}
}

func chatToolCallStopContext(client llmprotocol.WireFormat, request *llmprotocol.Request) *RequestContext {
	return &RequestContext{
		RequestID: "req-tool-stop", RequestModel: "public-model", VSRSelectedModel: "public-model",
		SourceFormat: client, TargetFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: request, IsStreamingResponse: request.Stream,
		StartTime: time.Now(), TraceContext: context.Background(), UpstreamStatusCode: 200,
	}
}

// A same-format stream whose tool call ended on "length" and whose trailing
// chunk repeats "stop" never tells the client "tool_calls": only a first
// finish of "stop" is restated, so the cut call is not run.
func TestASameFormatChatStreamRepeatingStopAfterALengthKeepsTheLength(t *testing.T) {
	chunk := func(delta, finish string) string {
		return `data: {"id":"gen-e2e","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` +
			delta + `,"finish_reason":` + finish + `}]}` + "\n\n"
	}
	stream := chunk(`{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}`, "null") +
		chunk(`{}`, `"length"`) + chunk(`{}`, `"stop"`) + "data: [DONE]\n\n"
	includeUsage := true
	ctx := chatToolCallStopContext(llmprotocol.OpenAIChatV1, &llmprotocol.Request{
		Generation: 1, Model: "public-model", Stream: true,
		StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
	})
	response := (&OpenAIRouter{}).handleSemanticStreamingResponseBody([]byte(stream), true, ctx)
	body := string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	require.Contains(t, body, `"finish_reason":"length"`)
	require.NotContains(t, body, `"finish_reason":"tool_calls"`)
}
