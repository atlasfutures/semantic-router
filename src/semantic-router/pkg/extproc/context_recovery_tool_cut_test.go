package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// recoveryTurn runs one buffered Chat reply through the response body
// handler with context recovery active, and returns the client's body, the
// immediate response if one replaced it, and how many follow-up calls the
// recovery made.
func recoveryTurn(t *testing.T, failureMode, upstream string) (string, *ext_proc.ImmediateResponse, int32) {
	body, immediate, followups, _ := recoveryTurnContext(t, failureMode, upstream)
	return body, immediate, followups
}

func recoveryTurnContext(t *testing.T, failureMode, upstream string) (string, *ext_proc.ImmediateResponse, int32, *RequestContext) {
	t.Helper()
	var followups atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		followups.Add(1)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"id":"followup","object":"chat.completion","created":1,"model":"model",` +
			`"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25},` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"final answer"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(server.Close)
	plugin := map[string]interface{}{
		"enabled":  true,
		"recovery": map[string]interface{}{"enabled": true, "store": "redis"},
	}
	if failureMode != "" {
		plugin["failure_mode"] = failureMode
	}
	decision := &config.Decision{
		Name: "recoverable",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginContextCompression, Configuration: config.MustStructuredPayload(plugin),
		}},
	}
	router := &OpenAIRouter{
		Config:              &config.RouterConfig{Looper: config.LooperConfig{Endpoint: server.URL}},
		CompressionRecovery: &recoveryStoreStub{},
	}
	ctx := &RequestContext{
		RequestID: "req-recovery-cut", RequestModel: "model", VSRSelectedModel: "model",
		VSRSelectedDecisionName: decision.Name, VSRSelectedDecision: decision,
		ContextCompressionRecoveryKeys: []string{"key-1"},
		Headers:                        map[string]string{"x-authz-user-id": "user-1"},
		SourceFormat:                   llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: testNeutralRequest("model", "question"),
		StartTime:       time.Now(), TraceContext: context.Background(), UpstreamStatusCode: 200,
	}
	scope := router.contextCompressionScope(ctx)
	router.CompressionRecovery.(*recoveryStoreStub).entries = map[string]contextcompression.RecoveryEntry{
		scope + ":key-1": {Key: "key-1", Scope: scope, Content: "full original context"},
	}
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(upstream), EndOfStream: true},
	}, ctx)
	require.NoError(t, err)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		return string(immediate.GetBody()), immediate, followups.Load(), ctx
	}
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		return string(mutation.GetBody()), nil, followups.Load(), ctx
	}
	return upstream, nil, followups.Load(), ctx
}

func recoveryCallReply(arguments, finish, native string) string {
	nativeMember := ""
	if native != "" {
		nativeMember = `,"native_finish_reason":` + strconv.Quote(native)
	}
	return `{"id":"initial","object":"chat.completion","created":1,"model":"model",` +
		`"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13},` +
		`"choices":[{"index":0,"finish_reason":` + strconv.Quote(finish) + nativeMember + `,"message":{"role":"assistant","content":null,"tool_calls":[{` +
		`"id":"retrieve-1","type":"function","function":{"name":"` + contextcompression.RetrieveToolName + `","arguments":` + strconv.Quote(arguments) + `}}]}}]}`
}

// A reply cut at the output limit while the model wrote the Router's own
// recovery call is served as the length stop it is: no follow-up, no 502,
// no end-turn rewrite, and no trace of the internal tool.
func TestACutContextRecoveryCallIsServedAsALengthStop(t *testing.T) {
	for _, failureMode := range []string{config.ContextCompressionFailureClosed, config.ContextCompressionFailureOpen} {
		for name, upstream := range map[string]string{
			"finish_reason length":       recoveryCallReply(`{"key":"ke`, "length", ""),
			"native max_output_tokens":   recoveryCallReply(`{"key":"ke`, "tool_calls", "max_output_tokens"),
			"cut before the key started": recoveryCallReply(`{"ke`, "length", ""),
		} {
			t.Run(failureMode+"/"+name, func(t *testing.T) {
				body, immediate, followups, ctx := recoveryTurnContext(t, failureMode, upstream)
				// The removed call still marks the turn cut, so it is not cached.
				require.True(t, turnCutMidToolCall(ctx))
				require.Nil(t, immediate, "the cut turn was answered with an immediate response: %s", body)
				require.Zero(t, followups, "recovery ran on a cut call")
				require.NotContains(t, body, contextcompression.RetrieveToolName)
				require.NotContains(t, body, "could not be retrieved")
				var wire struct {
					Choices []struct {
						FinishReason string `json:"finish_reason"`
					} `json:"choices"`
				}
				require.NoError(t, json.Unmarshal([]byte(body), &wire), body)
				require.Len(t, wire.Choices, 1)
				require.Equal(t, "length", wire.Choices[0].FinishReason, body)
			})
		}
	}
}

// The control: a whole recovery call still runs recovery and serves the
// follow-up's answer.
func TestAWholeContextRecoveryCallStillRecovers(t *testing.T) {
	body, immediate, followups := recoveryTurn(t, config.ContextCompressionFailureClosed,
		recoveryCallReply(`{"key":"key-1"}`, "tool_calls", ""))
	require.Nil(t, immediate)
	require.EqualValues(t, 1, followups)
	require.True(t, strings.Contains(body, "final answer"), body)
	_, _, _, ctx := recoveryTurnContext(t, config.ContextCompressionFailureClosed,
		recoveryCallReply(`{"key":"key-1"}`, "tool_calls", ""))
	require.False(t, turnCutMidToolCall(ctx))
}
