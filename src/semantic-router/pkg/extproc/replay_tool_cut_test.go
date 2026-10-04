package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// A non-streaming Messages reply cut mid tool call, as Anthropic returns it
// (recorded live; ids replaced).
const replayBufferedToolCut = `{"id":"msg_fixture","type":"message","role":"assistant","content":[{"type":"tool_use","id":"toolu_fixture","name":"bash","input":{}}],"model":"m","stop_reason":"max_tokens","stop_sequence":null,"usage":{"input_tokens":556,"output_tokens":120}}`

func newCutTurnReplay(t *testing.T) (*routerreplay.Recorder, string) {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, true, 4096)
	replayID, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID: "replay-cut", RequestID: "req-cut", Decision: "default_route",
	})
	require.NoError(t, err)
	return recorder, replayID
}

// requireCutReplay checks a replay record of a turn cut mid tool call: the
// call is traced as cut, never as an invocation; the trajectory export has
// no tool call; and the record's terminal reason says the turn was cut.
func requireCutReplay(t *testing.T, recorder *routerreplay.Recorder, replayID string) {
	t.Helper()
	record, found := recorder.GetRecord(replayID)
	require.True(t, found)
	require.NotNil(t, record.ToolTrace)
	var cut int
	for _, step := range record.ToolTrace.Steps {
		require.NotEqual(t, replayToolStepAssistantToolCall, step.Type, "a cut call was traced as an invocation")
		if step.Type == replayToolStepAssistantToolCallCut {
			cut++
			require.Equal(t, "bash", step.ToolName)
		}
	}
	require.Equal(t, 1, cut)
	require.Contains(t, record.ToolTrace.Flow, "LLM Tool Call Cut at max_tokens")
	for _, message := range buildTrajectoryMessages([]trajectoryTurn{{Steps: record.ToolTrace.Steps}}) {
		require.Empty(t, message.ToolCalls, "the trajectory export presents a cut call as made")
	}
	require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
	require.Equal(t, "response_cut_mid_tool_call", record.TerminalReason)
}

func TestReplayRecordsABufferedCutCallAsCut(t *testing.T) {
	response, _, _, err := protocolcodec.NewBuiltinEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, []byte(replayBufferedToolCut))
	require.NoError(t, err)
	recorder, replayID := newCutTurnReplay(t)
	ctx := &RequestContext{
		RequestID: "req-cut", RouterReplayID: replayID, RouterReplayRecorder: recorder,
		SourceFormat: llmprotocol.AnthropicMessagesV1, UpstreamStatusCode: 200, SemanticResponse: &response,
	}
	(&OpenAIRouter{ReplayRecorder: recorder}).attachRouterReplayResponse(ctx, []byte(replayBufferedToolCut), true)
	requireCutReplay(t, recorder, replayID)
}

func TestReplayRecordsAStreamedCutCallAsCut(t *testing.T) {
	recorder, replayID := newCutTurnReplay(t)
	chunks := maxTokensToolCutChunks(t, "stream/040-anthropic-max-tokens-mid-tool-in.json")
	ctx, _, _ := driveMaxTokensToolCut(t, llmprotocol.OpenAIResponsesV1, chunks, func(ctx *RequestContext) {
		ctx.RouterReplayID, ctx.RouterReplayRecorder = replayID, recorder
	})
	require.NoError(t, ctx.SemanticStreamErr)
	requireCutReplay(t, recorder, replayID)
}

// A whole call is still traced as an invocation, and the turn as complete.
func TestReplayRecordsAWholeCallAsAnInvocation(t *testing.T) {
	recorder, replayID := newCutTurnReplay(t)
	ctx := &RequestContext{
		RequestID: "req-cut", RouterReplayID: replayID, RouterReplayRecorder: recorder,
		SourceFormat: llmprotocol.AnthropicMessagesV1, UpstreamStatusCode: 200,
		SemanticResponse: &llmprotocol.Response{StopReason: llmprotocol.StopToolCall, Output: []llmprotocol.OutputItem{{
			Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "toolu_1", Name: "bash", Arguments: `{}`},
			}},
		}}},
	}
	(&OpenAIRouter{ReplayRecorder: recorder}).attachRouterReplayResponse(ctx, nil, true)
	record, found := recorder.GetRecord(replayID)
	require.True(t, found)
	require.Equal(t, replayToolStepAssistantToolCall, record.ToolTrace.Steps[0].Type)
	require.Equal(t, "response_complete", record.TerminalReason)
}
