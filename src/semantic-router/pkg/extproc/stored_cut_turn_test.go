package extproc

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

// storeTurn persists one served Responses turn through the Router's own
// path (store=true), and returns its public ID.
func storeTurn(t *testing.T, router *OpenAIRouter, previousID string, input string, response llmprotocol.Response) string {
	t.Helper()
	request := llmprotocol.Request{
		PreviousResponseID: previousID,
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: input}},
		}},
	}
	body := []byte(`{"model":"public-model","input":` + strconv.Quote(input) + `}`)
	state, err := router.ResponseAPIFilter.PrepareObjectState(t.Context(), request, body)
	require.NoError(t, err)
	response.ID = state.GeneratedResponseID
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1, UpstreamStatusCode: 200,
		ResponseObjectState: state, SemanticResponse: &response, TraceContext: t.Context(),
	}
	router.persistResponseObject(ctx)
	return state.GeneratedResponseID
}

// continuation builds the provider request a follow-up with
// previous_response_id produces, through the same state and materialization
// the request path runs.
func continuation(t *testing.T, router *OpenAIRouter, previousID string, input llmprotocol.Message) *llmprotocol.Request {
	t.Helper()
	request := &llmprotocol.Request{Generation: 1, Model: "public-model", PreviousResponseID: previousID, Messages: []llmprotocol.Message{input}}
	state, err := router.ResponseAPIFilter.PrepareObjectState(t.Context(), *request, nil)
	require.NoError(t, err)
	_, err = router.materializeResponseObjectContext(request, &RequestContext{ResponseObjectState: state})
	require.NoError(t, err)
	require.NoError(t, llmprotocol.ValidateRequest(*request, llmprotocol.DefaultPolicy().Limits))
	return request
}

func newStoredResponsesRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	store, err := responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true, TTLSeconds: 600})
	require.NoError(t, err)
	return &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(store)}
}

func storedTurnUserText(text string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
}

// A turn stored after a length stop mid tool call can be continued: the
// history keeps what the turn wrote before the cut and leaves out the cut
// call, which never ran. The stored object itself keeps the call as served.
func TestAStoredCutTurnCanBeContinued(t *testing.T) {
	router := newStoredResponsesRouter(t)
	cutID := storeTurn(t, router, "", "run the probe", llmprotocol.Response{
		Generation: 1, Model: "public-model", StopReason: llmprotocol.StopMaxTokens,
		Output: []llmprotocol.OutputItem{
			{ID: "item_reasoning", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentReasoning, Text: "I will write the probe.", Reasoning: llmprotocol.ReasoningScopeSummary,
			}}},
			{ID: "item_text", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText, Text: "Writing the probe now.",
			}}},
			{ID: "item_call", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
					ID: "call_cut", Name: "bash", Arguments: `{"command": "cat > /tmp/probe3.js <<'EOF'`, Incomplete: true,
				},
			}}},
		},
		Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
	})

	stored, err := router.ResponseAPIFilter.store.GetResponse(t.Context(), cutID)
	require.NoError(t, err)
	require.Equal(t, "incomplete", stored.Status)
	var storedCall *responseapi.OutputItem
	for index := range stored.Output {
		if stored.Output[index].Type == responseapi.ItemTypeFunctionCall {
			storedCall = &stored.Output[index]
		}
	}
	require.NotNil(t, storedCall, "the stored object lost the cut call")
	require.Equal(t, "incomplete", storedCall.Status)

	request := continuation(t, router, cutID, storedTurnUserText("try again"))
	var texts []string
	for _, message := range request.Messages {
		for _, content := range message.Content {
			require.Nil(t, content.ToolCall, "the cut call was replayed as history")
			if content.Text != "" {
				texts = append(texts, content.Text)
			}
		}
	}
	require.Equal(t, []string{"run the probe", "I will write the probe.", "Writing the probe now.", "try again"}, texts)
}

// The control: a complete call with its function_call_output still
// materializes as before.
func TestAStoredCompleteCallStillMaterializes(t *testing.T) {
	router := newStoredResponsesRouter(t)
	callID := storeTurn(t, router, "", "check Paris", llmprotocol.Response{
		Generation: 1, Model: "public-model", StopReason: llmprotocol.StopToolCall,
		Output: []llmprotocol.OutputItem{{ID: "item_call", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_weather", Name: "lookup_weather", Arguments: `{"city":"Paris"}`},
		}}}},
		Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
	})
	result := llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
			CallID: "call_weather", DeferredLink: true,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "sunny"}},
		},
	}}}
	request := continuation(t, router, callID, result)
	var call *llmprotocol.ToolCall
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.ToolCall != nil {
				call = content.ToolCall
			}
		}
	}
	require.NotNil(t, call)
	require.Equal(t, `{"city":"Paris"}`, call.Arguments)
}
