package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A Responses agent's episode in which Claude and a non-Claude worker took
// turns: a Claude call with its reasoning, a kimi-style call id, an
// assistant message before a parallel pair, and a call with no reasoning
// before it. Responses sends each of these as an item of its own.
const interleavedToolEpisodeResponses = `{
	"model":"m","store":false,
	"input":[
		{"type":"message","role":"user","content":[{"type":"input_text","text":"fix the failing test"}]},
		{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Look at the tree first."}]},
		{"type":"function_call","id":"fc_1","call_id":"toolu_A","name":"bash","arguments":"{\"cmd\":\"ls\"}"},
		{"type":"function_call_output","call_id":"toolu_A","output":"a_test.go"},
		{"type":"function_call","id":"fc_2","call_id":"functions.bash:0","name":"bash","arguments":"{\"cmd\":\"go test\"}"},
		{"type":"function_call_output","call_id":"functions.bash:0","output":"FAIL"},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Reading both files."}]},
		{"type":"function_call","id":"toolu_B","call_id":"toolu_B","name":"read","arguments":"{\"path\":\"a.go\"}"},
		{"type":"function_call","id":"toolu_C","call_id":"toolu_C","name":"read","arguments":"{\"path\":\"a_test.go\"}"},
		{"type":"function_call_output","call_id":"toolu_B","output":"package a"},
		{"type":"function_call_output","call_id":"toolu_C","output":"package a_test"}
	],
	"tools":[
		{"type":"function","name":"bash","parameters":{"type":"object"}},
		{"type":"function","name":"read","parameters":{"type":"object"}}
	]
}`

type messagesWireBlock struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	ToolUseID string `json:"tool_use_id"`
	Text      string `json:"text"`
}

type messagesWireMessage struct {
	Role    string              `json:"role"`
	Content []messagesWireBlock `json:"content"`
}

// dispatchToKimiOverMessages sends body from a Responses client through
// ingress, the thinking lever, provider dispatch and encode to an
// Anthropic-format OpenRouter worker serving a non-Claude model, and returns
// the upstream body.
func dispatchToKimiOverMessages(t *testing.T, body string) []byte {
	t.Helper()
	router, logicalModel := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	profile := router.Config.ProviderProfiles["provider"]
	profile.Type = "openrouter"
	router.Config.ProviderProfiles["provider"] = profile
	router.Config.ModelConfig[logicalModel].ExternalModelIDs["vllm"] = "moonshotai/kimi-k3"

	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	episode := raylinearc.HashEpisodeID(t.Name())
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	transaction := newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil)
	transaction.markSelection(0, 10)
	defer func() { _ = transaction.abort(context.Background(), "test") }()

	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1,
		RequestID: "messages-tool-turn-grouping", TraceContext: context.Background(),
		VSRSelectedDecision:   thinkingLeverDecision(true),
		RaylineARCDispatch:    &raylinearc.WorkerManifest{ID: leverWorker},
		RaylineARCTransaction: transaction,
	}
	request, immediate := router.prepareProtocolRequest([]byte(body), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("ingress refused %s: %+v", body, ctx.ImmediateProtocolError)
	}
	dispatch, err := router.prepareProviderDispatch(request, logicalModel, "arc", false, ctx)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if ctx.RaylineARCThinking == nil || !ctx.RaylineARCThinking.Emitted {
		t.Fatalf("the lever did not steer this turn: %+v", ctx.RaylineARCThinking)
	}
	response, err := router.finalizeProviderDispatchResponse(dispatch, router.buildProviderDispatchResponse(dispatch, ctx), ctx)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	return response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
}

// OpenRouter's Messages adapter for a non-Claude model converts message by
// message and names each tool result from the calls of the assistant message
// right before it. Every tool_result must therefore follow its tool_use in
// the immediately preceding assistant message, and the lever's steer must
// come after every tool_result of the user message that carries them.
func TestMessagesDispatchToKimiPairsEachToolResultWithThePrecedingAssistant(t *testing.T) {
	body := dispatchToKimiOverMessages(t, interleavedToolEpisodeResponses)
	var wire struct {
		Model    string                `json:"model"`
		Messages []messagesWireMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("upstream body: %v: %s", err, body)
	}
	if wire.Model != "moonshotai/kimi-k3" {
		t.Fatalf("model = %q, want the kimi worker", wire.Model)
	}
	results, steered := 0, false
	for index, message := range wire.Messages {
		if index > 0 && message.Role == wire.Messages[index-1].Role {
			t.Fatalf("messages %d and %d are both %s turns: %s", index-1, index, message.Role, body)
		}
		textSeen := false
		for _, block := range message.Content {
			switch block.Type {
			case "tool_result":
				results++
				if textSeen {
					t.Fatalf("message %d has a tool_result after a text block: %s", index, body)
				}
				if index == 0 || !callsOf(wire.Messages[index-1])[block.ToolUseID] {
					t.Fatalf("tool_result %q in message %d has no tool_use in the assistant message before it: %s",
						block.ToolUseID, index, body)
				}
			case "text":
				textSeen = true
				steered = steered || strings.Contains(block.Text, leverDown)
			}
		}
	}
	if results != 4 {
		t.Fatalf("sent %d tool results, want 4: %s", results, body)
	}
	last := wire.Messages[len(wire.Messages)-1]
	if !steered || last.Content[len(last.Content)-1].Text != leverDown ||
		last.Content[0].Type != "tool_result" {
		t.Fatalf("the steer is not the trailing block of the tool results' user message: %s", body)
	}
}

func callsOf(message messagesWireMessage) map[string]bool {
	calls := map[string]bool{}
	if message.Role != "assistant" {
		return calls
	}
	for _, block := range message.Content {
		if block.Type == "tool_use" {
			calls[block.ID] = true
		}
	}
	return calls
}

// The shape seen live: a non-Claude worker answered with two parallel calls
// whose ids are its own (bash:0, bash:1) and whose items carry no id, after
// its unsigned reasoning and an assistant message; the agent resends them
// with both outputs. Both calls go out in one assistant message and both
// results in the user message after it.
func TestMessagesDispatchToKimiGroupsAParallelPairWithNativeCallIDs(t *testing.T) {
	body := dispatchToKimiOverMessages(t, `{
	"model":"m","store":false,
	"input":[
		{"type":"message","role":"system","content":[{"type":"input_text","text":"You are a coding agent."}]},
		{"type":"message","role":"user","content":[{"type":"input_text","text":"run the two checks"}]},
		{"type":"reasoning","id":"item_r1","summary":[{"type":"summary_text","text":"Run both at once."}]},
		{"type":"message","role":"assistant","id":"item_m1","content":[{"type":"output_text","text":"Running both."}]},
		{"type":"function_call","call_id":"bash:0","name":"bash","arguments":"{\"cmd\":\"ls\"}"},
		{"type":"function_call","call_id":"bash:1","name":"bash","arguments":"{\"cmd\":\"pwd\"}"},
		{"type":"function_call_output","call_id":"bash:0","output":"a.go"},
		{"type":"function_call_output","call_id":"bash:1","output":"/app"}
	],
	"tools":[{"type":"function","name":"bash","parameters":{"type":"object"}}]
}`)
	var wire struct {
		Messages []messagesWireMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("upstream body: %v: %s", err, body)
	}
	for index, message := range wire.Messages {
		ids := map[string]bool{}
		for _, block := range message.Content {
			if block.Type == "tool_result" {
				ids[block.ToolUseID] = true
			}
		}
		if len(ids) == 0 {
			continue
		}
		if !ids["bash:0"] || !ids["bash:1"] {
			t.Fatalf("message %d carries tool results %v, want both bash:0 and bash:1 together: %s", index, ids, body)
		}
		calls := callsOf(wire.Messages[index-1])
		if !calls["bash:0"] || !calls["bash:1"] {
			t.Fatalf("the assistant message before the results calls %v, want both bash:0 and bash:1: %s", calls, body)
		}
		return
	}
	t.Fatalf("no tool results were dispatched: %s", body)
}
