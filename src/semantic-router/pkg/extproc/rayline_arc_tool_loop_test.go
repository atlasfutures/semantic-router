package extproc

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func toolCallMessage(ids ...string) llmprotocol.Message {
	message := llmprotocol.Message{Role: llmprotocol.RoleAssistant}
	for _, id := range ids {
		message.Content = append(message.Content, llmprotocol.Content{
			Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: id, Name: "read"},
		})
	}
	return message
}

func toolResultMessage(role llmprotocol.Role, id string, extra ...llmprotocol.Content) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: append([]llmprotocol.Content{{
		Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: id},
	}}, extra...)}
}

func userText(text string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
}

func TestAnswersToolCall(t *testing.T) {
	reminder := llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "<system-reminder>"}
	cases := []struct {
		name     string
		messages []llmprotocol.Message
		want     bool
	}{
		{"an Anthropic tool_result", []llmprotocol.Message{
			userText("fix it"), toolCallMessage("c1"), toolResultMessage(llmprotocol.RoleUser, "c1"),
		}, true},
		{"an Anthropic tool_result beside text", []llmprotocol.Message{
			userText("fix it"), toolCallMessage("c1"), toolResultMessage(llmprotocol.RoleUser, "c1", reminder),
		}, true},
		{"the second of two Chat tool messages", []llmprotocol.Message{
			userText("fix it"), toolCallMessage("c1", "c2"),
			toolResultMessage(llmprotocol.RoleTool, "c1"), toolResultMessage(llmprotocol.RoleTool, "c2"),
		}, true},
		{"Responses function calls as separate items", []llmprotocol.Message{
			userText("fix it"), toolCallMessage("c1"), toolCallMessage("c2"),
			toolResultMessage(llmprotocol.RoleTool, "c1"), toolResultMessage(llmprotocol.RoleTool, "c2"),
		}, true},
		{"a result whose call is in stored response state", []llmprotocol.Message{
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "c1", DeferredLink: true},
			}}},
		}, true},
		{"a new user turn", []llmprotocol.Message{
			userText("fix it"), toolCallMessage("c1"), toolResultMessage(llmprotocol.RoleUser, "c1"),
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "done"}}},
			userText("thanks, now the docs"),
		}, false},
		{"a result for a call made before the last user message", []llmprotocol.Message{
			toolCallMessage("c1"), userText("wait"), toolResultMessage(llmprotocol.RoleUser, "c1"),
		}, false},
		{"a result no call in the turn asked for", []llmprotocol.Message{
			userText("fix it"), toolCallMessage("c1"), toolResultMessage(llmprotocol.RoleUser, "c9"),
		}, false},
		{"no messages", nil, false},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if got := answersToolCall(test.messages); got != test.want {
				t.Fatalf("answersToolCall = %v, want %v", got, test.want)
			}
		})
	}
}

func TestArmFamilyIsTheProviderModelVendor(t *testing.T) {
	router := &OpenAIRouter{}
	for name, want := range map[string]string{
		"deepseek/deepseek-v4-flash@thinking-on":  "deepseek",
		"deepseek/deepseek-v4-pro@thinking-off":   "deepseek",
		"xiaomi/mimo-v2.5-pro@thinking-off":       "xiaomi",
		"Z-AI/glm-5.2":                            "z-ai",
		"claude-opus-5@thinking-off":              "claude-opus-5",
		"worker-a":                                "worker-a",
		"anthropic/claude-opus-5":                 "anthropic",
		"qwen/qwen3.6-35b-a3b@thinking-off":       "qwen",
		"deepseek/deepseek-v4-flash@thinking-off": "deepseek",
	} {
		if got := router.armFamily(name); got != want {
			t.Fatalf("armFamily(%q) = %q, want %q", name, got, want)
		}
	}
}

func toolLoopRequestContext(messages ...llmprotocol.Message) *RequestContext {
	return &RequestContext{SemanticRequest: &llmprotocol.Request{Messages: messages}}
}

func TestToolLoopForeignArms(t *testing.T) {
	refs := []config.ModelRef{
		{Model: "deepseek/deepseek-v4-pro@thinking-off"},
		{Model: "xiaomi/mimo-v2.5-pro@thinking-off"},
		{Model: "deepseek/deepseek-v4-flash@thinking-on"},
	}
	midLoop := toolLoopRequestContext(userText("fix it"), toolCallMessage("c1"), toolResultMessage(llmprotocol.RoleUser, "c1"))
	newTurn := toolLoopRequestContext(userText("fix it"))
	on := &config.RaylineARCAlgorithmConfig{HoldFamilyInToolLoop: true}
	previous := 2
	state, err := raylinearc.NewEpisodeState(len(refs))
	if err != nil {
		t.Fatal(err)
	}
	state.PreviousArm = &previous
	fresh, _ := raylinearc.NewEpisodeState(len(refs))
	router := &OpenAIRouter{}

	foreign, arm, family := router.toolLoopForeignArms(on, midLoop, state, refs)
	if !reflect.DeepEqual(foreign, []bool{false, true, false}) || arm != 2 || family != "deepseek" {
		t.Fatalf("mid-loop = %v arm %d family %q, want only the other family excluded", foreign, arm, family)
	}
	for name, test := range map[string]struct {
		arc   *config.RaylineARCAlgorithmConfig
		ctx   *RequestContext
		state *raylinearc.EpisodeState
		refs  []config.ModelRef
	}{
		"the hold is off":          {&config.RaylineARCAlgorithmConfig{}, midLoop, state, refs},
		"a new user turn":          {on, newTurn, state, refs},
		"no previous arm":          {on, midLoop, fresh, refs},
		"every arm is one family":  {on, midLoop, state, []config.ModelRef{{Model: "deepseek/a"}, {Model: "deepseek/b"}, {Model: "deepseek/c"}}},
		"a previous arm off range": {on, midLoop, state, refs[:2]},
	} {
		if foreign, _, _ := router.toolLoopForeignArms(test.arc, test.ctx, test.state, test.refs); foreign != nil {
			t.Fatalf("%s: foreign = %v, want nil", name, foreign)
		}
	}
}

// Mid-loop, the artifact scores only the family that opened the loop.
func TestRaylineARCSelectorHoldsTheToolLoopFamily(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())
	selectionContext := armConstraintContext(state, false, nil, nil)
	selectionContext.RaylineARC.ToolLoopForeignArms = []bool{false, true}
	if _, err := selector.Select(context.Background(), selectionContext); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{false, true}) {
		t.Fatalf("scorer exclusion = %v, want the other family excluded", scorer.excluded)
	}
}

// The hold yields to the hard constraints: when the family has no eligible
// arm it is lifted, and the turn is scored as if there were no loop.
func TestRaylineARCSelectorLiftsTheHoldWhenTheFamilyHasNoEligibleArm(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	scorer.decision.SelectedArm, scorer.decision.SelectedWorker = 1, "worker-b"
	selector := armedVisionSelector(scorer, visionEncoder())
	selectionContext := armConstraintContext(state, false, nil, []bool{true, false})
	selectionContext.RaylineARC.ToolLoopForeignArms = []bool{false, true}
	if _, err := selector.Select(context.Background(), selectionContext); err != nil {
		t.Fatalf("a lifted hold refused the turn: %v", err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{true, false}) {
		t.Fatalf("scorer exclusion = %v, want only the disabled arm", scorer.excluded)
	}
}

// The policy service is offered only the actions of the loop's family.
func TestPolicySelectorOffersOnlyTheToolLoopFamily(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	state, _ := raylinearc.NewEpisodeState(2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "go"})
	selectionContext := &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64), State: state, RawRequest: body,
			RequestFormat: policyFormatAnthropic, ToolLoopForeignArms: []bool{false, true},
		},
	}
	if _, err := fixture.selector.Select(context.Background(), selectionContext); err != nil {
		t.Fatalf("select: %v", err)
	}
	offered := fixture.fake.received()[0].Selection.AvailableActionIDs
	if len(offered) != 2 || offered[0] != bindings[0].ActionID || offered[1] != bindings[1].ActionID {
		t.Fatalf("mid-loop turn offered %v, want the think worker's two levels", offered)
	}
}

// End to end on a policy cell with the hold on: a Claude arm opens a tool
// loop, the tool result is decided only among Claude's arms, and the next
// user turn may leave the family.
func TestToolLoopStaysOnItsFamilyEndToEnd(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	path := writePolicyDispatchConfig(t, fake.URL(), actions)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	anchor := "        rayline_arc:\n"
	if !strings.Contains(string(raw), anchor) {
		t.Fatal("the policy e2e config no longer carries the rayline_arc anchor")
	}
	rendered := strings.Replace(string(raw), anchor, anchor+"          hold_family_in_tool_loop: true\n", 1)
	if writeErr := os.WriteFile(path, []byte(rendered), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)

	const episode = "episode-tool-loop"
	opening := `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"fix the failing test"}]}`
	toolTurn(t, router, fake, actions["claude"].ActionID, episode, opening)

	loop := `{"model":"auto","max_tokens":1024,"messages":[` +
		`{"role":"user","content":"fix the failing test"},` +
		`{"role":"assistant","content":[{"type":"thinking","thinking":"read it first","signature":"sig"},` +
		`{"type":"tool_use","id":"toolu_1","name":"read","input":{"path":"a.go"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"package a"}]}]}`
	toolTurn(t, router, fake, actions["claude-off"].ActionID, episode, loop)
	offered := fake.received()[len(fake.received())-1].Selection.AvailableActionIDs
	want := []string{actions["claude"].ActionID, actions["claude-off"].ActionID}
	slices.Sort(offered)
	slices.Sort(want)
	if !slices.Equal(offered, want) {
		t.Fatalf("mid-loop turn offered %v, want only Claude's actions %v", offered, want)
	}

	next := `{"model":"auto","max_tokens":1024,"messages":[` +
		`{"role":"user","content":"fix the failing test"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read","input":{"path":"a.go"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"package a"}]},` +
		`{"role":"assistant","content":"Fixed."},` +
		`{"role":"user","content":"now the docs"}]}`
	toolTurn(t, router, fake, actions["think"].ActionID, episode, next)
	if offered := fake.received()[len(fake.received())-1].Selection.AvailableActionIDs; len(offered) != len(actions) {
		t.Fatalf("a new user turn offered %v, want every action", offered)
	}
}

// toolTurn dispatches one client Messages body on the policy e2e cell with
// the service choosing action, and commits it as a complete 200.
func toolTurn(t *testing.T, router *OpenAIRouter, fake *fakePolicyService, action, episode, client string) {
	t.Helper()
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: fmt.Sprintf("tool-loop-%d", time.Now().UnixNano()),
		StartTime: time.Now(), TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/messages"},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: episode},
		}},
	}}
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(client), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		t.Fatalf("request refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	if _, err := router.handleResponseHeaders(arcResponseHeaders("200"), ctx); err != nil {
		t.Fatalf("commit: %v", err)
	}
	completeTestResponse(t, ctx)
	finalizeSelectionProcessTerminal(ctx)
}
