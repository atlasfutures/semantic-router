package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The thinking-signature guard, end to end through the request phases of a
// policy cell whose Claude arms are reached over Messages.

func thinkingBlock(signature string) string {
	return `{"type":"thinking","thinking":"thought ` + signature + `","signature":"` + signature + `"}`
}

func assistantTurn(blocks ...string) string {
	return `{"role":"assistant","content":[` + strings.Join(blocks, ",") + `]}`
}

func messagesHistory(messages ...string) string {
	return `{"model":"auto","max_tokens":4096,"messages":[` + strings.Join(messages, ",") + `]}`
}

// twoTurnHistory has two earlier assistant turns, each thinking under its own
// signature, and ends on a plain user message.
func twoTurnHistory(first, second string) string {
	return messagesHistory(
		`{"role":"user","content":"start"}`,
		assistantTurn(thinkingBlock(first), `{"type":"text","text":"turn one"}`),
		`{"role":"user","content":"next"}`,
		assistantTurn(thinkingBlock(second), `{"type":"text","text":"turn two"}`),
		`{"role":"user","content":"go on"}`,
	)
}

// toolLoopHistory has an earlier assistant turn, then an assistant message
// whose tool call the last message answers.
func toolLoopHistory(earlier, loop string) string {
	return messagesHistory(
		`{"role":"user","content":"start"}`,
		assistantTurn(thinkingBlock(earlier), `{"type":"text","text":"turn one"}`),
		`{"role":"user","content":"run the tests"}`,
		assistantTurn(thinkingBlock(loop), `{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"go test"}}`),
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}`,
	)
}

func newThinkingGuardRouter(t *testing.T, action string) (*OpenAIRouter, *fakePolicyService) {
	t.Helper()
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	chosen := actions[action].ActionID
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return chosen })
	router, err := NewOpenAIRouter(writeConsistentPolicyConfig(t, fake.URL(), config.RaylineARCConsistencyStrict))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	return router, fake
}

// seedThinkingRun commits an episode whose thinking run on worker is anchored
// at signature.
func seedThinkingRun(t *testing.T, router *OpenAIRouter, episode, worker, signature string) {
	t.Helper()
	store := router.RaylineARCEpisodeStore
	lease, state, err := store.Prepare(context.Background(), raylinearc.HashEpisodeID(episode), 4)
	if err != nil {
		t.Fatal(err)
	}
	state.ReasoningProvenance = reasoning.WithThinkingRun(state.ReasoningProvenance, worker, nil, false,
		protocolcodec.SignatureAnchorKey(signature))
	if err := state.Commit(0, 10, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	if err := store.Commit(context.Background(), lease, lease.Version(), state); err != nil {
		t.Fatal(err)
	}
}

// armThinkingRunWorker is the run worker of an arm as the router dispatches it.
func armThinkingRunWorker(t *testing.T, router *OpenAIRouter, arm string) string {
	t.Helper()
	dispatch, err := router.resolveProviderDispatch(arm, "", false, llmprotocol.AnthropicMessagesV1)
	if err != nil {
		t.Fatal(err)
	}
	return router.thinkingRunWorker(dispatch.backendName, dispatch.upstreamModel)
}

// dispatchThinkingTurn runs one Messages client turn through the request
// phases and returns its context and the provider-bound body.
func dispatchThinkingTurn(t *testing.T, router *OpenAIRouter, episode, client string) (*RequestContext, map[string]json.RawMessage) {
	t.Helper()
	return dispatchThinkingTurnAt(t, router, episode, "/v1/messages", client)
}

// dispatchThinkingTurnAt is dispatchThinkingTurn for a client sending to path.
func dispatchThinkingTurnAt(t *testing.T, router *OpenAIRouter, episode, path, client string) (*RequestContext, map[string]json.RawMessage) {
	t.Helper()
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "thinking-guard-" + episode,
		StartTime: time.Now(), TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: path},
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
	var body map[string]json.RawMessage
	if err := json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("provider body: %v", err)
	}
	return ctx, body
}

func assertSignatures(t *testing.T, label string, messages json.RawMessage, kept, stripped []string) {
	t.Helper()
	wire := string(messages)
	for _, signature := range kept {
		if !strings.Contains(wire, `"`+signature+`"`) {
			t.Fatalf("%s: thinking %s was stripped: %s", label, signature, wire)
		}
	}
	for _, signature := range stripped {
		if strings.Contains(wire, signature) {
			t.Fatalf("%s: thinking %s was sent: %s", label, signature, wire)
		}
	}
	for _, text := range []string{"turn one"} {
		if !strings.Contains(wire, text) {
			t.Fatalf("%s: the assistant's visible text was lost: %s", label, wire)
		}
	}
}

// Thinking the cell did not serve on the target's model -- signed before the
// conversation reached the cell, or by another Claude model -- is stripped
// before a Claude worker is sent it; the episode's own run on the model is
// kept, from its anchor on. Arms that differ only in their thinking settings
// share the run.
func TestClaudeWorkerGetsOnlyItsOwnRunsThinking(t *testing.T) {
	router, _ := newThinkingGuardRouter(t, "claude")
	other := reasoning.ThinkingRunWorker("openrouter\x00https://openrouter.ai/api/v1\x00anthropic/claude-sonnet-5.5")
	for _, tc := range []struct {
		name            string
		seedWorker      string
		anchor          string
		kept, stripped  []string
		wantStrippedLog string
	}{
		{name: "no record", kept: nil, stripped: []string{"sig-first", "sig-second"}, wantStrippedLog: "2"},
		{name: "run from the second turn", seedWorker: "claude", anchor: "sig-second",
			kept: []string{"sig-second"}, stripped: []string{"sig-first"}, wantStrippedLog: "1"},
		{name: "run from the first turn", seedWorker: "claude", anchor: "sig-first",
			kept: []string{"sig-first", "sig-second"}},
		{name: "a thinking-off arm of the same model", seedWorker: "claude-off", anchor: "sig-first",
			kept: []string{"sig-first", "sig-second"}},
		{name: "another model's run", seedWorker: "other", anchor: "sig-first",
			stripped: []string{"sig-first", "sig-second"}, wantStrippedLog: "2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logs := captureLogs(t)
			episode := "episode-thinking-run-" + strings.ReplaceAll(tc.name, " ", "-")
			switch tc.seedWorker {
			case "":
			case "other":
				seedThinkingRun(t, router, episode, other, tc.anchor)
			default:
				seedThinkingRun(t, router, episode, armThinkingRunWorker(t, router, tc.seedWorker), tc.anchor)
			}
			_, body := dispatchThinkingTurn(t, router, episode, twoTurnHistory("sig-first", "sig-second"))
			assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
			assertSignatures(t, tc.name, body["messages"], tc.kept, tc.stripped)
			if !strings.Contains(string(body["messages"]), "turn two") {
				t.Fatalf("the assistant's visible text was lost: %s", body["messages"])
			}
			for _, entry := range logs.All() {
				fields := entry.ContextMap()
				if fields["event"] != "unserved_reasoning_stripped" {
					continue
				}
				if fmt.Sprint(fields["stripped"]) != tc.wantStrippedLog {
					t.Fatalf("unserved_reasoning_stripped = %v, want stripped %s", fields, tc.wantStrippedLog)
				}
				return
			}
			if tc.wantStrippedLog != "" {
				t.Fatal("no unserved_reasoning_stripped event")
			}
		})
	}
}

// The cell's own minted marker never reaches a Claude worker, whatever the
// episode records.
func TestMintedThinkingMarkerIsStrippedForClaude(t *testing.T) {
	router, _ := newThinkingGuardRouter(t, "claude")
	marker := protocolcodec.ThinkingMarker("moonshotai", "kimi thought")
	history := messagesHistory(
		`{"role":"user","content":"start"}`,
		assistantTurn(`{"type":"thinking","thinking":"kimi thought","signature":"`+marker+`"}`, `{"type":"text","text":"turn one"}`),
		`{"role":"user","content":"go on"}`,
	)
	_, body := dispatchThinkingTurn(t, router, "episode-thinking-marker", history)
	wire := string(body["messages"])
	if strings.Contains(wire, "vsr.thinking") || strings.Contains(wire, "kimi thought") {
		t.Fatalf("a minted marker reached Claude: %s", wire)
	}
	if !strings.Contains(wire, "turn one") {
		t.Fatalf("the assistant's visible text was lost: %s", wire)
	}
}

// The active tool loop's latest thinking is never stripped before a Claude
// send: Anthropic requires it. Sent unserved, it starts the next run.
func TestActiveToolLoopThinkingIsKept(t *testing.T) {
	router, _ := newThinkingGuardRouter(t, "claude")
	for _, tc := range []struct {
		name     string
		anchor   string
		wantKeep bool
	}{
		{name: "served", anchor: "sig-loop", wantKeep: true},
		{name: "unserved"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			episode := "episode-thinking-loop-" + tc.name
			if tc.anchor != "" {
				seedThinkingRun(t, router, episode, armThinkingRunWorker(t, router, "claude"), tc.anchor)
			}
			ctx, body := dispatchThinkingTurn(t, router, episode, toolLoopHistory("sig-earlier", "sig-loop"))
			assertSignatures(t, tc.name, body["messages"], []string{"sig-loop"}, []string{"sig-earlier"})
			plan := ctx.ThinkingSignatureRetry
			if plan == nil {
				t.Fatal("no re-send was staged for a Claude dispatch over Messages")
			}
			wantAdopted := map[bool]string{true: "", false: protocolcodec.SignatureAnchorKey("sig-loop")}[tc.wantKeep]
			if adopted := ctx.RaylineARCTransaction.thinkingAdopted; adopted != wantAdopted {
				t.Fatalf("adopted = %q, want %q: a sent unserved loop message starts the next run", adopted, wantAdopted)
			}
			_ = plan
		})
	}
}

// A committed Claude turn anchors a new run at its first signature when the
// history held no anchor, keeps an anchored run as it is, and ends the runs
// of the cell's other Claude models.
func TestCommittedClaudeTurnAnchorsItsRun(t *testing.T) {
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	episode := raylinearc.HashEpisodeID(t.Name())
	commit := func(worker string, others []string, anchored bool, signature string) []reasoning.ReasoningProvenance {
		t.Helper()
		lease, state, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		transaction := newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil)
		transaction.markSelection(0, 10)
		transaction.thinkingWorker, transaction.thinkingOthers, transaction.thinkingAnchored = worker, others, anchored
		response := &llmprotocol.Response{Output: []llmprotocol.OutputItem{{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentReasoning, Text: "thought", Signature: signature},
			{Kind: llmprotocol.ContentText, Text: "answer"},
		}}}}
		if err := transaction.commit(context.Background(), &RequestContext{SemanticResponse: response}); err != nil {
			t.Fatalf("commit: %v", err)
		}
		next, reread, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		_ = store.Abort(context.Background(), next)
		return reread.ReasoningProvenance
	}
	sonnet, opus := reasoning.ThinkingRunWorker("sonnet"), reasoning.ThinkingRunWorker("opus")
	digest := func(signature string) string {
		return reasoning.ThinkingRunAnchorDigest(protocolcodec.SignatureAnchorKey(signature))
	}

	provenance := commit(sonnet, []string{opus}, false, "sig-1")
	if anchors := reasoning.ThinkingRunAnchors(provenance, sonnet); fmt.Sprint(anchors) != fmt.Sprint([]string{digest("sig-1")}) {
		t.Fatalf("first turn anchors = %v", anchors)
	}
	provenance = commit(sonnet, []string{opus}, true, "sig-2")
	if anchors := reasoning.ThinkingRunAnchors(provenance, sonnet); fmt.Sprint(anchors) != fmt.Sprint([]string{digest("sig-1")}) {
		t.Fatalf("an anchored turn moved the anchor: %v", anchors)
	}
	provenance = commit(opus, []string{sonnet}, false, "sig-3")
	if anchors := reasoning.ThinkingRunAnchors(provenance, sonnet); len(anchors) != 0 {
		t.Fatalf("a turn on another model left the run's anchors: %v", anchors)
	}
	if anchors := reasoning.ThinkingRunAnchors(provenance, opus); fmt.Sprint(anchors) != fmt.Sprint([]string{digest("sig-3")}) {
		t.Fatalf("the other model's run = %v", anchors)
	}
}

// A turn that sent an unserved tool loop message anchors the new run there,
// and a turn re-sent with its run's thinking stripped starts the run again,
// at the loop message the re-send kept or else at its own reply.
func TestThinkingRunAdoptsAndRestarts(t *testing.T) {
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	episode := raylinearc.HashEpisodeID(t.Name())
	worker := reasoning.ThinkingRunWorker("opus")
	commit := func(anchored, restart bool, adopted string) []string {
		t.Helper()
		lease, state, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		transaction := newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil)
		transaction.markSelection(0, 10)
		transaction.thinkingWorker, transaction.thinkingAnchored = worker, anchored
		transaction.thinkingRestart, transaction.thinkingAdopted = restart, adopted
		response := &llmprotocol.Response{Output: []llmprotocol.OutputItem{{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentReasoning, Text: "thought", Signature: "sig-reply"},
		}}}}
		if err := transaction.commit(context.Background(), &RequestContext{SemanticResponse: response}); err != nil {
			t.Fatalf("commit: %v", err)
		}
		next, reread, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		_ = store.Abort(context.Background(), next)
		return reasoning.ThinkingRunAnchors(reread.ReasoningProvenance, worker)
	}
	digest := reasoning.ThinkingRunAnchorDigest
	if got := commit(false, false, "sig-loop"); fmt.Sprint(got) != fmt.Sprint([]string{digest("sig-loop")}) {
		t.Fatalf("adopted run = %v", got)
	}
	if got := commit(true, true, ""); fmt.Sprint(got) != fmt.Sprint([]string{digest(protocolcodec.SignatureAnchorKey("sig-reply"))}) {
		t.Fatalf("restarted run = %v, want only the re-send's reply", got)
	}
	if got := commit(true, true, "sig-kept"); fmt.Sprint(got) != fmt.Sprint([]string{digest("sig-kept")}) {
		t.Fatalf("restarted run = %v, want only the kept loop message", got)
	}
}

// Anthropic's and OpenRouter's signature refusals are recognised; nothing else
// is, other 400s included.
func TestThinkingSignatureRefusalIsRecognised(t *testing.T) {
	for body, want := range map[string]bool{
		`{"type":"error","error":{"type":"invalid_request_error","message":"messages.19.content.12: Invalid ` + "`signature`" + ` in ` + "`thinking`" + ` block"}}`:                                                          true,
		`{"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\"type\":\"error\",\"error\":{\"message\":\"messages.3.content.0: Invalid ` + "`signature`" + ` in ` + "`thinking`" + ` block\"}}"}}}`: true,
		`{"error":{"message":"The request contains encrypted reasoning or compaction content that can only be processed by the provider that created it (anthropic) or by azure/global, claude-on-aws"}}`:                    true,
		`{"error":{"message":"max_tokens: 99999999 > 64000, which is the maximum allowed"}}`:                                                                                                                                 false,
		`{"error":{"message":"messages.1.content.0.type: Expected ` + "`thinking`" + ` or ` + "`redacted_thinking`" + `, but found ` + "`text`" + `"}}`:                                                                      false,
		`not json`: false,
	} {
		if got := thinkingSignatureRejected([]byte(body)); got != want {
			t.Fatalf("%s: recognised = %v, want %v", body, got, want)
		}
	}
}

// The re-send body strips the thinking of every assistant message, the
// active loop's included: a kept block's signature covers what precedes it.
// Thinking is turned off, with the clear_thinking edit and the effort that
// require it, only where the model can turn it off.
func TestThinkingRetryBodyStripsEveryTurn(t *testing.T) {
	var client map[string]json.RawMessage
	if err := json.Unmarshal([]byte(toolLoopHistory("sig-earlier", "sig-loop")), &client); err != nil {
		t.Fatal(err)
	}
	client["thinking"] = json.RawMessage(`{"type":"adaptive"}`)
	client["output_config"] = json.RawMessage(`{"effort":"max"}`)
	client["context_management"] = json.RawMessage(`{"edits":[{"type":"clear_thinking_20251015","keep":"all"},{"type":"clear_tool_uses_20250919"}]}`)
	dispatched, _ := json.Marshal(client)

	disabled, change, ok := thinkingRetryBody(dispatched, true)
	if !ok || change.stripped != 2 || !change.thinkingDisabled {
		t.Fatalf("can disable: ok=%v change=%+v", ok, change)
	}
	var members map[string]json.RawMessage
	if err := json.Unmarshal(disabled, &members); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(members["messages"]), `"thinking"`) || !strings.Contains(string(members["messages"]), "toolu_1") {
		t.Fatalf("messages: %s", members["messages"])
	}
	if !sameJSON(t, members["thinking"], `{"type":"disabled"}`) ||
		!sameJSON(t, members["context_management"], `{"edits":[{"type":"clear_tool_uses_20250919"}]}`) {
		t.Fatalf("thinking %s, context_management %s", members["thinking"], members["context_management"])
	}
	if _, kept := members["output_config"]; kept {
		t.Fatalf("an effort that requires thinking survived thinking off: %s", members["output_config"])
	}

	alwaysOn, change, ok := thinkingRetryBody(dispatched, false)
	if !ok || change.stripped != 2 || change.thinkingDisabled ||
		!strings.Contains(string(alwaysOn), `"adaptive"`) || strings.Contains(string(alwaysOn), "sig-loop") {
		t.Fatalf("always on: ok=%v change=%+v body=%s", ok, change, alwaysOn)
	}

	plain, _ := json.Marshal(map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}})
	if _, _, ok := thinkingRetryBody(plain, true); ok {
		t.Fatal("a body with no thinking was re-sent")
	}
}

// claudeUpstream is a fake Anthropic upstream: it records every call and
// answers each with the configured status and body.
type claudeUpstream struct {
	server   *httptest.Server
	mu       sync.Mutex
	bodies   [][]byte
	status   int
	response string
}

func newClaudeUpstream(t *testing.T, status int, response string) *claudeUpstream {
	t.Helper()
	upstream := &claudeUpstream{status: status, response: response}
	upstream.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received, _ := io.ReadAll(r.Body)
		upstream.mu.Lock()
		upstream.bodies = append(upstream.bodies, received)
		upstream.mu.Unlock()
		w.Header().Set("content-type", "application/json")
		w.WriteHeader(upstream.status)
		_, _ = io.WriteString(w, upstream.response)
	}))
	t.Cleanup(upstream.server.Close)
	return upstream
}

func (upstream *claudeUpstream) calls() [][]byte {
	upstream.mu.Lock()
	defer upstream.mu.Unlock()
	return append([][]byte(nil), upstream.bodies...)
}

const claudeServedReply = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5",` +
	`"content":[{"type":"text","text":"served after the strip"}],"stop_reason":"end_turn",` +
	`"usage":{"input_tokens":10,"output_tokens":5}}`

const signatureRefusal = `{"type":"error","error":{"type":"invalid_request_error",` +
	`"message":"messages.1.content.0: Invalid ` + "`signature`" + ` in ` + "`thinking`" + ` block"}}`

// refusedThinkingTurn dispatches a tool-loop turn on the claude arm, points
// its staged re-send at upstream, and answers the dispatch with status and
// body the way FULL_DUPLEX_STREAMED delivers them: the headers, then the
// body, which the Router may read before it answers the headers.
func refusedThinkingTurn(
	t *testing.T,
	episode string,
	upstream *claudeUpstream,
	status int,
	body string,
	anchor string,
) (*RequestContext, *MockStream) {
	t.Helper()
	router, _ := newThinkingGuardRouter(t, "claude")
	if anchor != "" {
		seedThinkingRun(t, router, episode, armThinkingRunWorker(t, router, "claude"), anchor)
	}
	ctx, _ := dispatchThinkingTurn(t, router, episode, toolLoopHistory("sig-earlier", "sig-loop"))
	return ctx, answerRefusal(t, router, ctx, upstream, status, body)
}

// answerRefusal points a dispatched turn's staged re-send at upstream and
// answers the dispatch with status and body the way FULL_DUPLEX_STREAMED
// delivers them, returning what the Router sent Envoy.
func answerRefusal(t *testing.T, router *OpenAIRouter, ctx *RequestContext, upstream *claudeUpstream, status int, body string) *MockStream {
	t.Helper()
	plan := ctx.ThinkingSignatureRetry
	if plan == nil {
		t.Fatal("no re-send was staged")
	}
	dispatch := *plan.dispatch
	profile := *dispatch.profile
	profile.BaseURL = upstream.server.URL
	dispatch.profile = &profile
	dispatch.backendAddress = upstream.server.Listener.Addr().String()
	plan.dispatch = &dispatch

	ctx.FullDuplexResponseBody = true
	upstreamStream := NewMockStream([]*ext_proc.ProcessingRequest{{
		Request: &ext_proc.ProcessingRequest_ResponseBody{ResponseBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true}},
	}})
	ctx.streamReceiver = startStreamReceiver(upstreamStream)
	envoy := NewMockStream(nil)
	headers := &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", RawValue: []byte(strconv.Itoa(status))},
			{Key: "content-type", RawValue: []byte("application/json")},
		}},
	}}
	if err := router.processResponseHeaders(envoy, headers, ctx); err != nil {
		t.Fatalf("processResponseHeaders: %v", err)
	}
	if len(envoy.Responses) != 1 {
		t.Fatalf("the Router sent %d replies, want 1", len(envoy.Responses))
	}
	return envoy
}

// A signature refusal is re-sent exactly once, before anything reaches the
// client, with no thinking left in any turn, even the run's own, and the
// client gets the re-send's reply. Both calls are marked on llm_usage.
func TestSignatureRefusalIsResentOnceWithoutEarlierThinking(t *testing.T) {
	logs := captureLogs(t)
	upstream := newClaudeUpstream(t, http.StatusOK, claudeServedReply)
	ctx, envoy := refusedThinkingTurn(t, "episode-thinking-retry", upstream, http.StatusBadRequest, signatureRefusal, "sig-earlier")
	if transaction := ctx.RaylineARCTransaction; !transaction.thinkingRestart || transaction.thinkingAdopted != "" {
		t.Fatalf("restart = %v, adopted = %q: the run must start again at the re-send's reply",
			transaction.thinkingRestart, transaction.thinkingAdopted)
	}

	immediate := envoy.Responses[0].GetImmediateResponse()
	if immediate == nil || int(immediate.GetStatus().GetCode()) != http.StatusOK ||
		!strings.Contains(string(immediate.GetBody()), "served after the strip") {
		t.Fatalf("the client did not get the re-send's reply: %v", envoy.Responses[0])
	}
	calls := upstream.calls()
	if len(calls) != 1 {
		t.Fatalf("the upstream was called %d times, want exactly once", len(calls))
	}
	var sent map[string]json.RawMessage
	if err := json.Unmarshal(calls[0], &sent); err != nil {
		t.Fatal(err)
	}
	assertSignatures(t, "re-send", sent["messages"], nil, []string{"sig-earlier", `"sig-loop"`})

	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want one per call: %v", len(lines), lines)
	}
	assertFields(t, "refused call", lines[0], map[string]interface{}{"thinking_signature_retry": thinkingRetryAttempt})
	if message, _ := lines[0]["upstream_error_message"].(string); !strings.Contains(message, "Invalid `signature` in `thinking` block") {
		t.Fatalf("the refused call's text = %q", message)
	}
	assertFields(t, "re-send", lines[1], map[string]interface{}{"thinking_signature_retry": thinkingRetryResend})
	event := findLogEvent(t, logs, "thinking_signature_retry")
	assertFields(t, "thinking_signature_retry", event, map[string]interface{}{
		"refused_status": http.StatusBadRequest, "resend_status": http.StatusOK, "stripped": 2,
	})
}

// A re-send refused again is the client's answer: there is no second.
func TestSignatureRefusalIsNeverResentTwice(t *testing.T) {
	upstream := newClaudeUpstream(t, http.StatusBadRequest, signatureRefusal)
	_, envoy := refusedThinkingTurn(t, "episode-thinking-twice", upstream, http.StatusBadRequest, signatureRefusal, "")
	if immediate := envoy.Responses[0].GetImmediateResponse(); immediate == nil ||
		int(immediate.GetStatus().GetCode()) != http.StatusBadRequest {
		t.Fatalf("the re-send's refusal did not reach the client: %v", envoy.Responses[0])
	}
	calls := upstream.calls()
	if len(calls) != 1 {
		t.Fatalf("the upstream was called %d times, want exactly once", len(calls))
	}
	if strings.Contains(string(calls[0]), "sig-") {
		t.Fatalf("an unserved loop's thinking was re-sent: %s", calls[0])
	}
}

// Any other 400 is not re-sent: the headers are answered as they came, and
// the body the Router read ahead is handed back, in order, untouched.
func TestOtherRefusalIsNotResent(t *testing.T) {
	upstream := newClaudeUpstream(t, http.StatusOK, claudeServedReply)
	refusal := `{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: too large"}}`
	ctx, envoy := refusedThinkingTurn(t, "episode-thinking-other", upstream, http.StatusBadRequest, refusal, "")
	if envoy.Responses[0].GetImmediateResponse() != nil || envoy.Responses[0].GetResponseHeaders() == nil {
		t.Fatalf("a non-signature 400 was replaced: %v", envoy.Responses[0])
	}
	if calls := upstream.calls(); len(calls) != 0 {
		t.Fatalf("a non-signature 400 was re-sent %d times", len(calls))
	}
	message, ok := ctx.streamReceiver.poll()
	if !ok || string(message.request.GetResponseBody().GetBody()) != refusal {
		t.Fatalf("the body read ahead was not handed back: %v %v", ok, message.request)
	}
}

// A refusal that names the loop's own block is re-sent without it, even when
// the run covers it, and with thinking off.
func TestRefusalOfTheLoopsOwnBlockStripsIt(t *testing.T) {
	upstream := newClaudeUpstream(t, http.StatusOK, claudeServedReply)
	refusal := strings.Replace(signatureRefusal, "messages.1.content.0", "messages.3.content.0", 1)
	_, envoy := refusedThinkingTurn(t, "episode-thinking-own-block", upstream, http.StatusBadRequest, refusal, "sig-loop")
	if envoy.Responses[0].GetImmediateResponse() == nil {
		t.Fatalf("the refusal was not re-sent: %v", envoy.Responses[0])
	}
	calls := upstream.calls()
	if len(calls) != 1 || strings.Contains(string(calls[0]), "sig-loop") {
		t.Fatalf("re-sends = %d, body %s", len(calls), calls)
	}
}

// OpenRouter's encrypted-reasoning refusal names no block, so the re-send
// strips the loop's own thinking too.
func TestUnindexedRefusalStripsTheLoopsBlock(t *testing.T) {
	upstream := newClaudeUpstream(t, http.StatusOK, claudeServedReply)
	refusal := `{"error":{"message":"The request contains encrypted reasoning or compaction content that can only be processed by the provider that created it (anthropic)"}}`
	_, envoy := refusedThinkingTurn(t, "episode-thinking-unindexed", upstream, http.StatusBadRequest, refusal, "sig-earlier")
	calls := upstream.calls()
	if envoy.Responses[0].GetImmediateResponse() == nil || len(calls) != 1 || strings.Contains(string(calls[0]), "sig-") {
		t.Fatalf("re-sends = %d: %s", len(calls), calls)
	}
}

// A run adopted at a redacted-only loop message is found by its data.
func TestRunAnchoredAtRedactedThinkingIsFound(t *testing.T) {
	messages := []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "start"}}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentUnmodeled, Unmodeled: &llmprotocol.UnmodeledBlock{
			Format: llmprotocol.AnthropicMessagesV1, Type: "redacted_thinking", Raw: []byte(`{"type":"redacted_thinking","data":"blob"}`),
		}}}},
	}
	if start := thinkingRunStart(messages, []string{reasoning.ThinkingRunAnchorDigest(protocolcodec.RedactedAnchorKey("blob"))}); start != 1 {
		t.Fatalf("run start = %d, want 1", start)
	}
}

// Opaque blocks a turn records never push the live run's anchor out.
func TestOpaqueBlocksNeverEvictTheRunAnchor(t *testing.T) {
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	episode := raylinearc.HashEpisodeID(t.Name())
	worker := reasoning.ThinkingRunWorker("opus")
	seed := func(state *raylinearc.EpisodeState) {
		state.ReasoningProvenance = reasoning.WithThinkingRun(nil, worker, nil, false, protocolcodec.SignatureAnchorKey("sig-anchor"))
	}
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	seed(state)
	transaction := newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil)
	transaction.markSelection(0, 10)
	transaction.thinkingWorker, transaction.thinkingAnchored, transaction.dispatchWorker = worker, true, "claude"
	output := []llmprotocol.Content{}
	for index := range reasoning.MaxReasoningProvenance + 4 {
		output = append(output, llmprotocol.Content{Kind: llmprotocol.ContentUnmodeled, Unmodeled: &llmprotocol.UnmodeledBlock{
			Format: llmprotocol.AnthropicMessagesV1, Type: "redacted_thinking",
			Raw: []byte(fmt.Sprintf(`{"type":"redacted_thinking","data":"blob-%d"}`, index)),
		}})
	}
	response := &llmprotocol.Response{Output: []llmprotocol.OutputItem{{Role: llmprotocol.RoleAssistant, Content: output}}}
	if err := transaction.commit(context.Background(), &RequestContext{SemanticResponse: response}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	next, reread, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	_ = store.Abort(context.Background(), next)
	if len(reread.ReasoningProvenance) > reasoning.MaxReasoningProvenance ||
		fmt.Sprint(reasoning.ThinkingRunAnchors(reread.ReasoningProvenance, worker)) != fmt.Sprint([]string{reasoning.ThinkingRunAnchorDigest(protocolcodec.SignatureAnchorKey("sig-anchor"))}) {
		t.Fatalf("provenance = %d entries, anchors %v", len(reread.ReasoningProvenance), reasoning.ThinkingRunAnchors(reread.ReasoningProvenance, worker))
	}
}

// claudeChatReasoning is a Chat client's replay of Claude's reasoning as
// OpenRouter returns it: signed text and a redacted blob.
const claudeChatReasoning = `"reasoning_content":"claude reasoning",` +
	`"reasoning_details":[{"type":"reasoning.text","text":"claude reasoning","signature":"claude-signature","format":"anthropic-claude-v1","index":0},` +
	`{"type":"reasoning.encrypted","data":"claude-redacted-blob","format":"anthropic-claude-v1","index":1}]`

// gptChatReasoning is a Chat client's replay of an OpenAI model's reasoning:
// its encrypted reasoning, readable only by OpenAI.
const gptChatReasoning = `"reasoning_content":"gpt reasoning",` +
	`"reasoning_details":[{"type":"reasoning.encrypted","data":"gAAAA-gpt-blob","format":"openai-responses-v1","index":0}]`

// A switch from Claude to a model that is not Claude: the history's Claude
// signatures, redacted blocks and signed reasoning_details never reach it,
// over Chat (OpenRouter refuses them, "encrypted reasoning ... can only be
// processed by the provider that created it") or from a Messages client; the
// reasoning's text does, as reasoning_content.
func TestClaudeToOtherModelSwitchDropsClaudesEncryptedReasoning(t *testing.T) {
	router, _ := newThinkingGuardRouter(t, "think")
	t.Run("a Chat client's reasoning_details", func(t *testing.T) {
		_, body := dispatchThinkingTurnAt(t, router, "episode-switch-claude-gpt-chat", "/v1/chat/completions",
			piChatHistory(claudeChatReasoning))
		assertJSONField(t, body, "model", `"vendor/think"`)
		wire := string(body["messages"])
		if strings.Contains(wire, "reasoning_details") || strings.Contains(wire, "claude-signature") ||
			strings.Contains(wire, "claude-redacted-blob") || !strings.Contains(wire, `"reasoning_content":"claude reasoning"`) {
			t.Fatalf("Claude's encrypted reasoning reached another model, or its text did not: %s", wire)
		}
	})
	t.Run("a Messages client's thinking", func(t *testing.T) {
		_, body := dispatchThinkingTurn(t, router, "episode-switch-claude-gpt-messages", messagesHistory(
			`{"role":"user","content":"start"}`,
			assistantTurn(thinkingBlock("sig-claude"), `{"type":"redacted_thinking","data":"claude-redacted-blob"}`,
				`{"type":"text","text":"turn one"}`),
			`{"role":"user","content":"go on"}`,
		))
		wire := string(body["messages"])
		if strings.Contains(wire, `"sig-claude"`) || strings.Contains(wire, "signature") ||
			strings.Contains(wire, "claude-redacted-blob") || !strings.Contains(wire, "turn one") {
			t.Fatalf("Claude's signed or redacted thinking reached another model: %s", wire)
		}
	})
}

// A switch from another model to Claude: that model's encrypted reasoning
// never reaches Claude.
func TestOtherModelToClaudeSwitchDropsItsEncryptedReasoning(t *testing.T) {
	router, _ := newThinkingGuardRouter(t, "claude")
	_, body := dispatchThinkingTurnAt(t, router, "episode-switch-gpt-claude", "/v1/chat/completions",
		piChatHistory(gptChatReasoning))
	assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
	wire := string(body["messages"])
	if strings.Contains(wire, "gAAAA-gpt-blob") || strings.Contains(wire, "gpt reasoning") {
		t.Fatalf("another model's reasoning reached Claude: %s", wire)
	}
	if !strings.Contains(wire, "call_1") {
		t.Fatalf("the tool call was lost: %s", wire)
	}
}

// The model's own run keeps its encrypted reasoning, the active tool loop's
// included; without the run, the loop's encrypted reasoning is stripped too
// for a worker that is not Claude.
func TestSameModelRunKeepsItsEncryptedReasoning(t *testing.T) {
	router, _ := newThinkingGuardRouter(t, "think")
	worker := router.thinkingRunWorker(armDispatch(t, router, "think").backendName, armDispatch(t, router, "think").upstreamModel)
	history := `{"model":"auto","max_completion_tokens":32000,"messages":[` +
		`{"role":"user","content":"Create fib.py and run it."},` +
		`{"role":"assistant","content":null,` + gptChatReasoning + `,` +
		`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"call_1","content":"ok"}],` +
		`"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{}}}}]}`
	for _, tc := range []struct {
		name string
		seed bool
	}{{"served", true}, {"unserved", false}} {
		t.Run(tc.name, func(t *testing.T) {
			episode := "episode-same-model-" + tc.name
			if tc.seed {
				store := router.RaylineARCEpisodeStore
				lease, state, err := store.Prepare(context.Background(), raylinearc.HashEpisodeID(episode), 4)
				if err != nil {
					t.Fatal(err)
				}
				state.ReasoningProvenance = reasoning.WithThinkingRun(nil, worker, nil, false, protocolcodec.ToolCallAnchorKey("call_1"))
				if err := state.Commit(0, 10, time.Now().UTC()); err != nil {
					t.Fatal(err)
				}
				if err := store.Commit(context.Background(), lease, lease.Version(), state); err != nil {
					t.Fatal(err)
				}
			}
			ctx, body := dispatchThinkingTurnAt(t, router, episode, "/v1/chat/completions", history)
			wire := string(body["messages"])
			if kept := strings.Contains(wire, "gAAAA-gpt-blob"); kept != tc.seed {
				t.Fatalf("encrypted reasoning kept = %v, want %v: %s", kept, tc.seed, wire)
			}
			if !strings.Contains(wire, `"reasoning_content":"gpt reasoning"`) {
				t.Fatalf("the reasoning text was lost: %s", wire)
			}
			if ctx.ThinkingSignatureRetry == nil {
				t.Fatal("no re-send was staged for a worker that is not Claude")
			}
		})
	}
}

// The re-send is not Claude's alone: a worker that is not Claude refused
// with OpenRouter's encrypted-reasoning 400 is re-sent once without any
// reasoning_details.
func TestEncryptedReasoningRefusalIsResentForAnyWorker(t *testing.T) {
	router, _ := newThinkingGuardRouter(t, "think")
	dispatch := armDispatch(t, router, "think")
	worker := router.thinkingRunWorker(dispatch.backendName, dispatch.upstreamModel)
	episode := "episode-gpt-retry"
	store := router.RaylineARCEpisodeStore
	lease, state, err := store.Prepare(context.Background(), raylinearc.HashEpisodeID(episode), 4)
	if err != nil {
		t.Fatal(err)
	}
	state.ReasoningProvenance = reasoning.WithThinkingRun(nil, worker, nil, false, protocolcodec.ToolCallAnchorKey("call_1"))
	if err := state.Commit(0, 10, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	if err := store.Commit(context.Background(), lease, lease.Version(), state); err != nil {
		t.Fatal(err)
	}
	ctx, body := dispatchThinkingTurnAt(t, router, episode, "/v1/chat/completions", piChatHistory(gptChatReasoning))
	if !strings.Contains(string(body["messages"]), "gAAAA-gpt-blob") {
		t.Fatalf("the run's own reasoning was not sent: %s", body["messages"])
	}
	upstream := newClaudeUpstream(t, http.StatusOK, flexTwinChatReply)
	refusal := `{"error":{"message":"Your request contains encrypted reasoning or compaction content that can only be processed by the provider that created it (anthropic) or by azure, azure/eu, azure/us","code":400}}`
	envoy := answerRefusal(t, router, ctx, upstream, http.StatusBadRequest, refusal)
	if envoy.Responses[0].GetImmediateResponse() == nil {
		t.Fatalf("the refusal was not re-sent: %v", envoy.Responses[0])
	}
	calls := upstream.calls()
	if len(calls) != 1 || strings.Contains(string(calls[0]), "reasoning_details") ||
		!strings.Contains(string(calls[0]), "gpt reasoning") {
		t.Fatalf("re-sends = %d: %s", len(calls), calls)
	}
}

func armDispatch(t *testing.T, router *OpenAIRouter, arm string) *providerDispatch {
	t.Helper()
	dispatch, err := router.resolveProviderDispatch(arm, "", false, llmprotocol.OpenAIChatV1)
	if err != nil {
		t.Fatal(err)
	}
	return dispatch
}

// Reasoning a Chat client holds only inside reasoning_details keeps its text
// when the details are stripped for a worker that is not Claude.
func TestStrippedReasoningDetailsKeepTheirText(t *testing.T) {
	router, _ := newThinkingGuardRouter(t, "think")
	_, body := dispatchThinkingTurnAt(t, router, "episode-details-text-only", "/v1/chat/completions", piChatHistory(
		`"reasoning_details":[{"type":"reasoning.text","text":"claude reasoning","signature":"claude-signature","format":"anthropic-claude-v1","index":0}]`))
	wire := string(body["messages"])
	if strings.Contains(wire, "reasoning_details") || !strings.Contains(wire, `"reasoning_content":"claude reasoning"`) {
		t.Fatalf("the details' text was lost or the details sent: %s", wire)
	}
}

// A reply's text repeated from another model's earlier turn does not anchor
// the run there: a text anchor names its place in the history.
func TestRepeatedTextDoesNotMoveTheRun(t *testing.T) {
	done := func() llmprotocol.Message {
		return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Done."}}}
	}
	user := llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "next"}}}
	messages := []llmprotocol.Message{user, done(), user, done(), user}
	reply := &llmprotocol.Response{Output: []llmprotocol.OutputItem{{Role: llmprotocol.RoleAssistant, Content: done().Content}}}
	anchor := reasoning.ThinkingRunAnchorDigest(protocolcodec.ResponseAnchorKey(reply, 3))
	if start := thinkingRunStart(messages, []string{anchor}); start != 3 {
		t.Fatalf("run start = %d, want the reply's own place, 3", start)
	}
}

// A Chat re-send strips the loop's own reasoning_details too: a refusal's
// index cannot show they are not the refused ones.
func TestChatRetryStripsTheLoopsDetails(t *testing.T) {
	dispatched := []byte(piChatHistory(claudeChatReasoning))
	retried, change, ok := chatRetryBody(dispatched, false)
	if !ok || change.stripped != 1 || strings.Contains(string(retried), "reasoning_details") ||
		!strings.Contains(string(retried), "claude reasoning") {
		t.Fatalf("ok=%v change=%+v body=%s", ok, change, retried)
	}
	// The loop here is no longer active (a user turn follows it), so a
	// Claude re-send keeps its reasoning request.
	var withReasoning map[string]json.RawMessage
	_ = json.Unmarshal([]byte(piChatHistory(claudeChatReasoning)), &withReasoning)
	withReasoning["reasoning"] = json.RawMessage(`{"max_tokens":2048}`)
	var messages []json.RawMessage
	_ = json.Unmarshal(withReasoning["messages"], &messages)
	withReasoning["messages"], _ = json.Marshal(messages[:len(messages)-1])
	active, _ := json.Marshal(withReasoning)
	retried, change, ok = chatRetryBody(active, true)
	if !ok || !change.thinkingDisabled || strings.Contains(string(retried), `"reasoning":{`) {
		t.Fatalf("an active loop's Claude re-send still asked for reasoning: ok=%v change=%+v body=%s", ok, change, retried)
	}
}

// A Responses client resends one reply as several items; the run is found at
// the reply's first one.
func TestRunIsFoundAcrossASplitReply(t *testing.T) {
	user := llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "go"}}}
	reasoningItem := llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentReasoning, Text: "thought"}}}
	textItem := llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "answer"}}}
	reply := &llmprotocol.Response{Output: []llmprotocol.OutputItem{
		{Role: llmprotocol.RoleAssistant, Content: reasoningItem.Content},
		{Role: llmprotocol.RoleAssistant, Content: textItem.Content},
	}}
	anchor := reasoning.ThinkingRunAnchorDigest(protocolcodec.ResponseAnchorKey(reply, 1))
	if start := thinkingRunStart([]llmprotocol.Message{user, reasoningItem, textItem, user}, []string{anchor}); start != 1 {
		t.Fatalf("run start = %d, want the reply's first item, 1", start)
	}
}

// A Chat re-send keeps the plaintext of reasoning held only in its details.
func TestChatRetryKeepsDetailOnlyText(t *testing.T) {
	retried, _, ok := chatRetryBody([]byte(piChatHistory(
		`"reasoning_details":[{"type":"reasoning.text","text":"only here","signature":"sig","format":"anthropic-claude-v1","index":0}]`)), false)
	if !ok || strings.Contains(string(retried), "reasoning_details") || !strings.Contains(string(retried), `"reasoning_content":"only here"`) {
		t.Fatalf("ok=%v body=%s", ok, retried)
	}
}

// End to end through the Responses codec: a Responses client resends the
// loop's latest reply as separate reasoning, function_call and
// function_call_output items. Claude, with no run on record, is still sent
// the reply's signed reasoning beside its call, which Anthropic requires,
// while an earlier turn's reasoning is stripped.
func TestSplitActiveLoopReplyKeepsItsReasoningForClaude(t *testing.T) {
	router, _ := newThinkingGuardRouter(t, "claude")
	history := `{"model":"auto","store":false,"input":[` +
		`{"role":"user","content":"start"},` +
		`{"type":"reasoning","id":"item_0","summary":[],"content":[{"type":"reasoning_text","text":"old reasoning"}],` +
		`"signature":"sig-old","format":"anthropic-claude-v1"},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"turn one"}]},` +
		`{"role":"user","content":"run the tests"},` +
		`{"type":"reasoning","id":"item_1","summary":[],"content":[{"type":"reasoning_text","text":"loop reasoning"}],` +
		`"signature":"sig-loop","format":"anthropic-claude-v1"},` +
		`{"type":"function_call","id":"fc_1","call_id":"call_1","name":"bash","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"call_1","output":"ok"}],` +
		`"tools":[{"type":"function","name":"bash","parameters":{"type":"object","properties":{}}}]}`
	ctx, body := dispatchThinkingTurnAt(t, router, "episode-split-loop", "/v1/responses", history)
	assertJSONField(t, body, "model", `"anthropic/claude-opus-5"`)
	wire := string(body["messages"])
	if !strings.Contains(wire, `"signature":"sig-loop"`) || strings.Contains(wire, "sig-old") {
		t.Fatalf("the loop's reasoning was stripped or the earlier turn's sent: %s", wire)
	}
	var messages []struct {
		Role    string           `json:"role"`
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatal(err)
	}
	for _, message := range messages {
		if message.Role != "assistant" || len(message.Content) == 0 {
			continue
		}
		for _, block := range message.Content {
			if block["type"] == "tool_use" && message.Content[0]["type"] != "thinking" {
				t.Fatalf("the loop's assistant message does not start with its thinking: %s", wire)
			}
		}
	}
	if adopted := ctx.RaylineARCTransaction.thinkingAdopted; adopted != protocolcodec.SignatureAnchorKey("sig-loop") {
		t.Fatalf("adopted = %q, want the loop reply's signature", adopted)
	}
}

// The active loop is found whatever order parallel results come back in, and
// with text beside the results.
func TestActiveLoopIsFoundAcrossParallelResultsAndTextBesideThem(t *testing.T) {
	router, _ := newThinkingGuardRouter(t, "claude")
	t.Run("reverse-ordered parallel results", func(t *testing.T) {
		history := `{"model":"auto","store":false,"input":[` +
			`{"role":"user","content":"run both"},` +
			`{"type":"reasoning","id":"item_1","summary":[],"content":[{"type":"reasoning_text","text":"loop reasoning"}],` +
			`"signature":"sig-loop","format":"anthropic-claude-v1"},` +
			`{"type":"function_call","id":"fc_a","call_id":"call_a","name":"bash","arguments":"{}"},` +
			`{"type":"function_call","id":"fc_b","call_id":"call_b","name":"bash","arguments":"{}"},` +
			`{"type":"function_call_output","call_id":"call_b","output":"b"},` +
			`{"type":"function_call_output","call_id":"call_a","output":"a"}],` +
			`"tools":[{"type":"function","name":"bash","parameters":{"type":"object","properties":{}}}]}`
		_, body := dispatchThinkingTurnAt(t, router, "episode-parallel-reverse", "/v1/responses", history)
		if !strings.Contains(string(body["messages"]), `"signature":"sig-loop"`) {
			t.Fatalf("the loop's reasoning was stripped: %s", body["messages"])
		}
	})
	t.Run("text beside the tool result", func(t *testing.T) {
		history := messagesHistory(
			`{"role":"user","content":"start"}`,
			assistantTurn(thinkingBlock("sig-earlier"), `{"type":"text","text":"turn one"}`),
			`{"role":"user","content":"run the tests"}`,
			assistantTurn(thinkingBlock("sig-loop"), `{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}`),
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"},{"type":"text","text":"and then?"}]}`,
		)
		_, body := dispatchThinkingTurn(t, router, "episode-text-beside-result", history)
		assertSignatures(t, "text beside result", body["messages"], []string{"sig-loop"}, []string{"sig-earlier"})
	})
}

// The Messages re-send strips a split active reply's thinking, in every one
// of its messages, and turns thinking off when the model can.
func TestThinkingRetryBodyTreatsASplitReplyAsOne(t *testing.T) {
	body := []byte(`{"thinking":{"type":"adaptive"},"messages":[` +
		`{"role":"user","content":"start"},` +
		assistantTurn(thinkingBlock("sig-earlier"), `{"type":"text","text":"turn one"}`) + `,` +
		`{"role":"user","content":"run the tests"},` +
		assistantTurn(thinkingBlock("sig-loop")) + `,` +
		assistantTurn(`{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}`) + `,` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}]}`)
	retried, change, ok := thinkingRetryBody(body, true)
	if !ok || !change.thinkingDisabled || change.stripped != 2 || strings.Contains(string(retried), `"sig-loop"`) {
		t.Fatalf("ok=%v change=%+v body=%s", ok, change, retried)
	}
}
