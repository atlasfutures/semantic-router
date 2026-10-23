//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/tidwall/gjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The fixtures are Claude Code 2.1.309 bodies captured on 2026-10-23 with every
// free-text value replaced by a placeholder and every id randomized; their
// structure and protocol members are as sent, but for the model, which names
// this test config's entry point.
//   - thread-continue: the message-threads continuation sent after a tool
//     result (prod rt_a9845555-41e): a tool_result whose call is in the
//     previous response, then the mid-conversation system message.
//   - thread-continue.refusal: the 400 the cell answered that body with before
//     2026-10-23 and that Claude Code answers with a full-history re-send.
//   - full-history: the 8-message turn that re-send carries (tools omitted).
const threadContinuationFixtures = "testdata/thread_continuation"

// claudeCodeBetas is the anthropic-beta Claude Code 2.1.309 sent with the
// thread continuation.
const claudeCodeBetas = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14," +
	"thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05," +
	"context-1m-2025-08-07,mid-conversation-tool-changes-2026-07-01,inline-tools-2026-09-15," +
	"advisor-tool-2026-03-01,advanced-tool-use-2025-11-20,mid-conversation-system-clear-at-2026-08-21," +
	"effort-2025-11-24,thinking-binding-controls-2026-08-01,extended-cache-ttl-2025-04-11," +
	"cache-diagnosis-2026-04-07,message-threads-2026-08-12"

func readThreadFixture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(threadContinuationFixtures, name))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func newThreadContinuationRouter(t *testing.T) (*OpenAIRouter, *fakePolicyService) {
	t.Helper()
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := map[string]config.RaylineARCPolicyBinding{
		"think":      policyAction("think", "up", "think-trained", policyTestEffort("high"), nil, policyTestUp),
		"claude":     policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"off":        policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"claude-off": policyAction("claude-off", "up", "claude-opus-5", policyTestEffort("none"), nil, policyTestUp),
	}
	catalog := make([]string, 0, len(actions))
	for _, action := range actions {
		catalog = append(catalog, action.ActionID)
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions["claude"].ActionID })
	router, err := NewOpenAIRouter(writePolicyDispatchConfig(t, fake.URL(), actions))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	return router, fake
}

// runClaudeCodeRequest runs one Claude Code Messages body through the request
// phases with the gateway's session header.
func runClaudeCodeRequest(t *testing.T, router *OpenAIRouter, body []byte) *ext_proc.ProcessingResponse {
	t.Helper()
	ctx := &RequestContext{
		Headers:      map[string]string{},
		RequestID:    "thread-continuation",
		StartTime:    time.Now(),
		TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/messages?beta=true"},
			{Key: "content-type", Value: "application/json"},
			{Key: "anthropic-version", Value: "2023-06-01"},
			{Key: "anthropic-beta", Value: claudeCodeBetas},
			{Key: "x-rayline-session", Value: "thread-continuation-episode"},
		}},
	}}
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: body, EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	// What the stream loop does before Envoy sends a refusal to the client.
	return router.encodeImmediateResponseForClient(response, ctx)
}

// A thread continuation is refused, before any routing, with byte for byte
// the 400 Claude Code answers with a full-history re-send. Served instead, the
// model sees a lone tool result and the user a reply with no history.
func TestThreadContinuationGetsTheRefusalClaudeCodeResendsOn(t *testing.T) {
	router, fake := newThreadContinuationRouter(t)
	response := runClaudeCodeRequest(t, router, readThreadFixture(t, "claude-code-2.1.309-thread-continue.request.json"))
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("a thread continuation was routed (%d policy decision(s)); provider body:\n%s",
			len(fake.received()), response.GetRequestBody().GetResponse().GetBodyMutation().GetBody())
	}
	if code := int(immediate.GetStatus().GetCode()); code != 400 {
		t.Fatalf("status = %d, want 400", code)
	}
	want := readThreadFixture(t, "claude-code-2.1.309-thread-continue.refusal.json")
	if string(immediate.GetBody()) != string(want) {
		t.Fatalf("refusal body =\n%s\nwant byte for byte\n%s", immediate.GetBody(), want)
	}
	if n := len(fake.received()); n != 0 {
		t.Fatalf("the policy service was asked %d time(s) about a thread continuation", n)
	}
}

// A thread continuation the codec would accept on its own (a text turn, no
// tool result) is refused all the same: what decides is the thread member,
// not whether the fragment happens to decode.
func TestThreadContinuationThatDecodesIsStillRefused(t *testing.T) {
	router, fake := newThreadContinuationRouter(t)
	body := []byte(`{"model":"auto","max_tokens":1024,"stream":true,` +
		`"thread":{"type":"continue","previous_message_id":"msg_Q2r9XyK1bWm4Zt7Lp0VaHc3E"},` +
		`"messages":[{"role":"user","content":"<text 1>"}]}`)
	immediate := runClaudeCodeRequest(t, router, body).GetImmediateResponse()
	if immediate == nil || immediate.GetStatus().GetCode() != 400 {
		t.Fatalf("a decodable thread continuation was not refused 400: %v", immediate)
	}
	const want = `{"type":"error","error":{"type":"invalid_request_error","message":` +
		`"tool result must follow its request tool call or carry a retained-history link: field \"thread\""}}`
	if string(immediate.GetBody()) != want {
		t.Fatalf("refusal body =\n%s\nwant\n%s", immediate.GetBody(), want)
	}
	if n := len(fake.received()); n != 0 {
		t.Fatalf("the policy service was asked %d time(s) about a thread continuation", n)
	}
}

// The full-history turn Claude Code re-sends after the refusal is served with
// every message it carries, in order, its tool call and result still linked.
func TestFullHistoryTurnReachesTheProviderWhole(t *testing.T) {
	router, _ := newThreadContinuationRouter(t)
	client := readThreadFixture(t, "claude-code-2.1.309-full-history.request.json")
	response := runClaudeCodeRequest(t, router, client)
	if immediate := response.GetImmediateResponse(); immediate != nil {
		t.Fatalf("the full-history turn was refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	mutation := response.GetRequestBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("no provider body was written")
	}

	type message struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	var sent, served struct {
		Messages []message `json:"messages"`
	}
	if err := json.Unmarshal(client, &sent); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(mutation.GetBody(), &served); err != nil {
		t.Fatalf("provider body: %v\n%s", err, mutation.GetBody())
	}
	if len(served.Messages) != len(sent.Messages) {
		t.Fatalf("provider body carries %d messages, the client sent %d:\n%s",
			len(served.Messages), len(sent.Messages), mutation.GetBody())
	}
	for i := range sent.Messages {
		if served.Messages[i].Role != sent.Messages[i].Role {
			t.Fatalf("message %d role = %q, the client sent %q", i, served.Messages[i].Role, sent.Messages[i].Role)
		}
	}
	// Every text the client sent reaches the provider: no message's content
	// was dropped on the way.
	servedText := jsonStrings(t, mutation.GetBody())
	for text := range jsonStrings(t, client) {
		if strings.HasPrefix(text, "<") && !servedText[text] {
			t.Fatalf("the provider body lost %q", text)
		}
	}
	lastCall := gjson.GetBytes(served.Messages[len(served.Messages)-2].Content, "0.id").String()
	lastResult := gjson.GetBytes(served.Messages[len(served.Messages)-1].Content, "0.tool_use_id").String()
	if lastCall == "" || lastResult != lastCall {
		t.Fatalf("the tool result is no longer linked to its call: %q -> %q", lastResult, lastCall)
	}
}

// jsonStrings is every string value in a JSON document, as a set.
func jsonStrings(t *testing.T, document []byte) map[string]bool {
	t.Helper()
	var root any
	if err := json.Unmarshal(document, &root); err != nil {
		t.Fatal(err)
	}
	found := map[string]bool{}
	var walk func(any)
	walk = func(value any) {
		switch typed := value.(type) {
		case string:
			found[typed] = true
		case []any:
			for _, item := range typed {
				walk(item)
			}
		case map[string]any:
			for _, item := range typed {
				walk(item)
			}
		}
	}
	walk(root)
	return found
}
