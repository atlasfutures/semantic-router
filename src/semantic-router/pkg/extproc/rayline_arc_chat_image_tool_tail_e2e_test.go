//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// semantic-router #238, ADR 0129, the shape the issue was found on: a
// Messages client (Hermes) whose tool returned an image, routed to a Chat
// worker. Pathfinder's chat image_tool_tail golden, sent as the Anthropic
// request it translates from, must reach the provider as the golden's
// messages: the tool message carries text and the image follows in a
// labelled user message. The corpus cell states no task-fidelity evidence,
// so the steer is refused (pathfinder#3996): call 1 goes out unsteered, and
// its routing record names the refusal and the level the provider sees.
func TestMessagesImageToolTailReachesAChatWorkerAsTheGolden(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	logs := captureLogs(t)
	dir := filepath.Join(responsesGoldenDir, "..", "chat", "image_tool_tail")
	var c responsesGoldenCase
	raw, err := os.ReadFile(filepath.Join(dir, "case.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	path, packageSHA, actionFor := responsesGoldenPolicyConfig(t, c, "{{POLICY_URL}}")
	var catalog []string
	for _, action := range actionFor {
		catalog = append(catalog, action)
	}
	sort.Strings(catalog)
	fake := newFakePolicyService(t, responsesGoldenAlias, packageSHA, catalog)
	config, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	rendered := strings.ReplaceAll(string(config), "{{POLICY_URL}}", fake.URL())
	if err = os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)

	const image = `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="}}`
	clients := []string{
		`{"model":"auto","max_tokens":32000,"stream":true,"system":"You are an agent.","messages":[` +
			`{"role":"user","content":"Read the screenshot."}]}`,
		`{"model":"auto","max_tokens":32000,"stream":true,"system":"You are an agent.","messages":[` +
			`{"role":"user","content":"Read the screenshot."},` +
			`{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"Bash","input":{"command":"cat shot.png"}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"text","text":"shot.png"},` + image + `]}]}]}`,
	}
	for index, call := range c.Calls {
		want, err := os.ReadFile(filepath.Join(dir, call.ExpectedBody))
		if err != nil {
			t.Fatal(err)
		}
		action := actionFor[*call.ControlID]
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
		got := dispatchMessagesLeverTurn(t, router, fmt.Sprintf("chat-image-tail-%d", index), "golden-chat-image-tail", clients[index])
		gotMessages, wantMessages := imageTailChatMessages(t, got), imageTailChatMessages(t, string(want))
		if imageTailRoles(gotMessages) != imageTailRoles(wantMessages) {
			t.Fatalf("call %d provider roles = %s, want %s (the steer must not be a user message of its own)",
				index, imageTailRoles(gotMessages), imageTailRoles(wantMessages))
		}
		// What ADR 0129 governs is the tail: the user message that carries
		// the hoisted image (and an evidenced cell's steer). The messages before it
		// differ from pathfinder's translator only in spellings the Chat
		// codec already used (an assistant tool call without content: "",
		// a text-only tool message as a string, with its tool name).
		last := len(gotMessages) - 1
		if gotTail, wantTail := marshalCanonical(gotMessages[last]), marshalCanonical(wantMessages[last]); gotTail != wantTail {
			t.Fatalf("call %d provider tail message =\n%s\nwant\n%s", index, gotTail, wantTail)
		}
	}
	var record map[string]interface{}
	for _, entry := range logs.All() {
		fields := entry.ContextMap()
		if fields["event"] == "routing_decision" && fields["request_id"] == "chat-image-tail-1" {
			record = fields
		}
	}
	if record == nil {
		t.Fatal("call 1 logged no routing decision")
	}
	for key, want := range map[string]interface{}{
		"thinking_refused": "image_tool_tail_task_fidelity", "thinking_level_requested": "up",
		"thinking_level_in_force": "none", "thinking_emitted": false,
		// Attributed to the control the provider sees (call 0's), not the
		// drawn one it never received.
		"thinking_control_sha256": *c.Calls[0].ControlID,
	} {
		if record[key] != want {
			t.Errorf("call 1 routing record %s = %#v, want %#v", key, record[key], want)
		}
	}
}

// imageTailChatMessages is a Chat body's messages.
func imageTailChatMessages(t *testing.T, body string) []map[string]any {
	t.Helper()
	var wire struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal([]byte(body), &wire); err != nil {
		t.Fatalf("not a Chat body: %v\n%s", err, body)
	}
	return wire.Messages
}

func imageTailRoles(messages []map[string]any) string {
	names := make([]string, len(messages))
	for index, message := range messages {
		names[index], _ = message["role"].(string)
	}
	return strings.Join(names, ",")
}

// marshalCanonical re-marshals a value so key order and spacing do not count.
func marshalCanonical(value any) string {
	out, _ := json.Marshal(value)
	return string(out)
}

// dispatchMessagesLeverTurn sends one Messages client body of an episode
// through the request phases, commits it as a 200, and returns the
// provider-bound bytes.
func dispatchMessagesLeverTurn(t *testing.T, router *OpenAIRouter, requestID, episode, body string) string {
	t.Helper()
	ctx := &RequestContext{Headers: map[string]string{}, RequestID: requestID, StartTime: time.Now(), TraceContext: context.Background()}
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
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		t.Fatalf("request refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	mutation := response.GetRequestBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("no provider body was written")
	}
	if _, err := router.handleResponseHeaders(arcResponseHeaders("200"), ctx); err != nil {
		t.Fatalf("commit: %v", err)
	}
	completeTestResponse(t, ctx)
	finalizeSelectionProcessTerminal(ctx)
	return string(mutation.GetBody())
}
