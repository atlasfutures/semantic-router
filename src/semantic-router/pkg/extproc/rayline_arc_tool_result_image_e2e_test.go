//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// An image inside an Anthropic tool result, end to end: the client's Messages
// request goes through the real ext_proc request path to a basket where only
// one arm takes image input and declares tool_result_images, and that arm is
// a Chat backend. The capability gate must offer that arm alone, and the
// provider-bound Chat body must carry the image in a user message after the
// tool messages, since a Chat tool message holds only text.
func TestToolResultImageReachesTheChatVisionArm(t *testing.T) {
	router, fake, actions := toolResultImageRouter(t, true)
	var offered []string
	fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		offered = append([]string(nil), request.Selection.AvailableActionIDs...)
		if len(offered) == 0 {
			return ""
		}
		return offered[0]
	})
	body := dispatchPolicyClientRequest(t, router, "tool-result-image", "/v1/messages", toolResultImageClient)

	if want := []string{actions["think-up"].ActionID}; !reflect.DeepEqual(offered, want) {
		t.Fatalf("offered actions = %v, want only the vision arm's %v", offered, want)
	}
	assertJSONField(t, body, "model", `"vendor/think"`)
	var messages []struct {
		Role       string          `json:"role"`
		ToolCallID string          `json:"tool_call_id"`
		Content    json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatalf("messages = %s: %v", body["messages"], err)
	}
	toolAt := -1
	for index, message := range messages {
		if message.Role == "tool" && message.ToolCallID == "toolu_view_red" {
			toolAt = index
		}
	}
	if toolAt < 0 || toolAt+1 >= len(messages) {
		t.Fatalf("no user message follows the tool message: %s", body["messages"])
	}
	if strings.Contains(string(messages[toolAt].Content), toolResultImageRed) {
		t.Fatalf("the tool message carries the image, which Chat cannot: %s", messages[toolAt].Content)
	}
	hoisted := messages[toolAt+1]
	var parts []struct {
		Type     string `json:"type"`
		ImageURL struct {
			URL string `json:"url"`
		} `json:"image_url"`
	}
	if err := json.Unmarshal(hoisted.Content, &parts); err != nil {
		t.Fatalf("the message after the tool message = %s: %v", hoisted.Content, err)
	}
	var images []string
	for _, part := range parts {
		if part.Type == "image_url" {
			images = append(images, part.ImageURL.URL)
		}
	}
	if want := []string{"data:image/png;base64," + toolResultImageRed}; hoisted.Role != "user" || !reflect.DeepEqual(images, want) {
		t.Fatalf("message after the tool message = %s %s, want a user message with %v", hoisted.Role, hoisted.Content, want)
	}
}

// Control: the same turn where no card declares tool_result_images is
// refused as no_capable_arm, the replayable 503, before any decide call.
func TestToolResultImageWithoutTheClaimIsNoCapableArm(t *testing.T) {
	router, fake, _ := toolResultImageRouter(t, false)
	logs := captureLogs(t)
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "policy-e2e-tool-result-image-unclaimed",
		StartTime: time.Now(), TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/messages"},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: "tool-result-image-unclaimed"},
		}},
	}}
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(toolResultImageClient), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil || immediate.GetStatus().GetCode() != typev3.StatusCode_ServiceUnavailable {
		t.Fatalf("immediate = %v, want the replayable 503", immediate)
	}
	// The public answer is the bounded "unavailable"; the selection log keeps
	// the internal class that says why.
	named := false
	for _, entry := range logs.All() {
		raw, _ := json.Marshal(entry.ContextMap())
		if strings.Contains(entry.Message+string(raw), arcFailureNoCapableArm) {
			named = true
		}
	}
	if !named {
		t.Fatalf("no log names %s", arcFailureNoCapableArm)
	}
	if calls := len(fake.received()); calls != 0 {
		t.Fatalf("%d decide calls, want none before the refusal", calls)
	}
}

const toolResultImageRed = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"

const toolResultImageClient = `{"model":"auto","max_tokens":32000,"messages":[` +
	`{"role":"user","content":"What is in red.png? Use view_image."},` +
	`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_view_red","name":"view_image","input":{"image":"red.png"}}]},` +
	`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_view_red","content":[` +
	`{"type":"text","text":"Loaded 1 image."},` +
	`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + toolResultImageRed + `"}}]}]}],` +
	`"tools":[{"name":"view_image","description":"Load an image.","input_schema":{"type":"object","properties":{"image":{"type":"string"}}}}]}`

// toolResultImageRouter builds the policy dispatch cell with only think's
// model card, a Chat backend, taking image input; claimed says whether that
// card also declares tool_result_images.
func toolResultImageRouter(t *testing.T, claimed bool) (*OpenAIRouter, *fakePolicyService, map[string]config.RaylineARCPolicyBinding) {
	t.Helper()
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := map[string]config.RaylineARCPolicyBinding{
		"think-up":      policyAction("think", "up", "think-trained", policyTestEffort("high"), nil, policyTestUp),
		"off":           policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		"claude":        policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"claude-off-up": policyAction("claude-off", "up", "claude-opus-5", policyTestEffort("none"), nil, policyTestUp),
	}
	catalog := make([]string, 0, len(actions))
	for _, declared := range actions {
		catalog = append(catalog, declared.ActionID)
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	path := writePolicyDispatchConfig(t, fake.URL(), actions)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	rendered := string(raw)
	rewrites := map[string]string{
		"    - name: \"off\"\n      modality: text\n":    "      vision: false\n",
		"    - name: claude\n      modality: text\n":     "      vision: false\n",
		"    - name: claude-off\n      modality: text\n": "      vision: false\n",
	}
	if claimed {
		rewrites["    - name: think\n      modality: text\n"] = "      capabilities: [tool_result_images]\n"
	}
	for card, added := range rewrites {
		if strings.Count(rendered, card) != 1 {
			t.Fatalf("the config template changed; update the card rewrite:\n%s", card)
		}
		rendered = strings.Replace(rendered, card, card+added, 1)
	}
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	return router, fake, actions
}
