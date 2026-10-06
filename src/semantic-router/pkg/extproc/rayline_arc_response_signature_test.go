//go:build !windows && cgo

package extproc

import (
	"context"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const claudeSignedThinkingAnswer = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5",` +
	`"content":[{"type":"thinking","thinking":"Check the tool.","signature":"EqQBsigclaude"},{"type":"text","text":"Done."}],` +
	`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":5}}`

const claudeSignedThinkingStream = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Check the tool.\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"EqQBsigclaude\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Done.\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

// End to end (atlasfutures/semantic-router#212): a Chat client routed to a
// Claude arm served only over Messages gets Claude's signed thinking as
// reasoning_content, without the signature, and a 200 -- buffered and
// streamed -- instead of a 502 for a capability Chat does not have.
func TestChatClientOnAMessagesClaudeArmGetsSignedThinkingUnsigned(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeConsistentPolicyConfig(t, fake.URL(), "strict"))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	claude := actions["claude"].ActionID
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return claude })

	for _, tc := range []struct {
		name, request, contentType, answer string
	}{
		{"buffered", `{"model":"auto","messages":[{"role":"user","content":"run the tests"}]}`, "application/json", claudeSignedThinkingAnswer},
		{"streamed", `{"model":"auto","stream":true,"messages":[{"role":"user","content":"run the tests"}]}`, "text/event-stream", claudeSignedThinkingStream},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := &RequestContext{
				Headers: map[string]string{}, RequestID: "signature-" + tc.name,
				StartTime: time.Now(), TraceContext: context.Background(),
			}
			headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
					{Key: ":method", Value: "POST"},
					{Key: ":path", Value: "/v1/chat/completions"},
					{Key: "content-type", Value: "application/json"},
					{Key: "x-rayline-session", Value: "episode-signature-" + tc.name},
				}},
			}}
			if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
				t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
			}
			request, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{Body: []byte(tc.request), EndOfStream: true},
			}, ctx)
			if err != nil || request.GetImmediateResponse() != nil {
				t.Fatalf("request body: err=%v immediate=%v", err, request.GetImmediateResponse())
			}
			if !strings.Contains(string(request.GetRequestBody().GetResponse().GetBodyMutation().GetBody()), `"model":"anthropic/claude-opus-5","messages":[{"role":"user","content":[{"cache_control"`) {
				t.Fatalf("not dispatched to the Messages Claude arm: %s", request.GetRequestBody().GetResponse().GetBodyMutation().GetBody())
			}
			responseHeaders := arcResponseHeaders("200")
			responseHeaders.ResponseHeaders.Headers.Headers = append(responseHeaders.ResponseHeaders.Headers.Headers,
				&core.HeaderValue{Key: "content-type", Value: tc.contentType})
			if response, headerErr := router.handleResponseHeaders(responseHeaders, ctx); headerErr != nil || response.GetImmediateResponse() != nil {
				t.Fatalf("response headers: err=%v immediate=%v", headerErr, response.GetImmediateResponse())
			}
			response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{Body: []byte(tc.answer), EndOfStream: true},
			}, ctx)
			if err != nil {
				t.Fatalf("response body: %v", err)
			}
			if immediate := response.GetImmediateResponse(); immediate != nil {
				t.Fatalf("the answer was refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
			}
			client := string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
			if strings.Contains(client, "EqQBsigclaude") || strings.Contains(client, "signature") {
				t.Fatalf("a signature reached the Chat client: %s", client)
			}
			if !strings.Contains(client, `"reasoning_content":"Check the tool."`) || !strings.Contains(client, "Done.") {
				t.Fatalf("the Chat client lost the thinking or the answer: %s", client)
			}
		})
	}
}
