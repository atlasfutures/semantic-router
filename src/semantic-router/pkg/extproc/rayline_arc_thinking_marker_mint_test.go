package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// End to end, the 2026-10-07 harness failure (pi T2/T3/T6): an open-weight
// worker's thinking reaches pi unsigned, pi resends it as text, and the next
// turn's Claude worker refuses another model's reasoning as the assistant's
// own words. With minting on, the thinking reaches the client signed with a
// Router marker, a client that keeps signed thinking resends it as thinking,
// and the Claude worker is sent none of it.

func writeMintingPolicyConfig(t *testing.T, policyURL string) string {
	t.Helper()
	path := writeClaudeOverChatPolicyConfig(t, policyURL)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(raw), "\nglobal:\n") != 1 || strings.Contains(string(raw), "\n  router:\n") {
		t.Fatal("the policy e2e config's global section changed shape")
	}
	rendered := strings.Replace(string(raw), "\nglobal:\n", "\nglobal:\n  router:\n    thinking_markers:\n      mint: true\n", 1)
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// mintingTurn runs one Messages client turn through the request phases and
// returns its context, for the response phases to run on.
func mintingTurn(t *testing.T, router *OpenAIRouter, episode, client string) *RequestContext {
	t.Helper()
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: fmt.Sprintf("mint-%s-%d", episode, time.Now().UnixNano()),
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
	if err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request body: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	return ctx
}

var mintedMarker = regexp.MustCompile(`vsr\.thinking\.v1\.[a-z0-9-]+\.[A-Za-z0-9_-]{32}`)

const openWeightChatReply = `{"id":"gen-1","object":"chat.completion","created":1,"model":"vendor/off","choices":[{"index":0,"finish_reason":"stop",` +
	`"message":{"role":"assistant","content":"Looking at it.","reasoning":"kimi reasoning"}}],` +
	`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

func openWeightChatStream() string {
	chunk := func(delta string) string {
		return `data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"vendor/off","choices":[{"index":0,"delta":` + delta + `,"finish_reason":null}]}` + "\n\n"
	}
	return chunk(`{"role":"assistant","content":"","reasoning":"kimi "}`) +
		chunk(`{"content":"","reasoning":"reasoning"}`) +
		chunk(`{"content":"Looking at it."}`) +
		`data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"vendor/off","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}` + "\n\ndata: [DONE]\n\n"
}

func TestMintedMarkerKeepsAnOpenWeightModelsThinkingFromClaude(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeMintingPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	off := actions["off"].ActionID
	first := `{"model":"auto","max_tokens":4096,"messages":[{"role":"user","content":"fix the failing test"}]}`

	for _, streamed := range []bool{false, true} {
		name := map[bool]string{false: "buffered", true: "streamed"}[streamed]
		t.Run(name, func(t *testing.T) {
			// Turn 1: the open-weight worker over Chat thinks, and the
			// Messages client receives that thinking signed with a marker.
			fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return off })
			client := first
			if streamed {
				client = strings.Replace(first, `"max_tokens":4096,`, `"max_tokens":4096,"stream":true,`, 1)
			}
			ctx := mintingTurn(t, router, "mint-"+name, client)
			var reply string
			if streamed {
				sendHeaders(t, router, ctx, streamingResponseHeaders("200"))
				response := router.handleSemanticStreamingResponseBody([]byte(openWeightChatStream()), true, ctx)
				reply = string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
			} else {
				if _, err := router.handleResponseHeaders(arcResponseHeaders("200"), ctx); err != nil {
					t.Fatal(err)
				}
				response := router.handleNonStreamingResponseBody([]byte(openWeightChatReply), ctx, time.Second)
				reply = string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
			}
			marker := mintedMarker.FindString(reply)
			want := protocolcodec.ThinkingMarker(protocolcodec.ThinkingMarkerFamily(ctx.VSRSelectedModel), "kimi reasoning")
			if marker == "" || marker != want {
				t.Fatalf("turn 1 reached the client with marker %q, want %q:\n%s", marker, want, reply)
			}

			// Turn 2: the client resends that thinking signed, as pi does
			// once it is signed, and the next turn goes to Claude, on
			// Messages and over Chat. Claude is sent none of it.
			history := `{"model":"auto","max_tokens":4096,"messages":[` +
				`{"role":"user","content":"fix the failing test"},` +
				`{"role":"assistant","content":[{"type":"thinking","thinking":"kimi reasoning","signature":"` + marker + `"},` +
				`{"type":"text","text":"Looking at it."}]},` +
				`{"role":"user","content":"go on"}]}`
			for _, worker := range []string{"claude", "claude-off"} {
				action := actions[worker].ActionID
				fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return action })
				logs := captureLogs(t)
				body := dispatchPolicyClientRequest(t, router, "mint-"+name+"-"+worker, "/v1/messages", history)
				wire := string(body["messages"])
				if strings.Contains(wire, "vsr.") || strings.Contains(wire, "kimi reasoning") {
					t.Fatalf("%s was sent the open-weight worker's thinking: %s", worker, wire)
				}
				if !strings.Contains(wire, "Looking at it.") {
					t.Fatalf("%s lost the assistant's visible text: %s", worker, wire)
				}
				if dropped := findLogEvent(t, logs, "reasoning_dropped"); fmt.Sprint(dropped["foreign_dropped"]) != "1" {
					t.Fatalf("%s: reasoning_dropped = %v, want 1 foreign drop", worker, dropped)
				}
			}

			// The same history to the open-weight worker keeps its
			// reasoning, unsigned.
			fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return off })
			body := dispatchPolicyClientRequest(t, router, "mint-"+name+"-off", "/v1/messages", history)
			if wire := string(body["messages"]); strings.Contains(wire, "vsr.") ||
				!strings.Contains(wire, `"reasoning_content":"kimi reasoning"`) {
				t.Fatalf("the open-weight worker was not sent its reasoning unsigned: %s", wire)
			}
		})
	}
}

// With minting off (the default), turn 1 reaches the client unsigned, as
// before.
func TestThinkingMarkersAreNotMintedByDefault(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	fake := newRelaxedPolicyFake(t)
	router, err := NewOpenAIRouter(writeClaudeOverChatPolicyConfig(t, fake.URL()))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	ctx := mintingTurn(t, router, "mint-default",
		`{"model":"auto","max_tokens":4096,"messages":[{"role":"user","content":"fix the failing test"}]}`)
	if _, err := router.handleResponseHeaders(arcResponseHeaders("200"), ctx); err != nil {
		t.Fatal(err)
	}
	response := router.handleNonStreamingResponseBody([]byte(openWeightChatReply), ctx, time.Second)
	reply := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	if strings.Contains(string(reply), "vsr.") {
		t.Fatalf("a marker was minted with minting off: %s", reply)
	}
	var body struct {
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(reply, &body); err != nil || len(body.Content) == 0 || !strings.Contains(string(body.Content[0]), `"thinking":"kimi reasoning"`) {
		t.Fatalf("turn 1 lost its thinking: %s", reply)
	}
}
