//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const emptyRetryArm = "deepseek-v4-pro@thinking-off"

// answeredChatCompletionBody is the turn a second provider returns: the same
// request, served.
const answeredChatCompletionBody = `{
  "id": "gen-1757030001-Kp9",
  "provider": "Ionstream",
  "model": "deepseek/deepseek-v4-pro",
  "object": "chat.completion",
  "created": 1757030001,
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {"role": "assistant", "content": "an answer"}
    }
  ],
  "usage": {"prompt_tokens": 31402, "completion_tokens": 118, "total_tokens": 31520}
}`

// reasonedEmptyCompletionBody names no content either, but the upstream billed
// 64 completion tokens for it. Something was generated; retrying would pay for
// the turn twice and could discard a real answer the codec refused for another
// reason.
const reasonedEmptyCompletionBody = `{
  "id": "gen-1757030002-Ww2",
  "provider": "StreamLake",
  "model": "deepseek/deepseek-v4-pro",
  "object": "chat.completion",
  "created": 1757030002,
  "choices": [
    {
      "index": 0,
      "finish_reason": "length",
      "message": {"role": "assistant", "content": null}
    }
  ],
  "usage": {"prompt_tokens": 31402, "completion_tokens": 64, "total_tokens": 31466}
}`

type recordedUpstream struct {
	server *httptest.Server
	mutex  sync.Mutex
	bodies [][]byte
}

func newRecordedUpstream(t *testing.T, replies ...string) *recordedUpstream {
	t.Helper()
	upstream := &recordedUpstream{}
	upstream.server = httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, request *http.Request) {
			body := make([]byte, request.ContentLength)
			if _, err := request.Body.Read(body); err != nil && len(body) == 0 {
				t.Errorf("read retry body: %v", err)
			}
			upstream.mutex.Lock()
			index := len(upstream.bodies)
			upstream.bodies = append(upstream.bodies, body)
			upstream.mutex.Unlock()
			writer.Header().Set("content-type", "application/json")
			if index < len(replies) {
				_, _ = writer.Write([]byte(replies[index]))
				return
			}
			http.Error(writer, "no reply was staged", http.StatusInternalServerError)
		},
	))
	t.Cleanup(upstream.server.Close)
	return upstream
}

func (upstream *recordedUpstream) calls() [][]byte {
	upstream.mutex.Lock()
	defer upstream.mutex.Unlock()
	return append([][]byte(nil), upstream.bodies...)
}

func emptyRetryRouter(baseURL string) *OpenAIRouter {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: emptyRetryArm,
			ModelConfig: map[string]config.ModelParams{emptyRetryArm: {
				PreferredEndpoints: []string{"backend"},
				APIFormat:          config.APIFormatOpenAI,
				ExternalModelIDs:   map[string]string{"backend": "deepseek/deepseek-v4-pro"},
			}},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "backend", Address: "127.0.0.1", Port: 8000,
				Type: "openai", APIKey: "test-key",
				ProviderProfileName: "provider",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"provider": {Type: "openai", BaseURL: baseURL},
			},
		},
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
}

// dispatchEmptyRetryTurn routes one turn through the request path, so the
// context carries whatever a second attempt would need.
func dispatchEmptyRetryTurn(t *testing.T, router *OpenAIRouter) *RequestContext {
	t.Helper()
	request := testNeutralRequest(emptyRetryArm, "call the tool")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.RequestModel = emptyRetryArm
	ctx.VSRSelectedModel = emptyRetryArm
	if _, err := router.handleEntrypointModelRouting(
		request, emptyRetryArm, "", entropy.ReasoningDecision{}, emptyRetryArm, ctx,
	); err != nil {
		t.Fatalf("route the turn: %v", err)
	}
	return ctx
}

func decodeRetryRequest(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode retry request %q: %v", raw, err)
	}
	return body
}

// The provider answered with a stop token and nothing else. The same request
// goes back once, told to skip the provider that returned nothing, and the
// answer that comes back is the one the client is served.
func TestEmptyCompletionIsRetriedWithoutTheProviderThatReturnedIt(t *testing.T) {
	logs := captureLogs(t)
	upstream := newRecordedUpstream(t, answeredChatCompletionBody)
	router := emptyRetryRouter(upstream.server.URL)
	ctx := dispatchEmptyRetryTurn(t, router)

	response := router.handleNonStreamingResponseBody([]byte(emptyChatCompletionBody), ctx, time.Second)

	calls := upstream.calls()
	if len(calls) != 1 {
		t.Fatalf("upstream was called %d times, want exactly one retry", len(calls))
	}
	preferences, _ := decodeRetryRequest(t, calls[0])["provider"].(map[string]any)
	ignored, _ := preferences["ignore"].([]any)
	if len(ignored) != 1 || ignored[0] != "StreamLake" {
		t.Fatalf("provider preferences = %#v, want the empty provider ignored", preferences)
	}
	if response.GetImmediateResponse() != nil {
		t.Fatal("an answered retry must not refuse the turn")
	}
	served := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	if len(served) == 0 {
		t.Fatal("the retry's answer must replace the empty body")
	}
	if got, _ := decodeRetryRequest(t, served)["id"].(string); got != "gen-1757030001-Kp9" {
		t.Fatalf("served response id = %q, want the retry's", got)
	}

	retry := findLogEvent(t, logs, "upstream_empty_retry")
	if got, _ := retry["excluded_provider"].(string); got != "StreamLake" {
		t.Fatalf("excluded_provider = %v, want the provider that returned nothing", retry["excluded_provider"])
	}
	if got, _ := retry["arm"].(string); got != emptyRetryArm {
		t.Fatalf("arm = %v, want the selected arm", retry["arm"])
	}
	if got, _ := retry["retry_provider"].(string); got != "Ionstream" {
		t.Fatalf("retry_provider = %v, want the provider that answered", retry["retry_provider"])
	}
	if got, _ := retry["outcome"].(string); got != "answered" {
		t.Fatalf("outcome = %v, want the answered retry", retry["outcome"])
	}
}

// Both attempts were billed. A turn that cost a stop token and then a whole
// answer settles for the sum, or the cell under-reports every retry it makes.
func TestRetriedTurnIsBilledForBothAttempts(t *testing.T) {
	logs := captureLogs(t)
	upstream := newRecordedUpstream(t, answeredChatCompletionBody)
	router := emptyRetryRouter(upstream.server.URL)
	ctx := dispatchEmptyRetryTurn(t, router)

	router.handleNonStreamingResponseBody([]byte(emptyChatCompletionBody), ctx, time.Second)

	fields := findLogEvent(t, logs, "llm_usage")
	if got, _ := fields["completion_tokens"].(int64); got != 119 {
		t.Fatalf("completion_tokens = %v, want the empty token plus the answer", fields["completion_tokens"])
	}
	if got, _ := fields["prompt_tokens"].(int64); got != 62804 {
		t.Fatalf("prompt_tokens = %v, want both attempts' prompts", fields["prompt_tokens"])
	}
	if got, _ := fields["retries"].(int64); got != 1 {
		t.Fatalf("retries = %v, want the one retry the turn made", fields["retries"])
	}
	if got, _ := fields["upstream_provider"].(string); got != "Ionstream" {
		t.Fatalf("upstream_provider = %v, want the provider that answered", fields["upstream_provider"])
	}
}

// A turn billed for more than a stop token generated something. Retrying it
// would pay twice and could throw away an answer refused for another reason.
func TestCompletionBilledForMoreThanAStopTokenIsNeverRetried(t *testing.T) {
	upstream := newRecordedUpstream(t, answeredChatCompletionBody)
	router := emptyRetryRouter(upstream.server.URL)
	ctx := dispatchEmptyRetryTurn(t, router)

	response := router.handleNonStreamingResponseBody([]byte(reasonedEmptyCompletionBody), ctx, time.Second)

	if calls := upstream.calls(); len(calls) != 0 {
		t.Fatalf("upstream was called %d times, want no retry", len(calls))
	}
	if response.GetImmediateResponse() == nil {
		t.Fatal("a turn that is not retried must still be refused")
	}
}

// One retry per turn. A provider pool that is answering nothing must not be
// walked one exclusion at a time on the client's latency.
func TestASecondEmptyAnswerEndsTheTurn(t *testing.T) {
	logs := captureLogs(t)
	upstream := newRecordedUpstream(t, emptyChatCompletionBody, answeredChatCompletionBody)
	router := emptyRetryRouter(upstream.server.URL)
	ctx := dispatchEmptyRetryTurn(t, router)

	response := router.handleNonStreamingResponseBody([]byte(emptyChatCompletionBody), ctx, time.Second)

	if calls := upstream.calls(); len(calls) != 1 {
		t.Fatalf("upstream was called %d times, want exactly one retry", len(calls))
	}
	if response.GetImmediateResponse() == nil {
		t.Fatal("a second empty answer must fail the turn")
	}
	retry := findLogEvent(t, logs, "upstream_empty_retry")
	if got, _ := retry["outcome"].(string); got != "empty" {
		t.Fatalf("outcome = %v, want the second empty answer", retry["outcome"])
	}
}

// A streamed turn has already sent bytes by the time an empty is visible, so
// no plan is kept for it and no retry is attempted.
func TestStreamedTurnsAreNotRetried(t *testing.T) {
	upstream := newRecordedUpstream(t, answeredChatCompletionBody)
	router := emptyRetryRouter(upstream.server.URL)
	request := testNeutralRequest(emptyRetryArm, "call the tool")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.RequestModel = emptyRetryArm
	ctx.ExpectStreamingResponse = true
	if _, err := router.handleEntrypointModelRouting(
		request, emptyRetryArm, "", entropy.ReasoningDecision{}, emptyRetryArm, ctx,
	); err != nil {
		t.Fatalf("route the turn: %v", err)
	}

	router.handleNonStreamingResponseBody([]byte(emptyChatCompletionBody), ctx, time.Second)

	if calls := upstream.calls(); len(calls) != 0 {
		t.Fatalf("upstream was called %d times, want no retry on a streamed turn", len(calls))
	}
}
