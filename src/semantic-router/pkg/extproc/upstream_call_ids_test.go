//go:build !windows && cgo

package extproc

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.uber.org/zap/zaptest/observer"
)

// The provider's own ids for a call ride its llm_usage and turn_failed
// lines, so a turn can be looked up at the provider: OpenRouter's generation
// id, the provider that served it, and a provider request id header. Only ids
// the upstream sent; never a body, a credential or another header.

const providerIDSecret = "sk-or-v1-provider-ids-secret"

func providerIDHeaders(status int, contentType string, extra ...string) *ext_proc.ProcessingRequest_ResponseHeaders {
	headers := []*core.HeaderValue{
		{Key: ":status", RawValue: []byte(strconv.Itoa(status))},
		{Key: "content-type", RawValue: []byte(contentType)},
		// Never logged: a credential echoed back and a cookie.
		{Key: "authorization", RawValue: []byte("Bearer " + providerIDSecret)},
		{Key: "set-cookie", RawValue: []byte("session=" + providerIDSecret)},
	}
	for index := 0; index+1 < len(extra); index += 2 {
		headers = append(headers, &core.HeaderValue{Key: extra[index], RawValue: []byte(extra[index+1])})
	}
	return &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: headers},
	}}
}

func deliverHeaders(t *testing.T, router *OpenAIRouter, ctx *RequestContext, headers *ext_proc.ProcessingRequest_ResponseHeaders) {
	t.Helper()
	if err := router.processResponseHeaders(NewMockStream(nil), headers, ctx); err != nil {
		t.Fatalf("processResponseHeaders: %v", err)
	}
}

func deliverBody(t *testing.T, router *OpenAIRouter, ctx *RequestContext, body string, end bool) {
	t.Helper()
	if err := router.processResponseBody(NewMockStream(nil), &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: end},
	}, ctx); err != nil {
		t.Fatalf("processResponseBody: %v", err)
	}
}

// assertNoSecretLogged fails when any logged line carries the secret.
func assertNoSecretLogged(t *testing.T, logs *observer.ObservedLogs) {
	t.Helper()
	for _, entry := range logs.All() {
		if line := fmt.Sprintf("%s %v", entry.Message, entry.ContextMap()); strings.Contains(line, providerIDSecret) {
			t.Fatalf("a log line carried the secret: %s", line)
		}
	}
}

func TestBufferedUsageLineNamesTheProviderIDs(t *testing.T) {
	logs := captureLogs(t)
	twin := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(twin.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendTwin, false)

	deliverHeaders(t, router, ctx, providerIDHeaders(http.StatusOK, "application/json",
		"request-id", "req_011CAnthropic", "cf-ray", "8f1e2d3c4b5a6978-SJC", "x-vsr-upstream-attempts", "1"))
	deliverBody(t, router, ctx, flexTwinChatReply, true)

	assertFields(t, "buffered usage", findLogEvent(t, logs, "llm_usage"), map[string]interface{}{
		"upstream_generation_id":     "gen-1",
		"upstream_provider":          "OpenAI",
		"upstream_request_id":        "req_011CAnthropic",
		"upstream_request_id_header": "request-id",
		"upstream_attempts":          1,
	})
	assertNoSecretLogged(t, logs)
}

const providerIDStream = `data: {"id":"gen-2","object":"chat.completion.chunk","created":1,"model":"openai/gpt-luna","provider":"OpenAI",` +
	`"choices":[{"index":0,"delta":{"role":"assistant","content":"streamed"},"finish_reason":null}]}` + "\n\n" +
	`data: {"id":"gen-2","object":"chat.completion.chunk","created":1,"model":"openai/gpt-luna","provider":"OpenAI",` +
	`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.00004}}` + "\n\n" +
	"data: [DONE]\n\n"

func TestStreamedUsageLineNamesTheProviderIDs(t *testing.T) {
	logs := captureLogs(t)
	twin := newFlexTwin(t, http.StatusOK, "text/event-stream", providerIDStream)
	router := flexResendRouter(twin.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendTwin, true)

	deliverHeaders(t, router, ctx, providerIDHeaders(http.StatusOK, "text/event-stream", "cf-ray", "8f1e2d3c4b5a6978-SJC"))
	deliverBody(t, router, ctx, providerIDStream, true)

	usage := findLogEvent(t, logs, "llm_usage")
	assertFields(t, "streamed usage", usage, map[string]interface{}{
		"streaming":                  true,
		"upstream_generation_id":     "gen-2",
		"upstream_provider":          "OpenAI",
		"upstream_request_id":        "8f1e2d3c4b5a6978-SJC",
		"upstream_request_id_header": "cf-ray",
	})
	assertNoSecretLogged(t, logs)
}

// A call whose upstream stated no id says so: the members are null, not
// absent and never invented.
func TestUsageLineWithoutProviderIDsIsNull(t *testing.T) {
	logs := captureLogs(t)
	twin := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(twin.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendTwin, false)
	noID := `{"object":"chat.completion","created":1,"model":"openai/gpt-luna",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"no id"},"finish_reason":"stop"}]}`

	deliverHeaders(t, router, ctx, providerIDHeaders(http.StatusOK, "application/json"))
	deliverBody(t, router, ctx, noID, true)

	usage := findLogEvent(t, logs, "llm_usage")
	for _, name := range []string{"upstream_generation_id", "upstream_request_id", "upstream_request_id_header"} {
		if value, present := usage[name]; !present || value != nil {
			t.Fatalf("%s = %v (present %v), want null", name, value, present)
		}
	}
}

// A stream cut at the first-content limit (OpenRouter's keepalives and
// nothing else) still names what the upstream did say: the headers' ids, and
// the first chunk's id and provider when one arrived before the cut.
func TestFirstContentTimeoutNamesTheProviderIDs(t *testing.T) {
	roleOnly := `data: {"id":"gen-cut","object":"chat.completion.chunk","created":1,"model":"openai/gpt-luna","provider":"Moonshot",` +
		`"choices":[]}` + "\n\n"
	for name, test := range map[string]struct {
		headers    []string
		first      string
		generation interface{}
		provider   interface{}
		requestID  interface{}
	}{
		"header ids only": {
			headers:    []string{"x-generation-id", "gen-header", "x-request-id", "req-or-1"},
			first:      openRouterKeepaliveChunk,
			generation: "gen-header", provider: nil, requestID: "req-or-1",
		},
		"first chunk before the keepalives": {
			first:      roleOnly + openRouterKeepaliveChunk,
			generation: "gen-cut", provider: "Moonshot", requestID: nil,
		},
	} {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			twin := newFlexTwin(t, http.StatusOK, "text/event-stream", providerIDStream)
			router := flexResendRouter(twin.server.URL)
			router.Config.ResponseStreamFirstContentSec = 60
			ctx, _ := routeFlexTurn(t, router, flexResendTwin, true)
			headers := append([]string{"x-vsr-upstream-attempts", "2"}, test.headers...)
			deliverHeaders(t, router, ctx, providerIDHeaders(http.StatusOK, "text/event-stream", headers...))
			ctx.StreamContentSince = time.Now().Add(-90 * time.Second)

			if err := router.handleProcessRequest(NewMockStream(nil), fullDuplexResponseBodyRequest(test.first, false), ctx); err != nil {
				t.Fatalf("handleProcessRequest: %v", err)
			}

			failed := findLogEvent(t, logs, "turn_failed")
			assertFields(t, "turn_failed", failed, map[string]interface{}{
				"failure_class":               turnFailureTimeout,
				"content_sent_before_failure": false,
				"upstream_attempts":           2,
			})
			for field, want := range map[string]interface{}{
				"upstream_generation_id": test.generation,
				"upstream_provider":      test.provider,
				"upstream_request_id":    test.requestID,
			} {
				if got, present := failed[field]; !present || got != want {
					t.Fatalf("turn_failed %s = %v (present %v), want %v", field, got, present, want)
				}
			}
			if got := findLogEvent(t, logs, "llm_usage")["upstream_generation_id"]; got != test.generation {
				t.Fatalf("llm_usage upstream_generation_id = %v, want %v", got, test.generation)
			}
			assertNoSecretLogged(t, logs)
		})
	}
}

// A provider error before any byte (a 429 here) names what its headers said.
func TestProviderErrorTurnFailedNamesTheProviderIDs(t *testing.T) {
	logs := captureLogs(t)
	twin := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(twin.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendTwin, false)

	deliverHeaders(t, router, ctx, providerIDHeaders(http.StatusTooManyRequests, "application/json",
		"x-generation-id", "gen-429", "x-vsr-upstream-attempts", "3"))
	deliverBody(t, router, ctx, `{"error":{"message":"Rate limit exceeded","code":429}}`, true)

	failed := findLogEvent(t, logs, "turn_failed")
	assertFields(t, "turn_failed", failed, map[string]interface{}{
		"failure_class": turnFailureRateLimited, "upstream_generation_id": "gen-429", "upstream_attempts": 3,
	})
	if failed["upstream_request_id"] != nil {
		t.Fatalf("an id nobody sent was logged: %v", failed["upstream_request_id"])
	}
	assertNoSecretLogged(t, logs)
}

// A flex re-send's line names the re-send's ids, read off the twin's own
// response, and the refused attempt's line keeps the refused call's.
func TestFlexResendLineNamesTheTwinsProviderIDs(t *testing.T) {
	logs := captureLogs(t)
	twin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "application/json")
		w.Header().Set("request-id", "req_twin")
		w.Header().Set("authorization", "Bearer "+providerIDSecret)
		_, _ = io.WriteString(w, flexTwinChatReply)
	}))
	t.Cleanup(twin.Close)
	router := flexResendRouter(twin.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendArm, false)

	deliverHeaders(t, router, ctx, providerIDHeaders(http.StatusTooManyRequests, "application/json", "request-id", "req_flex"))

	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want 2: %v", len(lines), lines)
	}
	assertFields(t, "flex attempt", lines[0], map[string]interface{}{
		"flex_resend": flexResendAttempt, "upstream_request_id": "req_flex",
	})
	if lines[0]["upstream_generation_id"] != nil {
		t.Fatalf("the refused attempt carried a generation id: %v", lines[0])
	}
	assertFields(t, "standard re-send", lines[1], map[string]interface{}{
		"flex_resend": flexResendStandard, "upstream_request_id": "req_twin",
		"upstream_request_id_header": "request-id", "upstream_generation_id": "gen-1",
	})
	assertNoSecretLogged(t, logs)
}

func TestUpstreamCallIDsFromHeaders(t *testing.T) {
	header := func(pairs ...string) *core.HeaderMap {
		headerMap := &core.HeaderMap{}
		for index := 0; index+1 < len(pairs); index += 2 {
			headerMap.Headers = append(headerMap.Headers, &core.HeaderValue{Key: pairs[index], Value: pairs[index+1]})
		}
		return headerMap
	}
	long := strings.Repeat("a", 300)
	for name, test := range map[string]struct {
		headers     *core.HeaderMap
		want        upstreamCallIDs
		callerReqID string
	}{
		"none":                  {headers: header("content-type", "application/json"), want: upstreamCallIDs{}},
		"anthropic first":       {headers: header("cf-ray", "ray", "x-request-id", "xr", "Request-Id", "req"), want: upstreamCallIDs{requestID: "req", requestIDHeader: "request-id"}},
		"caller echo is not it": {headers: header("x-request-id", "caller-1", "cf-ray", "ray"), callerReqID: "caller-1", want: upstreamCallIDs{requestID: "ray", requestIDHeader: "cf-ray"}},
		"bounded and printable": {headers: header("X-Generation-Id", "gen-\x01"+long), want: upstreamCallIDs{generationID: ("gen-" + long)[:maxUpstreamIDLength]}},
		"blank is absent":       {headers: header("x-generation-id", "  ", "request-id", ""), want: upstreamCallIDs{}},
	} {
		t.Run(name, func(t *testing.T) {
			if got := upstreamCallIDsFromHeaders(test.headers, test.callerReqID); got != test.want {
				t.Fatalf("ids = %+v, want %+v", got, test.want)
			}
		})
	}
}
