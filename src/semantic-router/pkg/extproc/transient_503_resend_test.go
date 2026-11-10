//go:build !windows && cgo

package extproc

import (
	"bytes"
	"net/http"
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The one re-send of a transient upstream refusal, end to end through the
// extproc phases: a plain arm (luna, which has no standard twin) is routed and
// dispatched, the cell's provider hop answers its headers with a 503, and a
// fake OpenRouter stands for the endpoint the call is sent to again.

const transientEdgeRefusal = "<html><body>503 Service Temporarily Unavailable</body></html>"

// quickTransientBackoff replaces the re-send's wait for the test.
func quickTransientBackoff(t *testing.T, wait time.Duration) {
	t.Helper()
	original := transientResendBackoff
	transientResendBackoff = func() time.Duration { return wait }
	t.Cleanup(func() { transientResendBackoff = original })
}

// answerHeadersAs delivers the provider hop's response headers with a
// content type ("" for none) and returns what the Router answered.
func answerHeadersAs(
	t *testing.T, router *OpenAIRouter, ctx *RequestContext, status int, contentType string,
) *ext_proc.ProcessingResponse {
	t.Helper()
	headers := flexUpstreamHeaders(status, contentType)
	if contentType == "" {
		headers.ResponseHeaders.Headers.Headers = headers.ResponseHeaders.Headers.Headers[:1]
	}
	stream := NewMockStream(nil)
	if err := router.processResponseHeaders(stream, headers, ctx); err != nil {
		t.Fatalf("processResponseHeaders: %v", err)
	}
	if len(stream.Responses) != 1 {
		t.Fatalf("the Router sent %d replies, want 1", len(stream.Responses))
	}
	return stream.Responses[0]
}

// A non-JSON 503 is re-sent once, after the backoff, with the dispatched body
// byte for byte, the arm's own credential and path, and the re-send's reply
// is the client's. Each call writes its own usage line; the refused one is
// unknown and never free.
func TestTransient503IsResentOnceAndTheResendServes(t *testing.T) {
	for _, contentType := range []string{"text/html", "text/plain; charset=utf-8", ""} {
		t.Run("content-type "+contentType, func(t *testing.T) {
			logs := captureLogs(t)
			quickTransientBackoff(t, 30*time.Millisecond)
			upstream := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
			router := flexResendRouter(upstream.server.URL)
			ctx, dispatched := routeFlexTurn(t, router, flexResendTwin, false)
			if ctx.TransientResend == nil || ctx.FlexResend != nil {
				t.Fatalf("a plain arm's dispatch staged transient=%v flex=%v", ctx.TransientResend, ctx.FlexResend)
			}

			started := time.Now()
			reply := answerHeadersAs(t, router, ctx, http.StatusServiceUnavailable, contentType)
			if elapsed := time.Since(started); elapsed < 30*time.Millisecond {
				t.Fatalf("the re-send went out after %v, before its backoff", elapsed)
			}

			immediate := reply.GetImmediateResponse()
			if immediate == nil {
				t.Fatalf("the 503 was not answered by the re-send: %v", reply)
			}
			if code := immediate.GetStatus().GetCode(); int(code) != http.StatusOK {
				t.Fatalf("client status = %d, want the re-send's 200", code)
			}
			if !strings.Contains(string(immediate.GetBody()), "served on standard") {
				t.Fatalf("the client did not get the re-send's reply:\n%s", immediate.GetBody())
			}
			calls := upstream.calls()
			if len(calls) != 1 {
				t.Fatalf("the endpoint was called %d times, want exactly once", len(calls))
			}
			if calls[0].path != "/api/v1/chat/completions" {
				t.Fatalf("re-send path = %q", calls[0].path)
			}
			if got := calls[0].headers.Get("Authorization"); got != "Bearer sk-standard-test" {
				t.Fatalf("re-send credential = %q, want the arm's own", got)
			}
			if !bytes.Equal(calls[0].body, dispatched) {
				t.Fatalf("the re-send changed the body:\ndispatched %s\nresent     %s", dispatched, calls[0].body)
			}

			lines := usageLines(logs)
			if len(lines) != 2 {
				t.Fatalf("got %d llm_usage lines, want one per call: %v", len(lines), lines)
			}
			assertFields(t, "refused attempt", lines[0], map[string]interface{}{
				"transient_resend": transientResendAttempt,
				"model":            flexResendTwin,
				"failure_class":    turnFailureUpstream5xx,
				"usage_source":     usageSourceUnknown,
				"pricing":          usagePricingNoUsage,
			})
			if lines[0]["cost"] != nil || lines[0]["prompt_tokens"] != nil {
				t.Fatalf("the refused attempt was settled with counts or a cost nobody stated: %v", lines[0])
			}
			assertFields(t, "re-send", lines[1], map[string]interface{}{
				"transient_resend":       transientResendServed,
				"model":                  flexResendTwin,
				"pricing":                usagePricingPriced,
				"provider_reported_cost": 0.00004,
				"upstream_attempts":      1,
			})
			if lines[1]["failure_class"] != nil || lines[1]["flex_resend"] != nil {
				t.Fatalf("the served re-send carried a failure class or a flex mark: %v", lines[1])
			}
			event := findLogEvent(t, logs, "transient_503_resend")
			assertFields(t, "transient_503_resend event", event, map[string]interface{}{
				"model":         flexResendTwin,
				"status":        http.StatusServiceUnavailable,
				"content_type":  contentType,
				"backoff_ms":    30,
				"resend_status": http.StatusOK,
			})
			if event["request_id"] != ctx.RequestID {
				t.Fatalf("event request_id = %v, want %q", event["request_id"], ctx.RequestID)
			}
		})
	}
}

// A 503 a provider states in JSON is its error, not an edge refusal, and no
// other status is re-sent: each takes the ordinary path untouched.
func TestJSON503AndOtherStatusesAreNotResent(t *testing.T) {
	cases := []struct {
		status      int
		contentType string
	}{
		{http.StatusServiceUnavailable, "application/json"},
		{http.StatusServiceUnavailable, "Application/JSON; charset=utf-8"},
		{http.StatusServiceUnavailable, "application/problem+json"},
		{http.StatusBadGateway, "text/html"},
		{http.StatusInternalServerError, ""},
		{http.StatusOK, "text/event-stream"},
	}
	for _, tc := range cases {
		t.Run(http.StatusText(tc.status)+" "+tc.contentType, func(t *testing.T) {
			upstream := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
			router := flexResendRouter(upstream.server.URL)
			ctx, _ := routeFlexTurn(t, router, flexResendTwin, false)

			reply := answerHeadersAs(t, router, ctx, tc.status, tc.contentType)
			if reply.GetImmediateResponse() != nil {
				t.Fatalf("a %d %q was re-sent: %v", tc.status, tc.contentType, reply)
			}
			if calls := upstream.calls(); len(calls) != 0 {
				t.Fatalf("the endpoint was called %d times", len(calls))
			}
			if ctx.TransientResent {
				t.Fatal("the turn was marked re-sent")
			}
		})
	}
}

// A re-send that is refused again is the client's outcome, through the
// ordinary pipeline, and there is no second re-send.
func TestTransient503ResendThatFailsIsTheClientsOutcome(t *testing.T) {
	logs := captureLogs(t)
	quickTransientBackoff(t, time.Millisecond)
	upstream := newFlexTwin(t, http.StatusServiceUnavailable, "text/html", transientEdgeRefusal)
	router := flexResendRouter(upstream.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendTwin, false)

	reply := answerHeadersAs(t, router, ctx, http.StatusServiceUnavailable, "text/html")

	immediate := reply.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("the 503 was not answered by the re-send: %v", reply)
	}
	if code := immediate.GetStatus().GetCode(); int(code) != http.StatusServiceUnavailable {
		t.Fatalf("client status = %d, want the re-send's 503", code)
	}
	if strings.Contains(string(immediate.GetBody()), "<html>") {
		t.Fatalf("the edge's page reached the client instead of an encoded error:\n%s", immediate.GetBody())
	}
	if calls := upstream.calls(); len(calls) != 1 {
		t.Fatalf("the endpoint was called %d times, want exactly once", len(calls))
	}
	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want 2: %v", len(lines), lines)
	}
	assertFields(t, "refused attempt", lines[0], map[string]interface{}{
		"transient_resend": transientResendAttempt, "failure_class": turnFailureUpstream5xx,
		"usage_source": usageSourceUnknown,
	})
	assertFields(t, "failed re-send", lines[1], map[string]interface{}{
		"transient_resend": transientResendServed, "failure_class": turnFailureUpstream5xx,
		"usage_source": usageSourceUnknown, "model": flexResendTwin,
	})
	if lines[1]["cost"] != nil {
		t.Fatalf("the failed re-send was settled with a cost: %v", lines[1])
	}
	if ctx.TransientResend != nil || !ctx.TransientResent {
		t.Fatalf("the turn could re-send again: plan=%v resent=%v", ctx.TransientResend, ctx.TransientResent)
	}
}

// A flex arm keeps its own re-send: a 503 on it is never re-sent here, staged
// or not, and the flex re-send never stacks one on top.
func TestTransient503ResendNeverStacksOnAFlexResend(t *testing.T) {
	quickTransientBackoff(t, time.Millisecond)
	upstream := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(upstream.server.URL)
	ctx, _ := routeFlexTurn(t, router, flexResendArm, false)
	if ctx.FlexResend == nil {
		t.Fatal("the flex arm's dispatch staged no flex re-send")
	}
	if ctx.TransientResend != nil {
		t.Fatal("a flex dispatch staged a transient re-send as well")
	}

	if reply := answerHeadersAs(t, router, ctx, http.StatusServiceUnavailable, "text/html"); reply.GetImmediateResponse() != nil {
		t.Fatalf("a flex arm's 503 was re-sent: %v", reply)
	}
	if calls := upstream.calls(); len(calls) != 0 {
		t.Fatalf("the endpoint was called %d times", len(calls))
	}

	// A plan staged before a flex re-send was spent does not fire either.
	ctx, _ = routeFlexTurn(t, router, flexResendTwin, false)
	ctx.FlexResend = &flexResendPlan{trigger: turnFailureRateLimited}
	if reply := answerHeadersAs(t, router, ctx, http.StatusServiceUnavailable, "text/html"); reply.GetImmediateResponse() != nil {
		t.Fatalf("a transient re-send stacked on a spent flex re-send: %v", reply)
	}
	if calls := upstream.calls(); len(calls) != 0 {
		t.Fatalf("the endpoint was called %d times", len(calls))
	}
}

// transient_503_resend: false turns the re-send off; it is on by default.
func TestTransient503ResendFlagTurnsItOff(t *testing.T) {
	upstream := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(upstream.server.URL)
	if !router.Config.Transient503ResendEnabled() {
		t.Fatal("the re-send is not on by default")
	}
	off := false
	router.Config.ResponseStreamTransient503Resend = &off
	ctx, _ := routeFlexTurn(t, router, flexResendTwin, false)
	if ctx.TransientResend != nil {
		t.Fatal("a re-send was staged with the flag off")
	}
	if reply := answerHeadersAs(t, router, ctx, http.StatusServiceUnavailable, "text/html"); reply.GetImmediateResponse() != nil {
		t.Fatalf("the 503 was re-sent with the flag off: %v", reply)
	}
	if calls := upstream.calls(); len(calls) != 0 {
		t.Fatalf("the endpoint was called %d times", len(calls))
	}
}

// A budget that leaves too little after the backoff is not spent: the
// endpoint is never dialled and the 503 takes the ordinary path. So is a
// budget turned off.
func TestTransient503ResendWithTooShortABudgetTakesTheOrdinaryPath(t *testing.T) {
	for _, tc := range []struct {
		budgetSec int
		reason    string
	}{
		{2, "budget_too_short"},
		// 5 s is flex's least budget, but the backoff comes out of it.
		{5, "budget_too_short"},
		{-1, "disabled"},
	} {
		t.Run(tc.reason, func(t *testing.T) {
			logs := captureLogs(t)
			quickTransientBackoff(t, 300*time.Millisecond)
			upstream := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
			router := flexResendRouter(upstream.server.URL)
			router.Config.FlexResendBudgetSec = tc.budgetSec
			ctx, _ := routeFlexTurn(t, router, flexResendTwin, false)

			if reply := answerHeadersAs(t, router, ctx, http.StatusServiceUnavailable, "text/html"); reply.GetImmediateResponse() != nil {
				t.Fatalf("a re-send ran on a %d s budget: %v", tc.budgetSec, reply)
			}
			stream := NewMockStream(nil)
			if err := router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{Body: []byte(transientEdgeRefusal), EndOfStream: true},
			}, ctx); err != nil {
				t.Fatalf("processResponseBody: %v", err)
			}
			if calls := upstream.calls(); len(calls) != 0 {
				t.Fatalf("the endpoint was dialled %d times on a skipped re-send", len(calls))
			}
			lines := usageLines(logs)
			if len(lines) != 1 {
				t.Fatalf("got %d llm_usage lines, want the refusal's one: %v", len(lines), lines)
			}
			assertFields(t, "refused call", lines[0], map[string]interface{}{
				"model": flexResendTwin, "failure_class": turnFailureUpstream5xx, "usage_source": usageSourceUnknown,
			})
			if lines[0]["transient_resend"] != nil {
				t.Fatalf("an unsent re-send marked the line: %v", lines[0])
			}
			skipped := findLogEvent(t, logs, "transient_503_resend_skipped")
			assertFields(t, "skipped event", skipped, map[string]interface{}{"reason": tc.reason})
		})
	}
}

// The backoff is jittered within 250-500 ms.
func TestTransient503ResendBackoffIsJitteredWithinBounds(t *testing.T) {
	seen := map[time.Duration]bool{}
	for range 200 {
		wait := transientResendBackoff()
		if wait < 250*time.Millisecond || wait > 500*time.Millisecond {
			t.Fatalf("backoff %v is outside 250-500 ms", wait)
		}
		seen[wait] = true
	}
	if len(seen) < 2 {
		t.Fatal("the backoff is not jittered")
	}
}

func TestJSONContentType(t *testing.T) {
	for value, want := range map[string]bool{
		"application/json":                 true,
		" application/json ; charset=utf8": true,
		"APPLICATION/JSON":                 true,
		"application/vnd.api+json":         true,
		"":                                 false,
		"text/html":                        false,
		"text/plain":                       false,
		"text/event-stream":                false,
		"application/jsonl":                false,
	} {
		if got := jsonContentType(value); got != want {
			t.Errorf("jsonContentType(%q) = %v, want %v", value, got, want)
		}
	}
}

// A streamed turn's 503 is re-sent the same way: the re-send's event stream
// reaches the client whole as one reply, settled from its own usage, and a
// staged host retry is dropped, since nothing is spliced into a reply that
// is already whole.
func TestTransient503ResendOfAStreamedTurnServesTheWholeStream(t *testing.T) {
	logs := captureLogs(t)
	quickTransientBackoff(t, time.Millisecond)
	sse := `data: {"id":"gen-2","object":"chat.completion.chunk","created":1,"model":"openai/gpt-luna","provider":"OpenAI",` +
		`"choices":[{"index":0,"delta":{"role":"assistant","content":"streamed after the edge"},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"gen-2","object":"chat.completion.chunk","created":1,"model":"openai/gpt-luna","provider":"OpenAI",` +
		`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":0.00004}}` + "\n\n" +
		"data: [DONE]\n\n"
	upstream := newFlexTwin(t, http.StatusOK, "text/event-stream", sse)
	router := flexResendRouter(upstream.server.URL)
	router.Config.ResponseStreamHostRetry = true
	router.Config.ResponseStreamSingleHostResend = true
	ctx, _ := routeFlexTurn(t, router, flexResendTwin, true)
	if ctx.HostRetry == nil {
		t.Fatal("the streamed dispatch staged no host retry")
	}
	ctx.FullDuplexResponseBody = true

	reply := answerHeadersAs(t, router, ctx, http.StatusServiceUnavailable, "text/html")

	immediate := reply.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("the streamed 503 was not answered by the re-send: %v", reply)
	}
	if !strings.Contains(string(immediate.GetBody()), "streamed after the edge") {
		t.Fatalf("the client did not get the re-send's stream:\n%s", immediate.GetBody())
	}
	if contentType := headerValuesByName(immediate.GetHeaders().GetSetHeaders())["content-type"]; !strings.Contains(contentType, "text/event-stream") {
		t.Fatalf("the re-sent stream travelled as %q", contentType)
	}
	if ctx.HostRetry != nil || ctx.FullDuplexResponseBody {
		t.Fatalf("the whole re-send kept a host retry (%v) or full duplex (%v)", ctx.HostRetry, ctx.FullDuplexResponseBody)
	}
	if calls := upstream.calls(); len(calls) != 1 {
		t.Fatalf("the endpoint was called %d times, want exactly once", len(calls))
	}
	lines := usageLines(logs)
	if len(lines) != 2 {
		t.Fatalf("got %d llm_usage lines, want 2: %v", len(lines), lines)
	}
	assertFields(t, "streamed re-send", lines[1], map[string]interface{}{
		"transient_resend": transientResendServed, "streaming": true, "provider_reported_cost": 0.00004,
	})
	if lines[1]["failure_class"] != nil {
		t.Fatalf("the served stream carried a failure class: %v", lines[1])
	}
}

// Only an OpenRouter call is re-sent: another provider's backend may be a
// pool of replicas, where the re-send could reach a different one.
func TestTransient503ResendOnlyStagesForOpenRouter(t *testing.T) {
	upstream := newFlexTwin(t, http.StatusOK, "application/json", flexTwinChatReply)
	router := flexResendRouter(upstream.server.URL)
	router.Config.ProviderProfiles["openrouter"] = config.ProviderProfile{Type: "openai", BaseURL: upstream.server.URL + "/v1"}
	ctx, _ := routeFlexTurn(t, router, flexResendTwin, false)
	if ctx.TransientResend != nil {
		t.Fatal("a non-OpenRouter dispatch staged a transient re-send")
	}
	if reply := answerHeadersAs(t, router, ctx, http.StatusServiceUnavailable, "text/html"); reply.GetImmediateResponse() != nil {
		t.Fatalf("a non-OpenRouter 503 was re-sent: %v", reply)
	}
	if calls := upstream.calls(); len(calls) != 0 {
		t.Fatalf("the endpoint was called %d times", len(calls))
	}
}
