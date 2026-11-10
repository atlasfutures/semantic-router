package extproc

import (
	"bytes"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

const edgeHTMLBody = "<html><body><h1>503 Service Temporarily Unavailable</h1></body></html>"

// edgeRefusedTurn dispatches a turn on the claude arm, points its staged
// edge re-send at upstream, and answers the dispatch with status and body the
// way FULL_DUPLEX_STREAMED delivers them (an empty body ends at the headers).
func edgeRefusedTurn(t *testing.T, upstream *claudeUpstream, status int, body string) (*RequestContext, *MockStream, []byte) {
	t.Helper()
	previous := edgeResendJitter
	edgeResendJitter = func() time.Duration { return time.Millisecond }
	t.Cleanup(func() { edgeResendJitter = previous })

	router, _ := newThinkingGuardRouter(t, "claude")
	ctx, _ := dispatchThinkingTurn(t, router, "episode-edge-503", toolLoopHistory("sig-earlier", "sig-loop"))
	plan := ctx.EdgeResend
	if plan == nil {
		t.Fatal("no edge re-send was staged")
	}
	// The signature guard answers only a 400; leave it out of the way.
	ctx.ThinkingSignatureRetry = nil
	dispatch := *plan.dispatch
	profile := *dispatch.profile
	profile.BaseURL = upstream.server.URL
	dispatch.profile = &profile
	dispatch.backendAddress = upstream.server.Listener.Addr().String()
	plan.dispatch = &dispatch
	dispatched := bytes.Clone(plan.dispatched)

	ctx.FullDuplexResponseBody = true
	var messages []*ext_proc.ProcessingRequest
	if body != "" {
		messages = append(messages, &ext_proc.ProcessingRequest{
			Request: &ext_proc.ProcessingRequest_ResponseBody{ResponseBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true}},
		})
	}
	ctx.streamReceiver = startStreamReceiver(NewMockStream(messages))
	contentType := "text/html"
	if strings.HasPrefix(body, "{") {
		contentType = "application/json"
	}
	envoy := NewMockStream(nil)
	headers := &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", RawValue: []byte(strconv.Itoa(status))},
			{Key: "content-type", RawValue: []byte(contentType)},
		}},
		EndOfStream: body == "",
	}}
	if err := router.processResponseHeaders(envoy, headers, ctx); err != nil {
		t.Fatalf("processResponseHeaders: %v", err)
	}
	if len(envoy.Responses) != 1 {
		t.Fatalf("the Router sent %d replies, want 1", len(envoy.Responses))
	}
	return ctx, envoy, dispatched
}

// A non-JSON 503 (an HTML page, or no body at all) is re-sent once with the
// dispatched bytes unchanged, and the client gets the re-send's reply. Both
// calls are marked on llm_usage.
func TestEdge503IsResentOnceWithTheSameBytes(t *testing.T) {
	for name, body := range map[string]string{"html": edgeHTMLBody, "empty": ""} {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			upstream := newClaudeUpstream(t, http.StatusOK, claudeServedReply)
			ctx, envoy, _ := edgeRefusedTurn(t, upstream, http.StatusServiceUnavailable, body)

			immediate := envoy.Responses[0].GetImmediateResponse()
			if immediate == nil || int(immediate.GetStatus().GetCode()) != http.StatusOK ||
				!strings.Contains(string(immediate.GetBody()), "served after the strip") {
				t.Fatalf("the client did not get the re-send's reply: %v", envoy.Responses[0])
			}
			calls := upstream.calls()
			if len(calls) != 1 {
				t.Fatalf("the upstream was called %d times, want exactly once", len(calls))
			}
			if ctx.EdgeResend != nil {
				t.Fatal("the plan was kept: a turn re-sends once")
			}
			lines := usageLines(logs)
			if len(lines) != 2 {
				t.Fatalf("got %d llm_usage lines, want one per call: %v", len(lines), lines)
			}
			assertFields(t, "refused call", lines[0], map[string]interface{}{
				"edge_resend": edgeResendAttempt, "failure_class": turnFailureUpstream5xx,
			})
			assertFields(t, "re-send", lines[1], map[string]interface{}{"edge_resend": edgeResendServed})
			// The re-send was read whole: it states no first byte or first content.
			if lines[1]["ttft_ms"] != nil || lines[1]["first_content_ms"] != nil {
				t.Fatalf("the re-send stated ttft_ms %v, first_content_ms %v", lines[1]["ttft_ms"], lines[1]["first_content_ms"])
			}
			event := findLogEvent(t, logs, "edge_resend")
			assertFields(t, "edge_resend", event, map[string]interface{}{
				"refused_status": http.StatusServiceUnavailable, "resend_status": http.StatusOK,
			})
		})
	}
}

// The re-send carries exactly the bytes the first call dispatched (the
// privacy invariant).
func TestEdge503ResendSendsTheDispatchedBytes(t *testing.T) {
	upstream := newClaudeUpstream(t, http.StatusOK, claudeServedReply)
	_, _, dispatched := edgeRefusedTurn(t, upstream, http.StatusServiceUnavailable, edgeHTMLBody)
	calls := upstream.calls()
	if len(calls) != 1 || !bytes.Equal(calls[0], dispatched) {
		t.Fatalf("the re-send's body differs from the dispatched bytes:\n%s\n%s", calls, dispatched)
	}
}

// A 503 carrying OpenRouter's JSON error, and any other status, take the
// ordinary path: no re-send, and the body read ahead is handed back.
func TestJSON503AndOtherStatusesAreNotEdgeResent(t *testing.T) {
	jsonError := `{"error":{"code":503,"message":"No instances available"}}`
	for _, tc := range []struct {
		status int
		body   string
	}{
		{http.StatusServiceUnavailable, jsonError},
		{http.StatusBadGateway, edgeHTMLBody},
		{http.StatusTooManyRequests, edgeHTMLBody},
	} {
		upstream := newClaudeUpstream(t, http.StatusOK, claudeServedReply)
		ctx, envoy, _ := edgeRefusedTurn(t, upstream, tc.status, tc.body)
		if envoy.Responses[0].GetImmediateResponse() != nil {
			t.Fatalf("status %d body %q was replaced: %v", tc.status, tc.body, envoy.Responses[0])
		}
		if calls := upstream.calls(); len(calls) != 0 {
			t.Fatalf("status %d was re-sent %d times", tc.status, len(calls))
		}
		if tc.status != http.StatusServiceUnavailable {
			continue
		}
		// The 503's body was read ahead to be judged: it is handed back.
		message, ok := ctx.streamReceiver.poll()
		if !ok || string(message.request.GetResponseBody().GetBody()) != tc.body {
			t.Fatalf("status %d: the body read ahead was not handed back: %v %v", tc.status, ok, message.request)
		}
	}
}

// A re-send refused again is the client's answer: there is no second.
func TestEdge503IsNeverResentTwice(t *testing.T) {
	upstream := newClaudeUpstream(t, http.StatusServiceUnavailable, edgeHTMLBody)
	_, envoy, _ := edgeRefusedTurn(t, upstream, http.StatusServiceUnavailable, edgeHTMLBody)
	if immediate := envoy.Responses[0].GetImmediateResponse(); immediate == nil ||
		int(immediate.GetStatus().GetCode()) != http.StatusServiceUnavailable {
		t.Fatalf("the re-send's 503 did not reach the client: %v", envoy.Responses[0])
	}
	if calls := upstream.calls(); len(calls) != 1 {
		t.Fatalf("the upstream was called %d times, want exactly once", len(calls))
	}
}

func TestEdgeErrorBodyIsAnythingButAJSONObject(t *testing.T) {
	for body, edge := range map[string]bool{
		"": true, edgeHTMLBody: true, "null": true, "[]": true, `"busy"`: true, "{": true,
		`{"error":{"code":503}}`: false, "{}": false,
	} {
		if got := edgeErrorBody([]byte(body)); got != edge {
			t.Errorf("edgeErrorBody(%q) = %v, want %v", body, got, edge)
		}
	}
}

func TestEdgeResendJitterIsWithinBounds(t *testing.T) {
	for range 200 {
		if wait := edgeResendJitter(); wait < 250*time.Millisecond || wait >= 500*time.Millisecond {
			t.Fatalf("jitter %v outside [250ms, 500ms)", wait)
		}
	}
}
