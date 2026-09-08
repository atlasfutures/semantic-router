package testcases

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// errorEnvelopeBody is the error envelope an OpenAI-compatible backend returns
// when a turn produced no answer.
const errorEnvelopeBody = `{"error":{"message":"upstream connection error","type":"server_error"}}`

// bodiesServedWith200 are three bodies a backend can return under HTTP 200.
// Only the first is a completed turn.
var bodiesServedWith200 = []struct {
	name             string
	body             string
	isACompletedTurn bool
}{
	{
		name: "chat completion",
		body: `{"id":"chatcmpl-1","object":"chat.completion","choices":` +
			`[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}]}`,
		isACompletedTurn: true,
	},
	{name: "error envelope", body: errorEnvelopeBody},
	{name: "empty choices", body: `{"id":"chatcmpl-2","object":"chat.completion","choices":[]}`},
}

// sendAgainstBody answers /v1/chat/completions with body under HTTP 200 and
// returns what sendLocalChatCompletion hands its caller.
func sendAgainstBody(t *testing.T, body string) (*localChatCompletionResponse, error) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)

	_, port, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatalf("test server URL %q: %v", server.URL, err)
	}
	return sendLocalChatCompletion(context.Background(), port, "MoM", "What is 2 + 2?", 5*time.Second)
}

// TestSendLocalChatCompletionReportsAnErrorBodyServedWith200 is the ask.
//
// sendLocalChatCompletion is the one path every chat-completion test goes
// through. It reads the body (chat_completion_helpers.go:60) and carries it
// back (chat_completion_helpers.go:68), but never looks at it, so it returns a
// nil error for a body that is not a completion. Its callers then decide the
// turn succeeded from the status alone: entrypoint_recipes.go:86 and :121,
// session_pricing_e2e.go:63, keyword_routing.go:148, entropy_routing.go:174,
// jailbreak_detection.go:139 and the rest.
//
// A status is not evidence of a body. An error envelope served with 200 is
// handed back as a success, and so is a completion carrying no choices, so a
// turn that produced no answer is counted as healthy. The helper must instead
// report that failure to its caller: a completion has no error member and at
// least one choices entry.
//
// response_api_basic.go:81 is the precedent. It is the one place that asks the
// body what happened, by requiring a completed or in_progress status.
func TestSendLocalChatCompletionReportsAnErrorBodyServedWith200(t *testing.T) {
	for _, testCase := range bodiesServedWith200 {
		t.Run(testCase.name, func(t *testing.T) {
			response, err := sendAgainstBody(t, testCase.body)
			if (err != nil) == testCase.isACompletedTurn {
				t.Errorf("sendLocalChatCompletion returned err=%v for a %s: %s",
					err, testCase.name, response.Body)
			}
		})
	}
}

// TestErrorEnvelopeUnder200IsHandedBackAsASuccessToday pins the present
// behaviour, so a change to what the helper reports shows up in the diff
// rather than only in the test above.
func TestErrorEnvelopeUnder200IsHandedBackAsASuccessToday(t *testing.T) {
	response, err := sendAgainstBody(t, errorEnvelopeBody)
	if err != nil {
		t.Fatalf("the recorded behaviour has changed: %v", err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("the recorded behaviour has changed: status %d", response.StatusCode)
	}
	if !strings.Contains(string(response.Body), `"error"`) {
		t.Fatalf("the error envelope must reach the caller, got %s", response.Body)
	}
}
