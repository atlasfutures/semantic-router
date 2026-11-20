package extproc

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"
)

func TestResentLastEventNamesHowTheStreamEnded(t *testing.T) {
	chunk := `data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}` + "\n\n"
	for name, tc := range map[string]struct{ body, want string }{
		"done":         {chunk + "data: [DONE]\n\n", "done"},
		"crlf done":    {"data: {\"choices\":[]}\r\n\r\ndata: [DONE]\r\n\r\n", "done"},
		"finish":       {chunk + `data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n", "finish"},
		"usage":        {chunk + `data: {"choices":[],"usage":{"completion_tokens":5}}` + "\n\n", "usage"},
		"mid-stream":   {chunk + chunk, "chunk"},
		"error":        {chunk + `data: {"error":{"message":"overloaded"}}` + "\n\n", "error"},
		"typed":        {"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n", "message_stop"},
		"responses":    {`data: {"type":"response.completed","response":{}}` + "\n\n", "response.completed"},
		"odd type":     {`data: {"type":"Weird Type!"}` + "\n\n", "other"},
		"unparsed":     {chunk + "data: {\"choices\":[{\"del\n\n", "unparsed"},
		"no event":     {": keep-alive\n\n", "none"},
		"empty":        {"", "none"},
		"partial line": {chunk + `data: {"choices":[{"index":0,"delta":{"content":"h`, "unparsed"},
	} {
		if got := resentLastEvent([]byte(tc.body)); got != tc.want {
			t.Errorf("%s: last event %q, want %q", name, got, tc.want)
		}
	}
}

func TestResentReadErrorClassesTheCut(t *testing.T) {
	for want, err := range map[string]error{
		"":               nil,
		"deadline":       fmt.Errorf("read: %w", context.DeadlineExceeded),
		"canceled":       context.Canceled,
		"unexpected_eof": io.ErrUnexpectedEOF,
		"other":          errors.New("connection reset by peer"),
	} {
		if got := resentReadError(err); got != want {
			t.Errorf("%v: class %q, want %q", err, got, want)
		}
	}
}

// A re-send that never answered has no body to describe.
func TestUndeliveredResendHasNoBodyFacts(t *testing.T) {
	if facts := newResentBodyFacts(flexTwinFailure(context.DeadlineExceeded), true); facts != nil {
		t.Fatalf("an undelivered re-send was described: %+v", facts)
	}
}
