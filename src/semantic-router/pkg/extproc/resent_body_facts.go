package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"regexp"
)

// resentBodyFacts describes how a re-send's body ended, for the served
// call's llm_usage line. A re-send is read whole before the response
// pipeline sees it, so without these a stream that ended early (a provider
// that closed after a few frames, a read cut by a deadline or a reset) is
// told apart from a complete one only by its token count.
type resentBodyFacts struct {
	// bytes is how much of the body was read.
	bytes int
	// lastEvent names the stream's last SSE data event (resentLastEvent);
	// empty for a body that is not a stream.
	lastEvent string
	// readError classes the error that ended the read; empty when the body
	// was read to its end.
	readError string
}

// eventTypePattern bounds a provider-stated event type that is recorded as
// is (message_stop, response.completed): anything else is "other".
var eventTypePattern = regexp.MustCompile(`^[a-z][a-z0-9_.]{0,63}$`)

func newResentBodyFacts(result flexTwinResult, stream bool) *resentBodyFacts {
	if !result.delivered {
		return nil
	}
	facts := &resentBodyFacts{bytes: len(result.body), readError: resentReadError(result.err)}
	if stream {
		facts.lastEvent = resentLastEvent(result.body)
	}
	return facts
}

// resentReadError classes the error that ended a re-send's body read.
func resentReadError(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, context.DeadlineExceeded):
		return "deadline"
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, io.ErrUnexpectedEOF):
		return "unexpected_eof"
	default:
		return "other"
	}
}

// resentLastEvent names a stream's last SSE data event: done ([DONE]),
// error (an error object), the event's own type when it states one
// (Anthropic's message_stop, a Responses response.completed), finish (a
// chat chunk that states a finish_reason), usage (a chat chunk with usage
// and no choices), chunk (any other chat chunk), unparsed (not JSON), or
// none (no data event at all).
func resentLastEvent(body []byte) string {
	var last []byte
	found := false
	for _, line := range bytes.Split(body, []byte("\n")) {
		line = bytes.TrimRight(line, "\r")
		if payload, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			last, found = bytes.TrimSpace(payload), true
		}
	}
	if !found {
		return "none"
	}
	if string(last) == "[DONE]" {
		return "done"
	}
	var event struct {
		Type    *string         `json:"type"`
		Error   json.RawMessage `json:"error"`
		Usage   json.RawMessage `json:"usage"`
		Choices []struct {
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	if json.Unmarshal(last, &event) != nil {
		return "unparsed"
	}
	if len(event.Error) > 0 && string(event.Error) != "null" {
		return "error"
	}
	if event.Type != nil {
		if eventTypePattern.MatchString(*event.Type) {
			return *event.Type
		}
		return "other"
	}
	for _, choice := range event.Choices {
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			return "finish"
		}
	}
	if len(event.Choices) == 0 && len(event.Usage) > 0 && string(event.Usage) != "null" {
		return "usage"
	}
	return "chunk"
}
