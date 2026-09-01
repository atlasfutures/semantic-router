package conformance

import (
	"bufio"
	"bytes"
	"fmt"
	"strings"
)

// SSEEvent is one dispatched server-sent event.
//
// Comparators work on parsed events rather than raw bytes so that chunk boundaries,
// CRLF, a missing final newline, comments, and keepalives cannot make an otherwise
// correct stream fail.
type SSEEvent struct {
	// Name is the "event:" field. Empty means the default "message" type was implied.
	Name string
	// Data is the concatenated "data:" lines, joined with "\n" per the SSE grammar.
	Data string
	// ID is the "id:" field, if any.
	ID string
}

// SSEDoneData is the OpenAI terminal sentinel. It is a literal, not JSON.
const SSEDoneData = "[DONE]"

// IsJSON reports whether Data is a JSON payload rather than a literal sentinel.
func (e SSEEvent) IsJSON() bool {
	trimmed := strings.TrimSpace(e.Data)
	return trimmed != "" && trimmed != SSEDoneData && (trimmed[0] == '{' || trimmed[0] == '[')
}

// ParseSSE splits an SSE stream into its dispatched events.
//
// It follows the WHATWG dispatch rules that matter to protocol conformance: an event
// is dispatched on a blank line, "data:" lines accumulate, a single optional space
// after the colon is stripped, lines starting with ":" are comments, and a trailing
// unterminated event is still dispatched so a truncated stream stays observable.
func ParseSSE(raw []byte) ([]SSEEvent, error) {
	var (
		events  []SSEEvent
		current SSEEvent
		data    []string
		open    bool
	)

	dispatch := func() {
		if !open {
			return
		}
		current.Data = strings.Join(data, "\n")
		events = append(events, current)
		current, data, open = SSEEvent{}, nil, false
	}

	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")

		if line == "" {
			dispatch()
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue // comment or keepalive
		}

		field, value, found := strings.Cut(line, ":")
		if !found {
			// A bare field name carries an empty value.
			field, value = line, ""
		}
		value = strings.TrimPrefix(value, " ")

		switch field {
		case "event":
			current.Name, open = value, true
		case "data":
			data, open = append(data, value), true
		case "id":
			current.ID, open = value, true
		case "retry":
			// Reconnection timing carries no protocol semantics for these fixtures.
		default:
			return nil, fmt.Errorf("sse: unknown field %q", field)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("sse: scan: %w", err)
	}

	// A stream that ends without a blank line is truncated, not empty. Keep the
	// partial event so a truncation case can assert on what did arrive.
	dispatch()
	return events, nil
}
