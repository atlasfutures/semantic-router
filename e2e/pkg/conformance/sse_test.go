package conformance

import (
	"strings"
	"testing"
)

func TestParseSSE(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want []SSEEvent
	}{
		{
			name: "named events with data",
			raw:  "event: message_start\ndata: {\"a\":1}\n\nevent: message_stop\ndata: {}\n\n",
			want: []SSEEvent{
				{Name: "message_start", Data: `{"a":1}`},
				{Name: "message_stop", Data: `{}`},
			},
		},
		{
			name: "an unnamed event keeps the implied default type",
			raw:  "data: {\"chunk\":1}\n\ndata: [DONE]\n\n",
			want: []SSEEvent{{Data: `{"chunk":1}`}, {Data: SSEDoneData}},
		},
		{
			name: "multi-line data joins with a newline",
			raw:  "data: line one\ndata: line two\n\n",
			want: []SSEEvent{{Data: "line one\nline two"}},
		},
		{
			name: "comments and keepalives are dropped",
			raw:  ": keepalive\ndata: {}\n\n: ping\n\n",
			want: []SSEEvent{{Data: `{}`}},
		},
		{
			name: "CRLF framing parses the same as LF",
			raw:  "event: ping\r\ndata: {}\r\n\r\n",
			want: []SSEEvent{{Name: "ping", Data: `{}`}},
		},
		{
			name: "only one leading space after the colon is stripped",
			raw:  "data:  padded\n\n",
			want: []SSEEvent{{Data: " padded"}},
		},
		{
			name: "a bare field name carries an empty value",
			raw:  "data\n\n",
			want: []SSEEvent{{Data: ""}},
		},
		{
			name: "an id field is captured",
			raw:  "id: 7\nevent: ping\ndata: {}\n\n",
			want: []SSEEvent{{Name: "ping", Data: `{}`, ID: "7"}},
		},
		{
			name: "retry is ignored",
			raw:  "retry: 500\ndata: {}\n\n",
			want: []SSEEvent{{Data: `{}`}},
		},
		{
			name: "a truncated final event is still dispatched",
			raw:  "event: content_block_delta\ndata: {\"text\":\"hi\"",
			want: []SSEEvent{{Name: "content_block_delta", Data: `{"text":"hi"`}},
		},
		{
			name: "an empty stream yields no events",
			raw:  "",
			want: nil,
		},
		{
			name: "trailing blank lines do not dispatch an empty event",
			raw:  "data: {}\n\n\n\n",
			want: []SSEEvent{{Data: `{}`}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseSSE([]byte(tt.raw))
			if err != nil {
				t.Fatalf("ParseSSE() error = %v", err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("ParseSSE() = %d events %+v, want %d", len(got), got, len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("event[%d] = %+v, want %+v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestParseSSERejectsUnknownField(t *testing.T) {
	if _, err := ParseSSE([]byte("evt: unknown\ndata: {}\n\n")); err == nil ||
		!strings.Contains(err.Error(), `unknown field "evt"`) {
		t.Fatalf("ParseSSE() error = %v, want an unknown-field error", err)
	}
}

func TestSSEEventIsJSON(t *testing.T) {
	tests := []struct {
		data string
		want bool
	}{
		{data: `{"a":1}`, want: true},
		{data: `[1,2]`, want: true},
		{data: SSEDoneData, want: false},
		{data: "", want: false},
		{data: "plain text", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.data, func(t *testing.T) {
			if got := (SSEEvent{Data: tt.data}).IsJSON(); got != tt.want {
				t.Errorf("IsJSON(%q) = %v, want %v", tt.data, got, tt.want)
			}
		})
	}
}
