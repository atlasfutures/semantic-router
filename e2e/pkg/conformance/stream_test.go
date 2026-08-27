package conformance

import (
	"strings"
	"testing"
)

func TestCompareStreams(t *testing.T) {
	const want = "event: message_start\n" +
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\"}}\n" +
		"\n" +
		"event: content_block_delta\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n" +
		"\n" +
		"event: message_stop\n" +
		"data: {\"type\":\"message_stop\"}\n" +
		"\n"

	tests := []struct {
		name      string
		cmp       Comparison
		got       string
		wantPaths []string
	}{
		{
			name: "reframed stream with CRLF, comments, and no final newline still matches",
			cmp:  Comparison{Mode: ModeSemantic, Volatile: []string{"/message/id"}},
			got: ": keepalive\r\n" +
				"event: message_start\r\n" +
				"data: {\"message\":{\"id\":\"msg_01H\"},\"type\":\"message_start\"}\r\n" +
				"\r\n" +
				"event: content_block_delta\r\n" +
				"data: {\"delta\":{\"text\":\"hi\"},\"type\":\"content_block_delta\"}\r\n" +
				"\r\n" +
				": ping\r\n" +
				"event: message_stop\r\n" +
				"data: {\"type\":\"message_stop\"}",
		},
		{
			name: "a truncated stream is not a success",
			cmp:  Comparison{Mode: ModeSemantic, Volatile: []string{"/message/id"}},
			got: "event: message_start\n" +
				"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01H\"}}\n" +
				"\n" +
				"event: content_block_delta\n" +
				"data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\n",
			wantPaths: []string{"events"},
		},
		{
			name: "a reordered stream fails on the event type, not the payload",
			cmp:  Comparison{Mode: ModeSemantic, Volatile: []string{"/message/id"}},
			got: "event: content_block_delta\n" +
				"data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n" +
				"\n" +
				"event: message_start\n" +
				"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01H\"}}\n" +
				"\n" +
				"event: message_stop\n" +
				"data: {\"type\":\"message_stop\"}\n\n",
			wantPaths: []string{"event[0].event", "event[1].event"},
		},
		{
			name:      "a changed delta fails with its event coordinate and pointer",
			cmp:       Comparison{Mode: ModeSemantic, Volatile: []string{"/message/id"}},
			got:       strings.Replace(want, `"text":"hi"`, `"text":"bye"`, 1),
			wantPaths: []string{"event[1].data/delta/text"},
		},
		{
			name:      "exact-except compares the terminal sentinel literally",
			cmp:       Comparison{Mode: ModeExactExcept},
			got:       want,
			wantPaths: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Compare(tt.cmp,
				Payload{Body: []byte(want), Stream: true},
				Payload{Body: []byte(tt.got), Stream: true})
			if err != nil {
				t.Fatalf("Compare() error = %v", err)
			}

			got := make([]string, 0, len(result.Mismatches))
			for _, m := range result.Mismatches {
				got = append(got, m.Path)
			}
			if !equalStrings(got, tt.wantPaths) {
				t.Fatalf("mismatch paths = %v, want %v\n%v", got, tt.wantPaths, result.Err())
			}
		})
	}
}

// TestCompareOpenAIDoneSentinel pins that the "[DONE]" literal is compared as text
// rather than parsed as JSON, so a stream that leaks or loses it fails loudly.
func TestCompareOpenAIDoneSentinel(t *testing.T) {
	const want = "data: {\"id\":\"chunk\"}\n\ndata: [DONE]\n\n"

	result, err := Compare(Comparison{Mode: ModeSemantic},
		Payload{Body: []byte(want), Stream: true},
		Payload{Body: []byte("data: {\"id\":\"chunk\"}\n\ndata: [done]\n\n"), Stream: true})
	if err != nil {
		t.Fatalf("Compare() error = %v", err)
	}
	if result.Pass() {
		t.Fatal("a changed terminal sentinel passed")
	}
	if got := result.Mismatches[0].Path; got != "event[1].data" {
		t.Errorf("mismatch path = %q, want %q", got, "event[1].data")
	}
}
