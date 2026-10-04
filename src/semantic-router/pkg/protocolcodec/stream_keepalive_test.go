package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const (
	anthropicPingFrame      = "event: ping\ndata: {\"type\":\"ping\"}\n\n"
	openRouterCommentFrame  = ": OPENROUTER PROCESSING\n\n"
	targetKeepaliveComment  = ": keepalive\n\n"
	keepaliveBeforeAnything = -1
)

// keepaliveSource is a provider stream, one SSE frame per element, and where
// its provider sends keepalives: after the frame at each index, or before the
// first frame for keepaliveBeforeAnything. An index may repeat.
type keepaliveSource struct {
	fixture   string
	keepalive string
	after     []int
}

var keepaliveSources = map[llmprotocol.WireFormat]keepaliveSource{
	// Anthropic pings once message_start is out, here also twice in a row
	// between deltas.
	llmprotocol.AnthropicMessagesV1: {
		fixture: "003-anthropic-text-in.json", keepalive: anthropicPingFrame,
		after: []int{0, 2, 2, 4},
	},
	// OpenRouter comments while the request is queued and between chunks.
	llmprotocol.OpenAIChatV1: {
		fixture: "001-chat-text-in.json", keepalive: openRouterCommentFrame,
		after: []int{keepaliveBeforeAnything, keepaliveBeforeAnything, 0, 2},
	},
	llmprotocol.OpenAIResponsesV1: {
		fixture: "002-responses-text-in.json", keepalive: openRouterCommentFrame,
		after: []int{keepaliveBeforeAnything, 1, 3, 3},
	},
}

func keepaliveSourceFrames(t *testing.T, fixture string) []string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "golden", "stream", fixture))
	if err != nil {
		t.Fatal(err)
	}
	var input goldenStreamInput
	if err := json.Unmarshal(body, &input); err != nil {
		t.Fatal(err)
	}
	var frames []string
	for _, frame := range strings.SplitAfter(strings.Join(input.Chunks, ""), "\n\n") {
		if frame != "" {
			frames = append(frames, frame)
		}
	}
	return frames
}

// withKeepalives interleaves a source's keepalives with its frames. A
// keepalive's element is marked true.
func withKeepalives(source keepaliveSource, frames []string) ([]string, []bool) {
	var chunks []string
	var keepalive []bool
	add := func(after int) {
		for _, index := range source.after {
			if index == after {
				chunks, keepalive = append(chunks, source.keepalive), append(keepalive, true)
			}
		}
	}
	add(keepaliveBeforeAnything)
	for index, frame := range frames {
		chunks, keepalive = append(chunks, frame), append(keepalive, false)
		add(index)
	}
	return chunks, keepalive
}

type keepaliveRun struct {
	pushed     [][][]byte
	final      [][]byte
	completion *llmprotocol.Event
}

func runKeepaliveStream(t *testing.T, source, target llmprotocol.WireFormat, chunks []string) keepaliveRun {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	var run keepaliveRun
	for _, chunk := range chunks {
		frames, _, _, err := stream.Push([]byte(chunk))
		if err != nil {
			t.Fatalf("push %q: %v", chunk, err)
		}
		run.pushed = append(run.pushed, frames)
	}
	frames, events, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	run.final = frames
	for index := range events {
		if events[index].Type == llmprotocol.EventResponseCompleted {
			run.completion = &events[index]
		}
	}
	if run.completion == nil {
		t.Fatal("the stream did not complete")
	}
	return run
}

func isTargetKeepalive(frame []byte) bool {
	return string(frame) == targetKeepaliveComment || string(frame) == anthropicPingFrame
}

// Each upstream keepalive, from every source format, reaches every target as
// exactly one keepalive in the target's own form, at the point it arrived. The
// content frames are byte for byte those of the same stream without
// keepalives, and the turn's usage and stop are unchanged. Control: a stream
// without keepalives carries none.
func TestEachUpstreamKeepaliveReachesTheClientOnce(t *testing.T) {
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"/"+string(target), func(t *testing.T) {
				spec := keepaliveSources[source]
				frames := keepaliveSourceFrames(t, spec.fixture)
				plain := runKeepaliveStream(t, source, target, frames)
				for _, pushed := range append(plain.pushed, plain.final) {
					for _, frame := range pushed {
						if isTargetKeepalive(frame) {
							t.Fatalf("a stream without keepalives carried one: %q", frame)
						}
					}
				}

				chunks, keepalive := withKeepalives(spec, frames)
				if len(chunks) != len(frames)+len(spec.after) {
					t.Fatalf("fixture %s has %d frames; a keepalive position is out of range", spec.fixture, len(frames))
				}
				kept := runKeepaliveStream(t, source, target, chunks)
				content := 0
				for index, pushed := range kept.pushed {
					if !keepalive[index] {
						if !reflect.DeepEqual(pushed, plain.pushed[content]) {
							t.Fatalf("frame %d changed:\n got %q\nwant %q", content, pushed, plain.pushed[content])
						}
						content++
						continue
					}
					// Anthropic pings after message_start; before it a
					// Messages client hears a comment, like the others.
					want := targetKeepaliveComment
					if target == llmprotocol.AnthropicMessagesV1 && content > 0 {
						want = anthropicPingFrame
					}
					if len(pushed) != 1 || string(pushed[0]) != want {
						t.Fatalf("upstream keepalive %d became %q, want one %q", index, pushed, want)
					}
				}
				if !reflect.DeepEqual(kept.final, plain.final) {
					t.Fatalf("terminal frames changed:\n got %q\nwant %q", kept.final, plain.final)
				}
				if kept.completion.StopReason != plain.completion.StopReason ||
					!reflect.DeepEqual(kept.completion.Usage, plain.completion.Usage) {
					t.Fatalf("completion changed: stop %q usage %+v, want stop %q usage %+v",
						kept.completion.StopReason, kept.completion.Usage, plain.completion.StopReason, plain.completion.Usage)
				}
			})
		}
	}
}

// A ping after message_stop is tolerated as before, with its diagnostic, and
// forwards nothing; nor does a comment after the terminal.
func TestNoKeepaliveFollowsTheTerminal(t *testing.T) {
	trailers := map[llmprotocol.WireFormat][]string{
		llmprotocol.AnthropicMessagesV1: {anthropicPingFrame, openRouterCommentFrame},
		llmprotocol.OpenAIChatV1:        {openRouterCommentFrame},
		llmprotocol.OpenAIResponsesV1:   {openRouterCommentFrame},
	}
	for _, source := range builtinFormats {
		for _, trailer := range trailers[source] {
			for _, target := range builtinFormats {
				frames := keepaliveSourceFrames(t, keepaliveSources[source].fixture)
				run := runKeepaliveStream(t, source, target, append(frames, trailer))
				if pushed := run.pushed[len(run.pushed)-1]; len(pushed) != 0 {
					t.Fatalf("%s -> %s: %q after the terminal forwarded %q", source, target, trailer, pushed)
				}
				for _, frame := range run.final {
					if isTargetKeepalive(frame) {
						t.Fatalf("%s -> %s: a keepalive followed the terminal", source, target)
					}
				}
			}
		}
	}
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	_, events, diagnostics, err := stream.Push([]byte(anthropicCompleteToolUseStream + anthropicPingFrame))
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range events {
		if event.Type == llmprotocol.EventKeepalive {
			t.Fatal("a ping after message_stop was decoded as a keepalive")
		}
	}
	if len(diagnostics) == 0 || diagnostics[len(diagnostics)-1].Field != "stream.after_message_stop" {
		t.Fatalf("diagnostics = %+v, want the after-message_stop drop", diagnostics)
	}
}

// A comment-only first frame is a keepalive, with or without a leading BOM,
// and the stream carries on; a BOM on a later frame is still refused.
func TestACommentOnlyFirstFrameIsAKeepalive(t *testing.T) {
	for _, source := range builtinFormats {
		frames := keepaliveSourceFrames(t, keepaliveSources[source].fixture)
		for _, first := range []string{openRouterCommentFrame, "\xef\xbb\xbf" + openRouterCommentFrame} {
			run := runKeepaliveStream(t, source, llmprotocol.OpenAIChatV1, append([]string{first}, frames...))
			if len(run.pushed[0]) != 1 || string(run.pushed[0][0]) != targetKeepaliveComment {
				t.Fatalf("%s: first frame %q became %q", source, first, run.pushed[0])
			}
		}
		stream, err := NewBuiltinEngine().NewStream(source, llmprotocol.OpenAIChatV1,
			llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
		if err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := stream.Push([]byte(openRouterCommentFrame)); err != nil {
			t.Fatal(err)
		}
		_, _, _, err = stream.Push([]byte("\xef\xbb\xbf" + openRouterCommentFrame))
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "unexpected_stream_bom")
	}
}

// A frame with any field line is not a keepalive, even beside a comment.
func TestACommentBesideAFieldIsNotAKeepalive(t *testing.T) {
	frame, err := parseSSEFrame([]byte(": note\nevent: message_start\n\n"), 1<<10)
	if err != nil {
		t.Fatal(err)
	}
	if frame.commentOnly() {
		t.Fatal("a frame with an event line was read as comment-only")
	}
	if frame, _ := parseSSEFrame([]byte(": one\n: two\n\n"), 1<<10); !frame.commentOnly() || bytes.Contains(frame.Data, []byte("one")) {
		t.Fatalf("frame of comments = %+v, want comment-only", frame)
	}
}

// A buffered caller that decodes a whole stream gets the same response with
// or without keepalives: the accumulator ignores them.
func TestKeepalivesLeaveADecodedStreamUnchanged(t *testing.T) {
	for _, source := range builtinFormats {
		spec := keepaliveSources[source]
		frames := keepaliveSourceFrames(t, spec.fixture)
		chunks, _ := withKeepalives(spec, frames)
		decode := func(body string) llmprotocol.Response {
			response, _, err := NewBuiltinEngine().DecodeResponseStream(source, []byte(body),
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
			if err != nil {
				t.Fatalf("%s: %v", source, err)
			}
			response.CreatedAt = time.Time{}
			return response
		}
		if plain, kept := decode(strings.Join(frames, "")), decode(strings.Join(chunks, "")); !reflect.DeepEqual(plain, kept) {
			t.Fatalf("%s: keepalives changed the decoded response:\n got %+v\nwant %+v", source, kept, plain)
		}
	}
}

// Keepalives count against the event limit, so a flood of them is bounded
// like any other frame.
func TestKeepalivesCountAgainstTheEventLimit(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.Events = 3
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = stream.Push([]byte(strings.Repeat(openRouterCommentFrame, 4)))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_event_limit")
}
