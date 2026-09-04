package extproc

import (
	"bytes"
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A turn that outruns the platform is cut by the platform, and the Router
// never gets to speak. Measured on the dev cell 2026-09-03: three thinking-arm
// turns reached the client as message_start, content_block_start, deltas, then
// a clean close. Closing the stream honestly at that point is impossible --
// the ext_proc stream is already torn down and the closing frames have nowhere
// to go.
//
// So the Router needs its own deadline, below the platform's, and it has to
// end the turn while the connection to the client is still open. The cell's
// ladder is ext_proc message_timeout 610 s, Envoy stream_idle_timeout 620 s,
// Cloud Run 630 s.
func TestRouterEndsAStreamAtItsOwnDeadline(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)

	// An upstream that keeps streaming and never terminates.
	body := streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(1))

	if !bytes.Contains(body, []byte("content_block_stop")) {
		t.Fatalf("the block the client is holding was left open:\n%s", body)
	}
	if !bytes.Contains(body, []byte(`"type":"error"`)) {
		t.Fatalf("the turn ended without naming the failure:\n%s", body)
	}
	// The Anthropic error object holds a type and a message and has no slot
	// for a code, so this leg names the failure by its vendor type.
	if !bytes.Contains(body, []byte(`"type":"timeout_error"`)) {
		t.Fatalf("the failure was not named as a timeout:\n%s", body)
	}
	// message_stop is the terminal of a message that finished. This one did
	// not, and the repo already holds that invariant for a cut stream.
	if bytes.Contains(body, []byte("message_stop")) {
		t.Fatalf("a truncated turn published a success terminal:\n%s", body)
	}
}

// The Chat error object does have a code, and it is where the Router's own
// name for this failure travels: the turn was truncated, by the Router, at its
// own deadline.
func TestTruncatedChatStreamNamesTheCode(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.OpenAIChatV1

	body := streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(1))

	if !bytes.Contains(body, []byte(`"code":"stream_truncated"`)) {
		t.Fatalf("the failure did not say the stream was truncated:\n%s", body)
	}
	if bytes.Contains(body, []byte("data: [DONE]")) {
		t.Fatalf("a truncated turn published a success sentinel:\n%s", body)
	}
}

// Once the Router has ended the turn, whatever the upstream keeps sending is
// no longer the client's. Anything after the error frame would arrive on a
// message the client has been told is over.
func TestNothingFollowsTheTruncationFrame(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(1))

	trailing := streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(2))
	if len(trailing) != 0 {
		t.Fatalf("the upstream kept reaching the client after the turn ended:\n%s", trailing)
	}
}

// The turn still cost what it cost. A truncated turn that carried counts is
// reported with them and marked truncated, the same as one the platform cut.
func TestTurnEndedAtTheRouterDeadlineIsCounted(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.StartTime = time.Now()

	// Inside the budget the turn streams normally, usage included.
	streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(1))
	streamDeadlineResponseBody(t, router, ctx, upstreamUsageChunk())

	ctx.StartTime = time.Now().Add(-600 * time.Second)
	streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(2))

	fields := findLogEvent(t, logs, "llm_usage")
	if truncated, _ := fields["truncated"].(bool); !truncated {
		t.Fatalf("a turn the Router cut was counted as a whole one: %v", fields)
	}
	if got, _ := fields["completion_tokens"].(int64); got != 16030 {
		t.Fatalf("completion_tokens = %v, want the tokens the stream carried", fields["completion_tokens"])
	}
}

// A stream inside the budget is untouched.
func TestStreamInsideTheBudgetIsNotCut(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.StartTime = time.Now()

	body := streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(1))
	if bytes.Contains(body, []byte(`"type":"error"`)) {
		t.Fatalf("a turn inside the budget was cut:\n%s", body)
	}
}

// The Router's deadline only works if it fires before the ext_proc stream is
// torn down. Above that the closing frames have nowhere to go, which is the
// whole failure.
func TestRouterDeadlineSitsBelowThePlatformLadder(t *testing.T) {
	if defaultResponseStreamDeadlineSeconds >= extProcMessageTimeoutSeconds {
		t.Fatalf(
			"the Router deadline %d s is not below the ext_proc message timeout %d s",
			defaultResponseStreamDeadlineSeconds, extProcMessageTimeoutSeconds,
		)
	}
}

func overrunningStreamContext(t *testing.T) *RequestContext {
	t.Helper()
	return &RequestContext{
		RequestID:           "rt_70472e24-000",
		RequestModel:        "xiaomi/mimo-v2.5-pro@thinking-on",
		SourceFormat:        llmprotocol.AnthropicMessagesV1,
		TargetFormat:        llmprotocol.OpenAIChatV1,
		IsStreamingResponse: true,
		TraceContext:        t.Context(),
		// Past the Router's own deadline and short of the platform's, which is
		// the window this exists to use.
		StartTime: time.Now().Add(-600 * time.Second),
	}
}

func streamDeadlineResponseBody(
	t *testing.T,
	router *OpenAIRouter,
	ctx *RequestContext,
	chunk string,
) []byte {
	t.Helper()
	response := router.handleSemanticStreamingResponseBody([]byte(chunk), false, ctx)
	return responseBodyMutationBytes(response)
}

func responseBodyMutationBytes(response *ext_proc.ProcessingResponse) []byte {
	return response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
}

// endlessUpstreamChunk is one delta from an upstream that never terminates.
func endlessUpstreamChunk(sequence int) string {
	return `data: {"id":"gen-1","object":"chat.completion.chunk","created":1788474769,` +
		`"model":"xiaomi/mimo-v2.5-pro","choices":[{"index":0,` +
		`"delta":{"role":"assistant","content":"` + strings.Repeat("still thinking ", sequence) +
		`"},"finish_reason":null}]}` + "\n\n"
}

func upstreamUsageChunk() string {
	return `data: {"id":"gen-1","object":"chat.completion.chunk","created":1788474769,` +
		`"model":"xiaomi/mimo-v2.5-pro","choices":[],` +
		`"usage":{"prompt_tokens":220,"completion_tokens":16030,"total_tokens":16250}}` + "\n\n"
}
