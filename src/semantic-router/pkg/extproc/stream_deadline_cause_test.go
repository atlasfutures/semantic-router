package extproc

import (
	"bytes"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// Every cut the Router makes names why it was made.
//
// A cut turn ends on finish_reason "length", because this format's vocabulary
// is closed and holds nothing for a turn infrastructure ended. Read literally
// that says the model exhausted its output budget: true enough of a deadline,
// wrong for a model that went quiet or a connection that broke. So the cause
// travels beside the stop, and a caller choosing between retrying and asking
// for less work reads that instead of inferring it from the stop.
//
// The codes and the causes live in different packages, so nothing makes the
// compiler notice a cut added without a cause beside it. This test is that
// link: it drives the real path with each code the Router cuts a stream with
// and fails if any of them reaches a client unexplained. A cut added to
// routerStreamCutCodes without a case in raylineTruncationCause fails here.
func TestEveryRouterCutNamesItsCauseToTheClient(t *testing.T) {
	// Each code the Router ends a stream with, and the mechanism a client is
	// told. Keep in step with routerStreamCutCodes and with the body cut.
	causes := map[string]string{
		"stream_truncated":            "deadline",
		streamIdleTimeoutCode:         "silence",
		streamFirstContentTimeoutCode: "silence",
		"standard_resend_body_cut":    "transport",
	}
	for code := range routerStreamCutCodes {
		if _, mapped := causes[code]; !mapped {
			t.Fatalf(
				"the Router cuts a stream with %q and this test states no cause for it; "+
					"add the case to raylineTruncationCause and the expectation here",
				code,
			)
		}
	}

	for code, cause := range causes {
		router := &OpenAIRouter{}
		ctx := overrunningStreamContext(t)
		ctx.SourceFormat = llmprotocol.OpenAIChatV1
		ctx.SemanticRequest = streamUsageRequest()
		router.initializeSemanticResponseStream(ctx)

		buffers := semanticStreamBuffers{}
		buffers.push([]byte(endlessUpstreamChunk(1)), ctx)
		buffers.recordError(ctx, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamTimeout, code, "the router ended this stream", nil,
		), true)
		router.attachTruncatedStreamUsage(ctx)
		buffers.finalize(ctx)

		want := []byte(`"rayline":{"truncated":"` + cause + `"}`)
		if !bytes.Contains(buffers.translated, want) {
			t.Fatalf("a %s cut did not tell the client why:\n%s", code, buffers.translated)
		}
		if !bytes.Contains(buffers.translated, []byte("data: [DONE]")) {
			t.Fatalf("a %s cut left the stream open:\n%s", code, buffers.translated)
		}
		if !bytes.Contains(buffers.translated, []byte(`"completion_tokens":4`)) {
			t.Fatalf("a %s cut billed nothing:\n%s", code, buffers.translated)
		}
	}
}
