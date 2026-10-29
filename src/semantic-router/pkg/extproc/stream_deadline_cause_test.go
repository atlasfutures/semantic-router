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
// Nothing makes the compiler notice a cut added to routerStreamCutCodes
// without an entry in routerStreamCutCauses. This test is that link: it holds
// the two maps in step, then drives the real path with each code the Router
// cuts a stream with and fails if any of them reaches a client unexplained.
func TestEveryRouterCutNamesItsCauseToTheClient(t *testing.T) {
	for code := range routerStreamCutCodes {
		if _, mapped := routerStreamCutCauses[code]; !mapped {
			t.Fatalf("the Router cuts a stream with %q and routerStreamCutCauses states no cause for it", code)
		}
	}
	if _, mapped := routerStreamCutCauses["standard_resend_body_cut"]; !mapped {
		t.Fatal("the body cut states no cause")
	}

	for code, cause := range routerStreamCutCauses {
		router := &OpenAIRouter{}
		ctx := overrunningStreamContext(t)
		ctx.SourceFormat = llmprotocol.OpenAIChatV1
		ctx.SemanticRequest = streamUsageRequest()
		router.initializeSemanticResponseStream(ctx)

		buffers := semanticStreamBuffers{}
		buffers.push([]byte(endlessUpstreamChunk(1)), ctx)
		cut := llmprotocol.NewError(llmprotocol.ErrorUpstreamTimeout, code, "the router ended this stream", nil)
		buffers.recordError(ctx, cut, true)
		router.attachTruncatedStreamUsage(ctx, cut)
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
