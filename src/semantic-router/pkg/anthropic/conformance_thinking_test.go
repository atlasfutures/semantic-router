package anthropic

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

// Conformance case: reasoning-block round trip through the Anthropic IR.
//
// WHY THIS TEST EXISTS
//
// A proposed migration puts this package on the request path for real client
// traffic: the client's Anthropic body would be parsed into the OpenAI IR and
// re-emitted as Anthropic on the way to the provider. That round trip is only
// safe if it preserves what the client sent.
//
// It does not preserve reasoning blocks, and the gap matters more than the
// name suggests: a `redacted_thinking` block is the carrier a gateway uses to
// ferry a provider's opaque reasoning state across turns. Drop it and the
// provider loses the thread of its own prior reasoning, silently, with a 200
// on every request.
//
// These tests pin the CURRENT behaviour rather than assert a desired one.
// They are written to fail loudly if the round trip ever starts preserving
// these blocks, because that is a change worth noticing on purpose.

const thinkingRoundTripBody = `{
  "model": "claude-sonnet-4-6",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "solve this"},
    {"role": "assistant", "content": [
      {"type": "thinking", "thinking": "step one, then step two", "signature": "sig-abc"},
      {"type": "redacted_thinking", "data": "opaque-provider-state"},
      {"type": "text", "text": "the answer is 42"}
    ]},
    {"role": "user", "content": "why?"}
  ]
}`

// The parse must not fail, and must record WHY each block vanished. A silent
// drop would leave a caller no way to detect the loss.
func TestReasoningBlocksAreDroppedButReported(t *testing.T) {
	_, ext, err := ParseAnthropicRequest([]byte(thinkingRoundTripBody))
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if ext == nil {
		t.Fatal("no IR extensions returned; the drop would be undetectable")
	}

	var reported bool
	var viaDedicatedReason bool
	for _, w := range ext.Warnings {
		if strings.Contains(w.Detail, "redacted_thinking") ||
			strings.Contains(w.Field, "content[1]") {
			reported = true
			if w.Reason == ir.ReasonRedactedThinkingDropped {
				viaDedicatedReason = true
			}
		}
	}
	if !reported {
		t.Fatalf("redacted_thinking was dropped with no warning at all; "+
			"the loss is undetectable. warnings=%+v", ext.Warnings)
	}

	// FINDING. ir.ReasonRedactedThinkingDropped exists and is documented as
	// meaning exactly this, but the parser reports the generic
	// "unsupported_block_type" instead and puts the specific cause in a free
	// text Detail. A caller filtering on the dedicated reason -- the obvious
	// way to detect this loss programmatically -- matches nothing.
	//
	// Not failing on it: the drop IS reported, so this is a usability defect
	// in the warning contract rather than a correctness bug. Logged so the
	// migration does not build detection on a constant that never fires.
	if !viaDedicatedReason {
		t.Logf("NOTE: drop reported without ir.%s; a caller filtering on that "+
			"constant would miss it. warnings=%+v",
			ir.ReasonRedactedThinkingDropped, ext.Warnings)
	}
}

// The load-bearing assertion. Parse then re-emit, exactly as a proxy on the
// request path would, and confirm the opaque reasoning state does not survive.
func TestReasoningBlocksDoNotSurviveTheRoundTrip(t *testing.T) {
	params, _, err := ParseAnthropicRequest([]byte(thinkingRoundTripBody))
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	// The passthrough is the mechanism that restores Anthropic-only fields
	// with no OpenAI representation (cache_control, top_k, images,
	// tool_result content). If reasoning blocks are ever preserved, this is
	// where it would happen, so build it the way a real caller would.
	pt, err := BuildPassthroughFromAnthropicBody([]byte(thinkingRoundTripBody))
	if err != nil {
		t.Fatalf("passthrough build failed: %v", err)
	}

	out, err := ToAnthropicRequestBodyWithPassthrough(params, pt)
	if err != nil {
		t.Fatalf("re-emit failed: %v", err)
	}
	emitted := string(out)

	// The provider's opaque state is gone. This is the migration blocker:
	// the client sent it, the provider never sees it.
	if strings.Contains(emitted, "opaque-provider-state") {
		t.Fatal("redacted_thinking data now survives the round trip. " +
			"That is an improvement, but it changes a documented contract " +
			"(inbound.go: unknown block types are warn-and-dropped) and the " +
			"migration plan assumed the loss. Update the plan, then this test.")
	}
	if strings.Contains(emitted, "redacted_thinking") {
		t.Fatal("a redacted_thinking block reappeared on the outbound body")
	}

	// The plaintext thinking block and its signature go the same way. The
	// signature is what lets a provider verify its own prior reasoning, so
	// losing it is not cosmetic either.
	if strings.Contains(emitted, "sig-abc") {
		t.Fatal("thinking signature now survives the round trip; see above")
	}

	// What DOES survive: the visible conversation. This half is the reason
	// the loss is easy to miss -- the request still looks complete.
	if !strings.Contains(emitted, "the answer is 42") {
		t.Fatalf("assistant text was lost too, which is a worse bug: %s", emitted)
	}
	if !gjson.Get(emitted, "messages").Exists() {
		t.Fatalf("no messages on the outbound body: %s", emitted)
	}
}

// A turn whose ONLY assistant content is reasoning is the sharp edge: after
// the drop there is nothing left of it. This pins what the round trip does
// with that, because an empty or absent assistant turn can break a provider's
// alternation rules rather than merely losing context.
func TestAssistantTurnOfOnlyReasoningAfterRoundTrip(t *testing.T) {
	body := `{
      "model": "claude-sonnet-4-6",
      "max_tokens": 512,
      "messages": [
        {"role": "user", "content": "think first"},
        {"role": "assistant", "content": [
          {"type": "redacted_thinking", "data": "only-content-here"}
        ]},
        {"role": "user", "content": "now answer"}
      ]
    }`

	params, _, err := ParseAnthropicRequest([]byte(body))
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	pt, err := BuildPassthroughFromAnthropicBody([]byte(body))
	if err != nil {
		t.Fatalf("passthrough build failed: %v", err)
	}
	out, err := ToAnthropicRequestBodyWithPassthrough(params, pt)
	if err != nil {
		t.Fatalf("re-emit failed: %v", err)
	}
	emitted := string(out)

	if strings.Contains(emitted, "only-content-here") {
		t.Fatal("redacted_thinking data survived; see the round-trip test")
	}

	// Record the observed shape. Both outcomes are defensible, but which one
	// happens decides whether the migration needs a repair step here, so it
	// must be pinned rather than assumed.
	roles := []string{}
	gjson.Get(emitted, "messages").ForEach(func(_, m gjson.Result) bool {
		roles = append(roles, m.Get("role").String())
		return true
	})
	t.Logf("observed outbound roles after dropping a reasoning-only turn: %v", roles)

	if len(roles) == 0 {
		t.Fatal("all messages vanished")
	}

	// FINDING, and worse than losing context.
	//
	// The assistant turn does not merely lose its reasoning -- it disappears
	// entirely, because reasoning was all it had. The outbound body is then
	// user,user: two consecutive user turns where the client sent
	// user,assistant,user.
	//
	// That is a malformed conversation for providers that require alternating
	// roles, so the failure mode is not "the model lost the thread" but a
	// provider-side 400 on a request the client formed correctly. A proxy
	// that drops a block must also decide what to do with the turn that block
	// leaves empty; today it drops the turn.
	//
	// Logged rather than failed because this test pins current behaviour. It
	// is a repair the migration has to make, not a bug in this package's own
	// stated contract.
	for i := 1; i < len(roles); i++ {
		if roles[i] == roles[i-1] {
			t.Logf("FINDING: consecutive %q turns at index %d -- the emptied "+
				"assistant turn was dropped, leaving a role sequence some "+
				"providers reject outright: %v", roles[i], i, roles)
			break
		}
	}
}
