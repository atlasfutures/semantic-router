//go:build !windows && cgo

package extproc

import (
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// refusal_prelude_all refuses the first attempt and the re-served one: the
// turn is re-served once, the re-served arm's answer never reaches the
// client, its refusal passes through, and both attempts are recorded and
// settled on their own usage.
func TestRefusalPreludeAllExhaustsTheReserve(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	ctx.InjectedFault = headers.FaultRefusalPreludeAll

	for _, chunk := range []string{reserveRoleChunk, reserveTextChunk} {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	require.True(t, ctx.HostRetry.started(), "the fault did not re-serve")
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.NotContains(t, sent, "partial answer", "the first attempt's content reached the client")
	assert.NotContains(t, sent, "served on "+reserveGLMHost, "the re-served attempt's content reached the client")
	assert.Contains(t, sent, "content_filter", "the re-served refusal did not reach the client")
	assert.Equal(t, turnFailureRefusal, ctx.ResponseFailureClass)
	assert.Equal(t, reserveGLMArm, ctx.VSRSelectedModel)
	assert.Len(t, upstream.calls(), 1, "more than one re-serve was made")

	record := attemptsRecord(t, sent)
	attempts := record["attempts"].([]interface{})
	require.Len(t, attempts, 2)
	first, second := attempts[0].(map[string]interface{}), attempts[1].(map[string]interface{})
	assert.Equal(t, "refused", first["outcome"])
	assert.Equal(t, "fault_injection", first["signal"])
	assert.Equal(t, reserveGLMArm, second["model"])
	assert.Equal(t, "refused", second["outcome"])
	assert.NotNil(t, second["usage"], "the re-served attempt's usage was not recorded")

	lines := usageLines(logs)
	require.Len(t, lines, 2, "want one usage line per call: %v", lines)
	assertFields(t, "re-served call", lines[1], map[string]interface{}{
		"refusal_reserve_attempt": 2, "failure_class": turnFailureRefusal,
	})
}

// On a Messages arm the re-served stream's text blocks are dropped and its
// stop is made a refusal.
func TestRefusalPreludeAllRefusesAMessagesReserve(t *testing.T) {
	ctx, stream, _ := messagesReserveTurnWithFault(t, "messages_content", headers.FaultRefusalPreludeAll)

	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.NotContains(t, sent, "served on "+reserveGLMHost)
	assert.Equal(t, turnFailureRefusal, ctx.ResponseFailureClass)
	attempts := attemptsRecord(t, sent)["attempts"].([]interface{})
	require.Len(t, attempts, 2)
	assert.Equal(t, "refused", attempts[1].(map[string]interface{})["outcome"])
}

// refusal_prelude leaves the re-served attempt alone, as before.
func TestRefusalPreludeServesTheReserve(t *testing.T) {
	ctx, stream, _ := messagesReserveTurnWithFault(t, "messages_content", headers.FaultRefusalPrelude)
	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "served on "+reserveGLMHost)
	assert.Empty(t, ctx.ResponseFailureClass)
}

// The rewrite holds an event split across chunks until it is whole, and
// keeps the reasoning and usage.
func TestReserveFaultRewritesWholeEventsOnly(t *testing.T) {
	ctx := &RequestContext{
		InjectedFault: headers.FaultRefusalPreludeAll, TargetFormat: llmprotocol.OpenAIChatV1,
		RefusalReserve: &refusalReservePlan{reserved: true},
	}
	require.True(t, reserveFaultActive(ctx))
	text := `data: {"choices":[{"index":0,"delta":{"content":"secret","reasoning":"thinking"},"finish_reason":null}]}` + "\n\n"
	stop := `data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":34}}` + "\n\n"
	out := string(faultReservedChunk(ctx, []byte(text[:20])))
	assert.Empty(t, out, "a partial event was passed on")
	out += string(faultReservedChunk(ctx, []byte(text[20:]+stop+"data: [DONE]\n\n")))
	assert.NotContains(t, out, "secret")
	assert.Contains(t, out, `"reasoning":"thinking"`)
	assert.Contains(t, out, `"finish_reason":"content_filter"`)
	assert.Contains(t, out, `"completion_tokens":34`)
	assert.True(t, strings.HasSuffix(out, "data: [DONE]\n\n"))

	ctx.InjectedFault = headers.FaultRefusalPrelude
	assert.False(t, reserveFaultActive(ctx), "refusal_prelude rewrote the re-served call")
}

// The new value is inert unless the cell enables fault injection.
func TestRefusalPreludeAllNeedsFaultInjection(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		_, requestContext, algorithm := faultInjectionContext(t, enabled, headers.FaultRefusalPreludeAll)
		want := ""
		if enabled {
			want = headers.FaultRefusalPreludeAll
		}
		assert.Equal(t, want, requestedFault(algorithm.RaylineARC, requestContext), "enabled=%v", enabled)
	}
}
