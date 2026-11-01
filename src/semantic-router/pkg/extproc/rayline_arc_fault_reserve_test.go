//go:build !windows && cgo

package extproc

import (
	"strings"
	"testing"
	"time"

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

func reserveFaultContext(format llmprotocol.WireFormat) *RequestContext {
	return &RequestContext{
		InjectedFault: headers.FaultRefusalPreludeAll, TargetFormat: format,
		RefusalReserve: &refusalReservePlan{reserved: true},
	}
}

// CRLF framing and data split across lines are read as an SSE reader reads
// them, so their content is dropped too.
func TestReserveFaultReadsCRLFAndMultiLineData(t *testing.T) {
	ctx := reserveFaultContext(llmprotocol.OpenAIChatV1)
	crlf := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"secret-one\"},\"finish_reason\":null}]}\r\n\r\n"
	multi := "data: {\"choices\":[{\"index\":0,\r\ndata: \"delta\":{\"content\":\"secret-two\"},\"finish_reason\":\"stop\"}]}\r\n\r\n"
	out := string(faultReservedChunk(ctx, []byte(crlf+multi[:30])))
	out += string(faultReservedChunk(ctx, []byte(multi[30:])))
	assert.NotContains(t, out, "secret")
	assert.Equal(t, 2, strings.Count(out, "\n\n"), "an event was lost: %q", out)
	assert.Contains(t, out, `"finish_reason":"content_filter"`)
}

// A Messages stream's kept thinking blocks are renumbered from zero once the
// text blocks before them are dropped.
func TestReserveFaultRenumbersKeptMessagesBlocks(t *testing.T) {
	ctx := reserveFaultContext(llmprotocol.AnthropicMessagesV1)
	stream := "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"secret\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":9}}\n\n"
	out := string(faultReservedChunk(ctx, []byte(stream)))
	assert.NotContains(t, out, "secret")
	assert.NotContains(t, out, `"index":1`)
	assert.Equal(t, 3, strings.Count(out, `"index":0`), "the thinking block was not renumbered: %s", out)
	assert.Contains(t, out, `"stop_reason":"refusal"`)
	assert.Contains(t, out, `"output_tokens":9`)
}

// Dropped content still counts as the call's activity for the silence clock.
func TestReserveFaultDroppedContentKeepsTheCallAlive(t *testing.T) {
	ctx := reserveFaultContext(llmprotocol.OpenAIChatV1)
	ctx.StreamContentSince = time.Now().Add(-time.Minute)
	faultReservedChunk(ctx, []byte(`data: {"choices":[{"index":0,"delta":{"content":"secret"},"finish_reason":null}]}`+"\n\n"))
	assert.WithinDuration(t, time.Now(), ctx.StreamContentSince, time.Second)
	assert.False(t, ctx.StreamContentSeen, "dropped content was counted as content the client saw")
}

// A last event the stream ended without terminating is still rewritten.
func TestReserveFaultFlushesAnUnterminatedLastEvent(t *testing.T) {
	ctx := reserveFaultContext(llmprotocol.OpenAIChatV1)
	assert.Contains(t, string(faultReservedChunk(ctx, []byte(`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]"))), "content_filter")
	assert.Equal(t, "data: [DONE]\n\n", string(flushReservedFault(ctx)))
	assert.Empty(t, flushReservedFault(ctx))
}
