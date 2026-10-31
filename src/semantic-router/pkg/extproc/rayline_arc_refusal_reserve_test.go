//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// A streamed turn on a held family whose model refuses after thinking and
// before any text is re-served in place, end to end through the extproc
// phases: the claude arm's call streams reasoning and then a content_filter
// stop; the cell asks again, builds the glm arm's dispatch from the client's
// body and streams a fake OpenRouter's answer into the response the client
// has open. The client sees the glm reply alone, ahead of it an SSE comment
// recording both attempts; each call writes its own usage line.

const (
	reserveClaudeArm  = "claude"
	reserveGLMArm     = "glm"
	reserveClaudeID   = "anthropic/claude-opus-5.5"
	reserveGLMID      = "z-ai/glm-5.3"
	reserveGLMHost    = "z-ai"
	reserveClaudeHost = "anthropic"
)

const reserveRoleChunk = `data: {"id":"gen-refused","object":"chat.completion.chunk","created":1,` +
	`"model":"anthropic/claude-opus-5.5","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n"

const reserveThinkingChunk = `data: {"id":"gen-refused","object":"chat.completion.chunk","created":1,` +
	`"model":"anthropic/claude-opus-5.5","choices":[{"index":0,"delta":{"reasoning":"weighing the request"},"finish_reason":null}]}` + "\n\n"

const reserveRefusalChunk = `data: {"id":"gen-refused","object":"chat.completion.chunk","created":1,` +
	`"model":"anthropic/claude-opus-5.5","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}],` +
	`"usage":{"prompt_tokens":120,"completion_tokens":7,"total_tokens":127,"cost":0.0009}}` + "\n\n" + "data: [DONE]\n\n"

const reserveTextChunk = `data: {"id":"gen-refused","object":"chat.completion.chunk","created":1,` +
	`"model":"anthropic/claude-opus-5.5","choices":[{"index":0,"delta":{"content":"partial answer"},"finish_reason":null}]}` + "\n\n"

// reserveRouter is a cell with a claude arm and a glm arm on one OpenRouter
// backend, each pinned to one host.
func reserveRouter(baseURL string) *OpenAIRouter {
	router := hostRetryRouter(baseURL, false, 0, reserveClaudeHost)
	models := router.Config.ModelConfig
	claude := models[hostRetryArm]
	claude.ExternalModelIDs = map[string]string{"vllm": reserveClaudeID}
	glm := claude
	fallbacks := true
	glm.ProviderPreferences = &config.OpenRouterProviderPreferences{
		Order: []string{reserveGLMHost}, AllowFallbacks: &fallbacks, Ignore: []string{"z-ai/flex"},
	}
	glm.ExternalModelIDs = map[string]string{"vllm": reserveGLMID}
	delete(models, hostRetryArm)
	models[reserveClaudeArm] = claude
	models[reserveGLMArm] = glm
	router.Config.DefaultModel = reserveClaudeArm
	return router
}

// startReserveTurn routes a streamed Chat turn to the claude arm, holds it
// for a re-serve that asks for the glm arm, and answers its headers.
func startReserveTurn(t *testing.T, router *OpenAIRouter) (*RequestContext, *recordingSelectionTransaction) {
	t.Helper()
	ctx, _ := routeFlexTurn(t, router, reserveClaudeArm, true)
	ctx.ProcessingStartTime = ctx.StartTime
	ctx.RaylineARCRawBody = []byte(`{"model":"auto","stream":true,"messages":[{"role":"user","content":"route this turn"}]}`)
	ctx.VSRSelectedModel = reserveClaudeArm
	ctx.RefusalReserve = &refusalReservePlan{
		maxReserves: 2, budget: 45 * time.Second, hold: 60 * time.Second, families: []string{"claude"},
		family: "claude", attemptStarted: time.Now(),
		redecide: func(*RequestContext) (*config.ModelRef, string, error) {
			return &config.ModelRef{Model: reserveGLMArm}, "test", nil
		},
	}
	if ctx.HostRetry == nil {
		ctx.HostRetry = &hostRetryPlan{attempts: []hostAttempt{{started: ctx.StartTime}}}
	}
	require.NoError(t, router.processResponseHeaders(NewMockStream(nil), flexUpstreamHeaders(http.StatusOK, "text/event-stream"), ctx))
	require.True(t, ctx.IsStreamingResponse)
	transaction := &recordingSelectionTransaction{}
	ctx.SelectionTransaction = newSelectionTransactionOwner(configRaylineARC, transaction)
	return ctx, transaction
}

// withoutAttemptsRecord is what the client was sent, the attempt record
// taken out, as the gateway strips it.
func withoutAttemptsRecord(sent string) string {
	start := strings.Index(sent, refusalAttemptsCommentPrefix)
	if start < 0 {
		return sent
	}
	end := strings.Index(sent[start:], "\n\n")
	return sent[:start] + sent[start+end+2:]
}

// attemptsRecord is the vsr-attempts comment the client was sent.
func attemptsRecord(t *testing.T, sent string) map[string]interface{} {
	t.Helper()
	start := strings.Index(sent, refusalAttemptsCommentPrefix)
	require.GreaterOrEqual(t, start, 0, "no attempt record was sent: %s", sent)
	line := sent[start+len(refusalAttemptsCommentPrefix):]
	line = line[:strings.Index(line, "\n")]
	record := map[string]interface{}{}
	require.NoError(t, json.Unmarshal([]byte(line), &record))
	return record
}

func TestRefusalAfterThinkingIsReservedOnTheNextArm(t *testing.T) {
	logs := captureLogs(t)
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, transaction := startReserveTurn(t, router)

	for _, chunk := range []string{reserveRoleChunk, reserveThinkingChunk} {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	require.True(t, router.holdingForRefusalReserve(ctx), "thinking released the hold")
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	require.True(t, ctx.HostRetry.started(), "the refusal was not re-served")
	assert.False(t, ctx.StreamingComplete)
	drainSpliced(t, router, ctx, stream)

	sent, ended := clientBytes(stream)
	assert.True(t, ended, "the re-served stream did not end the response")
	assert.Contains(t, sent, "served on "+reserveGLMHost)
	frames := withoutAttemptsRecord(sent)
	assert.NotContains(t, frames, "gen-refused", "the refused call's frames reached the client")
	assert.NotContains(t, frames, "weighing the request", "the refused call's thinking reached the client")
	assert.NotContains(t, frames, "content_filter")
	assert.Less(t, strings.Index(sent, refusalAttemptsCommentPrefix), strings.Index(sent, "data: [DONE]"),
		"the attempt record does not precede the terminal frame")
	assert.Empty(t, ctx.ResponseFailureClass)
	assert.Equal(t, 0, transaction.aborts)
	assert.Equal(t, reserveGLMArm, ctx.VSRSelectedModel)

	record := attemptsRecord(t, sent)
	assert.Equal(t, float64(1), record["v"])
	assert.Equal(t, reserveGLMArm, record["served_model"])
	attempts := record["attempts"].([]interface{})
	require.Len(t, attempts, 2)
	refused, served := attempts[0].(map[string]interface{}), attempts[1].(map[string]interface{})
	assert.Equal(t, reserveClaudeArm, refused["model"])
	assert.Equal(t, "refused", refused["outcome"])
	assert.Equal(t, "refusal", refused["failure_class"])
	assert.Equal(t, "content_filter", refused["signal"])
	assert.Equal(t, 0.0009, refused["cost_usd"])
	assert.Equal(t, "gen-refused", refused["generation_id"])
	assert.Equal(t, float64(120), refused["usage"].(map[string]interface{})["input_tokens"])
	assert.Equal(t, reserveGLMArm, served["model"])
	assert.Equal(t, "served", served["outcome"])
	assert.Nil(t, served["failure_class"])
	assert.Equal(t, 0.00004, served["cost_usd"])

	calls := upstream.calls()
	require.Len(t, calls, 1, "the re-serve was not made exactly once")
	assert.Equal(t, []string{reserveGLMHost}, providerList(t, calls[0], "order"))

	lines := usageLines(logs)
	require.Len(t, lines, 2, "want one usage line per call: %v", lines)
	assertFields(t, "refused call", lines[0], map[string]interface{}{
		"refusal_reserve": refusalReserveRefused, "refusal_reserve_attempt": 1,
		"failure_class": turnFailureRefusal, "content_sent_before_failure": false,
	})
	assertFields(t, "served call", lines[1], map[string]interface{}{
		"refusal_reserve": refusalReserveServed, "refusal_reserve_attempt": 2,
		"upstream_response_id": "gen-served",
	})
	assert.Nil(t, lines[1]["failure_class"])
	event := findLogEvent(t, logs, "rayline_arc_refusal_reserve")
	assertFields(t, "reserve event", event, map[string]interface{}{
		"outcome": "reserved", "refused_model": reserveClaudeArm, "served_model": reserveGLMArm,
		"signal": "content_filter", "attempt": 1,
	})
}

// A refusal after text reached the client is passed through: nothing is
// re-served, and the turn fails as a refusal as before.
func TestRefusalAfterTextIsPassedThrough(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)

	for _, chunk := range []string{reserveRoleChunk, reserveTextChunk} {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	assert.False(t, router.holdingForRefusalReserve(ctx), "text did not release the hold")
	sent, _ := clientBytes(stream)
	assert.Contains(t, sent, "partial answer", "the released frames did not reach the client")
	assert.Contains(t, sent, "gen-refused", "the held response start was not released with the text")
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	assert.False(t, ctx.HostRetry.started())
	assert.Empty(t, upstream.calls())
	assert.Equal(t, turnFailureRefusal, ctx.ResponseFailureClass)
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.NotContains(t, sent, refusalAttemptsCommentPrefix, "a turn that was not re-served carried an attempt record")
}

// With no re-serve left, the last refusal reaches the client.
func TestRefusalWithNoReserveLeftIsPassedThrough(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	ctx.RefusalReserve.maxReserves = 0

	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRefusalChunk, true), ctx))
	assert.False(t, ctx.HostRetry.started())
	assert.Empty(t, upstream.calls())
	assert.Equal(t, turnFailureRefusal, ctx.ResponseFailureClass)
	sent, ended := clientBytes(stream)
	assert.True(t, ended)
	assert.Contains(t, sent, "gen-refused", "the held frames were not released with the refusal")
}

// The dev fault refuses the first attempt at its first visible content,
// which the client is never sent.
func TestRefusalPreludeFaultReservesTheFirstAttempt(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{reserveGLMHost: "content"})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	ctx.InjectedFault = headers.FaultRefusalPrelude

	for _, chunk := range []string{reserveRoleChunk, reserveTextChunk} {
		require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(chunk, false), ctx))
	}
	require.True(t, ctx.HostRetry.started(), "the fault did not re-serve")
	drainSpliced(t, router, ctx, stream)
	sent, _ := clientBytes(stream)
	assert.NotContains(t, sent, "partial answer")
	assert.Contains(t, sent, "served on "+reserveGLMHost)
	assert.Equal(t, "fault_injection", attemptsRecord(t, sent)["attempts"].([]interface{})[0].(map[string]interface{})["signal"])
}

// A held turn's client gets a keepalive on the loop's clock.
func TestHeldTurnIsOwedKeepalives(t *testing.T) {
	upstream := newHostUpstream(t, map[string]string{})
	router := reserveRouter(upstream.server.URL)
	stream := NewMockStream(nil)
	ctx, _ := startReserveTurn(t, router)
	require.NoError(t, router.handleProcessRequest(stream, fullDuplexResponseBodyRequest(reserveRoleChunk, false), ctx))
	wait := router.refusalReserveKeepAliveWait(ctx)
	assert.True(t, wait > 0 && wait <= refusalReserveKeepAlive, "wait %v", wait)
	assert.False(t, router.refusalReserveKeepAliveDue(ctx), "a keepalive is due right after a held frame's comment")
	ctx.RefusalReserve.lastKeepAlive = time.Now().Add(-refusalReserveKeepAlive)
	assert.True(t, router.refusalReserveKeepAliveDue(ctx))
	require.NoError(t, sendRefusalReserveKeepAlive(stream, ctx))
	sent, ended := clientBytes(stream)
	assert.False(t, ended)
	assert.True(t, strings.HasSuffix(sent, string(hostRetryHoldFrame)))
}
